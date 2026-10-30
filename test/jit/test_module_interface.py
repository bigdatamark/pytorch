# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] TorchScript tests for module interfaces (torch.jit.interface on nn.Module
# classes): calling methods through an interface, interface subtyping and contravariance,
# inheritance errors, swapping interface-typed submodules on scripted modules (including
# incompatible swaps), swapping with ScriptModules, freezing modules that hold interface
# submodules (with and without mutation, set-attribute and fork), and module API use on
# interface-typed attributes. Also defines two shared implementation modules, OrigModule and
# NewModule. This module is imported by the misc test file so the test class runs there as well.
# Import time: appends the test directory to sys.path.

import os
import sys
from typing import Any, List

import torch
import torch.nn as nn
from torch import Tensor
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class OrigModule(nn.Module):
    # [CLASS DOC] HELPER: Reference implementation module with methods one, two and forward.
    # Forward returns the input plus the result of method one applied to the input twice, plus
    # one, i.e. three times the input plus two.
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the sum of both inputs plus one.
        return inp1 + inp2 + 1

    def two(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the input plus two.
        return input + 2

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the input plus method one applied to the input twice,
        # plus one.
        return input + self.one(input, input) + 1


class NewModule(nn.Module):
    # [CLASS DOC] HELPER: Alternative implementation providing one and forward, compatible
    # with the interface that needs those two methods. Forward calls one with input and input
    # plus one.
    def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns the product of the inputs plus one.
        return inp1 * inp2 + 1

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] HELPER: Returns method one applied to the input and the input plus one,
        # i.e. input times input plus one, plus one.
        return self.one(input, input + 1)


class TestModuleInterface(JitTestCase):
    # [CLASS DOC] JitTestCase for module interfaces: interface calls, subtyping, swapping
    # submodules, freezing, and error reporting.
    def test_not_submodule_interface_call(self):
        # [FUNCTION DOC] Scripting a module whose interface-typed submodule calls a method not
        # declared in the interface must raise RuntimeError saying the object has no such
        # attribute or method, highlighting the offending call expression.
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Interface declaring only the method one.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestNotModuleInterfaceCall(nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule,
            # initially OrigModule.
            proxy_mod: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the interface-typed submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            # [FUNCTION DOC] HELPER: Calls the submodule method two, which the interface does
            # not declare; scripting must fail.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, input: Tensor) -> Tensor:
                return self.proxy_mod.two(input)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", "self.proxy_mod.two"
        ):
            torch.jit.script(TestNotModuleInterfaceCall())

    def test_module_interface(self):
        # [FUNCTION DOC] Checks module interface and class interface usage. Two implementations
        # are scripted and passed in a list to functions that call interface methods;
        # checkScript compares scripted and eager. Calling an exported method not in the
        # interface on an interface-typed argument must fail with an error about a missing
        # attribute or method, highlighting the call.
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            # [CLASS DOC] HELPER: Module interface declaring one, two and forward.
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def two(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        @torch.jit.interface
        class OneTwoClass:
            # [CLASS DOC] HELPER: Class (not module) interface declaring one and two.
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def two(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class FooMod(nn.Module):
            # [CLASS DOC] HELPER: Implementation using addition and doubling.
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of the inputs.
                return x + y

            def two(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns twice the input.
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Doubles the input, then adds the original input.
                return self.one(self.two(x), x)

        class BarMod(nn.Module):
            # [CLASS DOC] HELPER: Implementation using product and reciprocal, plus an exported
            # extra method.
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the product of the inputs.
                return x * y

            def two(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns two divided by the input.
                return 2 / x

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Applies method one to the input twice, then method two.
                return self.two(self.one(x, x))

            @torch.jit.export
            def forward2(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Exported method (not in the interface); same as forward
                # plus one.
                return self.two(self.one(x, x)) + 1

        make_global(OneTwoModule, OneTwoClass)

        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Sums the forward results of the first two modules in
            # the list.
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        def use_class_interface(mod_list: List[OneTwoClass], x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Sums the two of the first element and the one of the second
            # element, both class-interface calls.
            return mod_list[0].two(x) + mod_list[1].one(x, x)

        scripted_foo_mod = torch.jit.script(FooMod())
        scripted_bar_mod = torch.jit.script(BarMod())
        self.checkScript(
            use_module_interface,
            (
                [scripted_foo_mod, scripted_bar_mod],
                torch.rand(3, 4),
            ),
        )
        self.checkScript(
            use_class_interface,
            (
                [scripted_foo_mod, scripted_bar_mod],
                torch.rand(3, 4),
            ),
        )

        # [FUNCTION DOC] HELPER: Calls forward2 on an argument typed as the module interface,
        # which does not declare it; compiling must fail.
        # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above the
        #     def line.
        def call_module_interface_on_other_method(
            mod_interface: OneTwoModule, x: Tensor
        ) -> Tensor:
            return mod_interface.forward2(x)

        # ensure error out when we call the module on the method other than the interface specified.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "object has no attribute or method", "mod_interface.forward2"
        ):
            self.checkScript(
                call_module_interface_on_other_method,
                (
                    scripted_bar_mod,
                    torch.rand(3, 4),
                ),
            )

    def test_module_doc_string(self):
        # [FUNCTION DOC] Checks that interface method bodies using type comments, docstring-
        # style string statements and a pass statement are accepted, and that a module
        # calling an interface-typed submodule is scripted correctly (compared with eager
        # via checkModule).
        @torch.jit.interface
        class TestInterface(nn.Module):
            # [CLASS DOC] HELPER: Module interface whose methods use type comment signatures.
            # [FUNCTION DOC] HELPER: Interface method declaration with a type comment signature.
            # KNOWN ISSUE: the body begins with a type comment, so this doc block sits above the
            #     def line.
            def one(self, inp1, inp2):
                # type: (Tensor, Tensor) -> Tensor
                pass

            # [FUNCTION DOC] HELPER: Interface forward declaration with a type comment
            # signature, string statements and a pass statement.
            # KNOWN ISSUE: the body begins with a type comment, so this doc block sits above the
            #     def line.
            def forward(self, input):
                # type: (Tensor) -> Tensor
                r"""stuff 1"""
                r"""stuff 2"""
                pass  # noqa: PIE790
                r"""stuff 3"""

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule initially
            # an OrigModule.
            proxy_mod: TestInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the interface-typed submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            # [FUNCTION DOC] HELPER: Forwards the input through the submodule forward.
            # KNOWN ISSUE: the body begins with a type comment, so this doc block sits above the
            #     def line.
            def forward(self, input):
                # type: (Tensor) -> Tensor
                return self.proxy_mod.forward(input)

        input = torch.randn(3, 4)
        self.checkModule(TestModule(), (input,))

    def test_module_interface_subtype(self):
        # [FUNCTION DOC] Checks which objects are subtypes of a module interface. A scripted
        # class instance is rejected (only script modules may be subtypes); a module with an
        # incompatible method signature is rejected with an error about incompatibility.
        # Arguments may be contravariant and return types covariant: modules with Any input or
        # tensor output are accepted where an Any to Any or tensor to Any interface is expected.
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            # [CLASS DOC] HELPER: Module interface declaring one, two and forward.
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def two(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        make_global(OneTwoModule)

        @torch.jit.script
        def as_module_interface(x: OneTwoModule) -> OneTwoModule:
            # [FUNCTION DOC] HELPER: Scripted identity function typed to accept and return
            # the interface.
            return x

        @torch.jit.script
        class Foo:
            # [CLASS DOC] HELPER: Scripted plain class with the same methods as the interface.
            def one(self, x: Tensor, y: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of the inputs.
                return x + y

            def two(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns twice the input.
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Doubles the input and adds the original.
                return self.one(self.two(x), x)

        # check class object is not a subtype of module interface
        with self.assertRaisesRegex(
            RuntimeError, "ScriptModule class can be subtype of module interface"
        ):
            as_module_interface(Foo())

        class WrongMod(nn.Module):
            # [CLASS DOC] HELPER: Module whose two method uses integers, so it does not match
            # the interface.
            def two(self, x: int) -> int:
                # [FUNCTION DOC] HELPER: Returns twice the integer.
                return 2 * x

            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the input plus random values with a width
                # obtained from two.
                return x + torch.randn(3, self.two(3))

        scripted_wrong_mod = torch.jit.script(WrongMod())

        # wrong module that is not compatible with module interface
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            as_module_interface(scripted_wrong_mod)

        # Check that interface implementations can be contravariant in argument types and covariant in return type.
        @torch.jit.interface
        class TensorToAny(nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes a tensor and returns Any.
            def forward(self, input: torch.Tensor) -> Any:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        make_global(TensorToAny)

        @torch.jit.script
        def as_tensor_to_any(x: TensorToAny) -> TensorToAny:
            # [FUNCTION DOC] HELPER: Scripted identity function typed with the tensor to
            # Any interface.
            return x

        @torch.jit.interface
        class AnyToAny(nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes Any and returns Any.
            def forward(self, input: Any) -> Any:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        make_global(AnyToAny)

        @torch.jit.script
        def as_any_to_any(x: AnyToAny) -> AnyToAny:
            # [FUNCTION DOC] HELPER: Scripted identity function typed with the Any to
            # Any interface.
            return x

        class TensorToAnyImplA(nn.Module):
            # [CLASS DOC] HELPER: Module returning Any, accepted for the tensor to
            # Any interface.
            def forward(self, input: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns its input.
                return input

        class TensorToAnyImplB(nn.Module):
            # [CLASS DOC] HELPER: Module returning a tensor, accepted for the tensor to
            # Any interface.
            def forward(self, input: Any) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns a one-element tensor of value one.
                return torch.tensor([1])

        class AnyToAnyImpl(nn.Module):
            # [CLASS DOC] HELPER: Module taking Any and returning a tensor, accepted for the
            # Any interface.
            def forward(self, input: Any) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns a one-element tensor of value one.
                return torch.tensor([1])

        as_tensor_to_any(torch.jit.script(TensorToAnyImplA()))
        as_tensor_to_any(torch.jit.script(TensorToAnyImplB()))
        as_any_to_any(torch.jit.script(AnyToAnyImpl()))

    def test_module_interface_inheritance(self):
        # [FUNCTION DOC] Checks that declaring a module interface by inheriting from a concrete
        # module class raises RuntimeError saying inheritance is not supported.
        with self.assertRaisesRegex(
            RuntimeError, "does not support inheritance yet. Please directly"
        ):

            @torch.jit.interface
            class InheritMod(nn.ReLU):
                # [CLASS DOC] HELPER: Attempted interface subclassing ReLU; declaring it
                # must fail.
                def three(self, x: Tensor) -> Tensor:
                    # [FUNCTION DOC] HELPER: Method of the invalid interface; returns three
                    # times the input.
                    return 3 * x

    def test_module_swap(self):
        # [FUNCTION DOC] Scripts a module with an interface-typed submodule; checks output with
        # OrigModule is three times input plus two. Swaps in a scripted NewModule (same
        # interface) and checks the new output. Assigning an unscripted module raises
        # RuntimeError about non-scripted modules.
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Interface declaring one and forward.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule.
            proxy_mod: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the interface-typed submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Forwards the input through the submodule forward.
                return self.proxy_mod.forward(input)

        scripted_mod = torch.jit.script(TestModule())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        # module swap with module that have the same interface
        scripted_mod.proxy_mod = torch.jit.script(NewModule())
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

        # module swap with non-scripted module should throw error
        with self.assertRaisesRegex(
            RuntimeError, "a ScriptModule with non-scripted module"
        ):
            scripted_mod.proxy_mod = NewModule()

    def test_module_swap_wrong_module(self):
        # [FUNCTION DOC] Swapping an interface-typed submodule with a scripted module whose
        # forward takes and returns integers must raise RuntimeError saying it is not compatible
        # with the interface.
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Interface declaring one and forward.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class NewModuleWrong(nn.Module):
            # [CLASS DOC] HELPER: Module with an integer forward, incompatible with
            # the interface.
            def forward(self, input: int) -> int:
                # [FUNCTION DOC] HELPER: Returns the integer plus one.
                return input + 1

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule.
            proxy_mod: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the interface-typed submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Forwards the input through the submodule forward.
                return self.proxy_mod.forward(input)

        scripted_mod = torch.jit.script(TestModule())
        # module swap with in-compatible interface
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleWrong())

    def test_module_swap_no_lazy_compile(self):
        # [FUNCTION DOC] Methods that are not exported are not compiled on a scripted module, so
        # swapping in a module whose method one is never reached from forward fails with an
        # incompatibility error. The same module with an explicitly exported method one swaps
        # successfully and gives the input plus one.
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Interface declaring one and forward.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule.
            proxy_mod: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the interface-typed submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Forwards the input through the submodule forward.
                return self.proxy_mod.forward(input)

        class NewModuleMethodNotLazyCompile(nn.Module):
            # [CLASS DOC] HELPER: Module with method one that forward does not call, so one is
            # not compiled.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the product of inputs plus one.
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return input + 1

        scripted_mod = torch.jit.script(TestModule())
        # module swap with module that have the same interface, but the method not get
        # lazily compiled from forward, user need to export it explicitly for swap to work
        with self.assertRaisesRegex(RuntimeError, "is not compatible with interface"):
            scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodNotLazyCompile())

        class NewModuleMethodManualExport(nn.Module):
            # [CLASS DOC] HELPER: Same as the previous module but method one is
            # exported explicitly.
            @torch.jit.export
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Exported method returning the product of inputs
                # plus one.
                return inp1 * inp2 + 1

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the input plus one.
                return input + 1

        scripted_mod.proxy_mod = torch.jit.script(NewModuleMethodManualExport())
        input = torch.randn(3, 4)
        self.assertEqual(scripted_mod(input), input + 1)

    def test_module_swap_no_module_interface(self):
        # [FUNCTION DOC] Swapping a submodule that has no interface type: replacing it with
        # another scripted OrigModule (same JIT type) works; replacing it with a NewModule
        # raises RuntimeError about expecting the original type for the field.
        # test module swapping with no module interface
        class TestNoModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Module with a plain OrigModule submodule (no
            # interface annotation).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Calls the submodule on the input.
                return self.proxy_mod(input)

        scripted_no_module_interface = torch.jit.script(TestNoModuleInterface())
        # proxy mod is swapped with the new ScriptModule that share the same JIT type, should succeed.
        scripted_no_module_interface.proxy_mod = torch.jit.script(OrigModule())
        # proxy_mod is neither a module interface or have the same JIT type, should fail
        with self.assertRaisesRegex(
            RuntimeError,
            r"Expected a value of type '__torch__.jit.test_module_interface.OrigModule \(.*\)' "
            + r"for field 'proxy_mod', but found '__torch__.jit.test_module_interface.NewModule \(.*\)'",
        ):
            scripted_no_module_interface.proxy_mod = torch.jit.script(NewModule())

    def test_script_module_as_interface_swap(self):
        # [FUNCTION DOC] Same swapping flow as the module swap test, but the submodule is a
        # ScriptModule and the replacement ScriptModule is assigned directly without scripting;
        # checks outputs before and after the swap.
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Interface declaring one and forward.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class OrigScriptModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule equivalent of the reference module.
            @torch.jit.script_method
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Script method returning the sum of inputs plus one.
                return inp1 + inp2 + 1

            @torch.jit.script_method
            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Script method returning the input plus method one applied
                # to the input twice, plus one.
                return input + self.one(input, input) + 1

        class NewScriptModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule equivalent of the alternative implementation.
            @torch.jit.script_method
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Script method returning the product of inputs plus one.
                return inp1 * inp2 + 1

            @torch.jit.script_method
            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Script method applying method one to the input and input plus one.
                return self.one(input, input + 1)

        class TestNNModuleWithScriptModule(nn.Module):
            # [CLASS DOC] HELPER: nn.Module with an interface-typed ScriptModule submodule.
            proxy_mod: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigScriptModule as the interface-
                # typed submodule.
                super().__init__()
                self.proxy_mod = OrigScriptModule()

            def forward(self, input: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Forwards the input through the submodule forward.
                return self.proxy_mod.forward(input)

        input = torch.randn(3, 4)
        scripted_mod = torch.jit.script(TestNNModuleWithScriptModule())
        self.assertEqual(scripted_mod(input), 3 * input + 2)

        scripted_mod.proxy_mod = NewScriptModule()
        self.assertEqual(scripted_mod(input), input * (input + 1) + 1)

    # The call to forward of proxy_mod cannot be inlined. Making sure
    # Freezing is throwing an error for now.
    def test_freeze_module_with_interface(self):
        # [FUNCTION DOC] Freezes a scripted module with an interface-typed submodule. Plain
        # freezing is called first (the comment above says the interface call cannot be
        # inlined), then freezing with interface freezing enabled; the frozen module output must
        # equal the scripted output.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute b of twenty, returned
            # by forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the attribute b as twenty.
                super().__init__()
                self.b = 20

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the attribute b.
                return self.b

        class OrigMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute a of zero, returned
            # by forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the attribute a as zero.
                super().__init__()
                self.a = 0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the attribute a.
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes a tensor and returns an int.
            def forward(self, x: Tensor) -> int:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule and a
            # plain submodule.
            proxy_mod: ModInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores OrigMod as the interface submodule and SubModule
                # as sub.
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()  # folded

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the interface submodule output and
                # the sub output.
                return self.proxy_mod(x) + self.sub(x)

        m = torch.jit.script(TestModule())
        m.eval()
        mf = torch._C._freeze_module(m._c)
        # Assume interface has no aliasing
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)
        input = torch.tensor([1])
        out_s = m.forward(input)
        out_f = mf.forward(input)
        self.assertEqual(out_s, out_f)

    def test_freeze_module_with_setattr_in_interface(self):
        # [FUNCTION DOC] Freezing with interface freezing enabled must succeed when a
        # submodule's forward mutates its own integer attribute and the interface attribute is
        # reassigned to that submodule. The test only checks that freezing does not raise.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward increments its integer attribute b and
            # returns it, with an exported getter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the attribute b as twenty.
                super().__init__()
                self.b = 20

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds two to the attribute b in place and returns it.
                self.b += 2
                return self.b

            @torch.jit.export
            def getb(self, x):
                # [FUNCTION DOC] HELPER: Exported method returning the attribute b.
                return self.b

        class OrigMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning an integer attribute a of zero.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the attribute a as zero.
                super().__init__()
                self.a = 0

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the attribute a.
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes a tensor and returns an int.
            def forward(self, x: Tensor) -> int:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule and a
            # plain submodule.
            proxy_mod: ModInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores OrigMod as the interface submodule and SubModule
                # as sub.
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the interface submodule output and the sub
                # getter output.
                return self.proxy_mod(x) + self.sub.getb(x)

        m = torch.jit.script(TestModule())
        m.proxy_mod = m.sub
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_freeze_module_with_inplace_mutation_in_interface(self):
        # [FUNCTION DOC] Like the previous test, but the submodule's attribute is a tensor
        # modified in place, and the submodule's tensor is shared with the interface module
        # before freezing. The test only checks that freezing with interface freezing enabled
        # does not raise.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a one-element tensor attribute whose forward adds
            # two in place.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor attribute b as 1.5.
                super().__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds two in place to the first element of b and
                # returns b.
                self.b[0] += 2
                return self.b

            @torch.jit.export
            def getb(self, x):
                # [FUNCTION DOC] HELPER: Exported method returning b.
                return self.b

        class OrigMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a one-element tensor attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor attribute a as 0.5.
                super().__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the attribute a.
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes and returns a tensor.
            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule and a
            # plain submodule.
            proxy_mod: ModInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores OrigMod as the interface submodule and SubModule
                # as sub.
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Sums the first elements of the interface submodule
                # output and the sub getter output.
                y = self.proxy_mod(x)
                z = self.sub.getb(x)
                return y[0] + z[0]

        m = torch.jit.script(TestModule())
        m.proxy_mod = m.sub
        m.sub.b = m.proxy_mod.b
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_freeze_module_with_mutated_interface(self):
        # [FUNCTION DOC] Freezing with interface freezing enabled must raise RuntimeError saying
        # that set-attribute on an interface type is unsupported, because the scripted forward
        # reassigns the interface-typed attribute.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor attribute b and an exported getter.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor attribute b as 1.5.
                super().__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns b.
                return self.b

            @torch.jit.export
            def getb(self, x):
                # [FUNCTION DOC] HELPER: Exported method returning b.
                return self.b

        class OrigMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a one-element tensor attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor attribute a as 0.5.
                super().__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a.
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes and returns a tensor.
            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule and a plain
            # submodule; its forward reassigns the interface attribute to the plain submodule.
            proxy_mod: ModInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores OrigMod as the interface submodule and SubModule
                # as sub.
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Reassigns the interface attribute to the plain
                # submodule, then sums the first elements of both calls.
                self.proxy_mod = self.sub
                y = self.proxy_mod(x)
                z = self.sub.getb(x)
                return y[0] + z[0]

        m = torch.jit.script(TestModule())
        m.eval()
        with self.assertRaisesRegex(
            RuntimeError, "Freezing does not support SetAttr on an interface type."
        ):
            mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_freeze_module_with_interface_and_fork(self):
        # [FUNCTION DOC] Freezing with interface freezing enabled for a module that calls an
        # interface submodule and a plain submodule inside a forked task and in the main task.
        # Checks freezing does not raise.
        class SubModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a tensor attribute whose forward adds an in-
            # place increment.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor attribute b as 1.5.
                super().__init__()
                self.b = torch.tensor([1.5])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds 3.2 in place to the first element of b and
                # returns b.
                self.b[0] += 3.2
                return self.b

        class OrigMod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module returning a one-element tensor attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor attribute a as 0.5.
                super().__init__()
                self.a = torch.tensor([0.5])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a.
                return self.a

        @torch.jit.interface
        class ModInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Interface whose forward takes and returns a tensor.
            def forward(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule and a
            # plain submodule.
            proxy_mod: ModInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores OrigMod as the interface submodule and SubModule
                # as sub.
                super().__init__()
                self.proxy_mod = OrigMod()
                self.sub = SubModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the sum of the interface submodule output and
                # the sub output.
                y = self.proxy_mod(x)
                z = self.sub(x)
                return y + z

        class MainModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Wraps TestModule and calls its forward in a forked task as
            # well as directly.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the wrapped TestModule.
                super().__init__()
                self.test = TestModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Forks the wrapped forward, calls the module directly,
                # waits for the fork and returns the sum of both results.
                fut = torch.jit._fork(self.test.forward, x)
                y = self.test(x)
                z = torch.jit._wait(fut)
                return y + z

        m = torch.jit.script(MainModule())
        m.eval()
        mf = torch._C._freeze_module(m._c, freezeInterfaces=True)

    def test_module_apis_interface(self):
        # [FUNCTION DOC] Scripting a module with an interface-typed attribute and an
        # exported method that iterates its submodules must fail with an error containing
        # could not compile.
        @torch.jit.interface
        class ModuleInterface(nn.Module):
            # [CLASS DOC] HELPER: Interface declaring only method one.
            def one(self, inp1: Tensor, inp2: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration; placeholder body.
                pass

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module with an interface-typed submodule; exported method
            # iterates modules.
            proxy_mod: ModuleInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an OrigModule as the interface-typed submodule.
                super().__init__()
                self.proxy_mod = OrigModule()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Returns the input times two.
                return input * 2

            @torch.jit.export
            def method(self, input):
                # [FUNCTION DOC] HELPER: Exported method that applies every module returned
                # by the module iterator (including self) to the input in turn and returns
                # the result.
                for module in self.modules():
                    input = module(input)
                return input

        with self.assertRaisesRegex(Exception, "Could not compile"):
            scripted_mod = torch.jit.script(TestModule())


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
