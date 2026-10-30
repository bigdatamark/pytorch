# Owner(s): ["oncall: jit"]

# [MODULE DOC] TorchScript tests for module containers (Sequential, ModuleList, ModuleDict,
# ParameterList, ParameterDict) and related module features: custom container subclasses with
# their own forward, indexing with literal and non-literal indices, special-method generation
# (length, iteration, getitem, contains), typed annotations of container items via module
# interfaces, properties, and reconstructing a module in place. Several tests also check the
# error messages and source highlighting for unsupported indexing. Import time: appends the test
# directory to sys.path.
import os
import sys
from collections import OrderedDict
from typing import Any, List, Tuple

import torch
import torch.nn as nn
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestModuleContainers(JitTestCase):
    # [CLASS DOC] JitTestCase with container-related scripting tests; most use checkModule to
    # compare scripted and eager behavior, and some assert on compile errors.
    def test_sequential_intermediary_types(self):
        # [FUNCTION DOC] Scripts a Sequential whose first module adds three and whose second
        # returns a dictionary; checks scripted equals eager, i.e. the intermediate types need
        # not match.
        class A(torch.nn.Module):
            # [CLASS DOC] HELPER: Module adding three to its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus three.
                return x + 3

        class B(torch.nn.Module):
            # [CLASS DOC] HELPER: Module wrapping the input in a one-entry dictionary.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a dictionary mapping the string key to
                # the input.
                return {"1": x}

        class C(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a Sequential of the adding and dictionary-
            # returning modules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the sequential container from instances of
                # both modules.
                super().__init__()
                self.foo = torch.nn.Sequential(A(), B())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the sequential container on the input and returns
                # its output.
                return self.foo(x)

        self.checkModule(C(), (torch.tensor(1),))

    def test_moduledict(self):
        # [FUNCTION DOC] Checks scripting of ModuleDict iteration (keys, items, values,
        # enumerate, zip) for a base module and a subclass, with each possible skipped name
        # including one that is absent. Compares scripted to eager results via checkModule.
        class Inner(torch.nn.Module):
            # [CLASS DOC] HELPER: Adds ten to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus ten.
                return x + 10

        class Inner2(torch.nn.Module):
            # [CLASS DOC] HELPER: Doubles the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input times two.
                return x * 2

        class Inner3(torch.nn.Module):
            # [CLASS DOC] HELPER: Subtracts four then triples.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns three times the input minus four.
                return (x - 4) * 3

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a ModuleDict of the three inner modules ordered one,
            # two, three; forward iterates it in several ways.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the ModuleDict from an ordered mapping of the
                # three modules.
                super().__init__()
                modules = OrderedDict(
                    [
                        ("one", Inner()),
                        ("two", Inner2()),
                        ("three", Inner3()),
                    ]
                )
                self.moduledict = nn.ModuleDict(modules)

            # [FUNCTION DOC] HELPER: Collects key names by iterating keys, then applies each
            # module whose name is not the skipped name (recording names and outputs), applies
            # all modules via values, and collects keys again; returns the final tensor and the
            # list of names.
            # KNOWN ISSUE: the body begins with a type comment, so this doc block sits above the
            #     def line.
            def forward(self, x, skip_name):
                # type: (Tensor, str)
                names = torch.jit.annotate(List[str], [])
                values = []
                for name in self.moduledict:
                    names.append(name)

                for name, mod in self.moduledict.items():
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)

                for mod in self.moduledict.values():
                    x = mod(x)
                    values.append(x)

                for key in self.moduledict:
                    names.append(key)

                return x, names

        class M2(M):
            # [CLASS DOC] HELPER: Variant of M whose forward also tracks an index counter via
            # enumerate and a second tensor run through each module twice via zip.
            # [FUNCTION DOC] HELPER: Same traversals as the base class but with enumerate over
            # items, values and keys, accumulating loop indices in a counter, plus a zip loop
            # (it reuses one loop variable name for both entries and adds the leftover index)
            # that applies each module twice to a copy of the input. Returns the tensor, the
            # copy, the names and the counter.
            # KNOWN ISSUE: the body begins with a type comment, so this doc block sits above the
            #     def line.
            def forward(self, x, skip_name):
                # type: (Tensor, str)
                names = torch.jit.annotate(List[str], [])
                values = []
                x2 = x
                iter = 0
                for name in self.moduledict:
                    names.append(name)

                for i, (name, mod) in enumerate(self.moduledict.items()):
                    iter += i
                    if name != skip_name:
                        names.append(name)
                        x = mod(x)
                        values.append(x)

                for i, mod in enumerate(self.moduledict.values()):
                    iter += i
                    x = mod(x)
                    values.append(x)

                for i, key in enumerate(self.moduledict.keys()):
                    iter += i
                    names.append(key)

                for mod, mod in zip(self.moduledict.values(), self.moduledict.values()):
                    iter += i
                    x2 = mod(mod(x2))

                return x, x2, names, iter

        for name in ["", "one", "two", "three"]:
            inp = torch.tensor(1)
            self.checkModule(M(), (inp, name))
            self.checkModule(M2(), (inp, name))

    def test_custom_container_forward(self):
        # [FUNCTION DOC] Checks that subclasses of Sequential, ModuleList and ModuleDict that
        # define their own forward (adding three, iterating, subtracting five) can be scripted
        # and match eager results.
        class Inner(torch.nn.Module):
            # [CLASS DOC] HELPER: Adds ten to the input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input plus ten.
                return x + 10

        class CustomSequential(nn.Sequential):
            # [CLASS DOC] HELPER: Sequential of relu and the add-ten module with a
            # custom forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base Sequential with a relu and the
                # add-ten module.
                super().__init__(nn.ReLU(), Inner())

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds three, applies each contained module in order,
                # subtracts five.
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5

        self.checkModule(CustomSequential(), (torch.tensor(0.5),))

        class CustomModuleList(nn.ModuleList):
            # [CLASS DOC] HELPER: ModuleList of relu and the add-ten module with a
            # custom forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base ModuleList with a relu and the
                # add-ten module.
                super().__init__([nn.ReLU(), Inner()])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds three, applies each contained module in order,
                # subtracts five.
                x = x + 3
                for mod in self:
                    x = mod(x)
                return x - 5

        self.checkModule(CustomModuleList(), (torch.tensor(0.5),))

        class CustomModuleDict(nn.ModuleDict):
            # [CLASS DOC] HELPER: ModuleDict of add-ten, relu and add-ten with a custom forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the base ModuleDict with an ordered mapping
                # of three modules.
                super().__init__(
                    OrderedDict(
                        [
                            ("one", Inner()),
                            ("two", nn.ReLU()),
                            ("three", Inner()),
                        ]
                    )
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds three, applies each module while recording its
                # name, and returns the names list and the result minus five.
                x = x + 3
                names = torch.jit.annotate(List[str], [])
                for name, mod in self.items():
                    x = mod(x)
                    names.append(name)
                return names, x - 5

        self.checkModule(CustomModuleDict(), (torch.tensor(0.5),))

    def test_script_module_list_sequential(self):
        # [FUNCTION DOC] Builds a ScriptModule that applies a Sequential in a loop, with
        # optimized execution disabled, and checks it survives an export and import round trip.
        class M(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: ScriptModule applying each module of a given list-like
            # container in order.
            def __init__(self, mod_list):
                # [FUNCTION DOC] HELPER: Stores the given container of modules.
                super().__init__()
                self.mods = mod_list

            @torch.jit.script_method
            def forward(self, v):
                # [FUNCTION DOC] HELPER: Script method. Passes the value through each contained
                # module in turn.
                for m in self.mods:
                    v = m(v)
                return v

        with torch.jit.optimized_execution(False):
            m = M(nn.Sequential(nn.ReLU()))
            self.assertExportImportModule(m, (torch.randn(2, 2),))

    def test_script_modulelist_index(self):
        # [FUNCTION DOC] Checks ModuleList indexing with literal positive and negative indices,
        # both via an explicit forward call and via call syntax. Also checks error cases when
        # scripting: an index out of range raises with a message about the index and highlights
        # the indexing expression; indexing with a non-literal variable raises errors mentioning
        # enumeration support or a non-literal index.
        class Sub(torch.nn.Module):
            # [CLASS DOC] HELPER: Subtracts a stored integer from the input.
            def __init__(self, i):
                # [FUNCTION DOC] HELPER: Stores the integer offset.
                super().__init__()
                self.i = i

            def forward(self, thing):
                # [FUNCTION DOC] HELPER: Returns the input minus the stored offset.
                return thing - self.i

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds ten Sub modules; forward indexes with literals four,
            # minus one and minus nine using explicit forward calls.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a ModuleList of ten Sub modules with offsets
                # zero to nine.
                super().__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                # [FUNCTION DOC] HELPER: Applies modules at positions four, last and ninth from
                # last in turn.
                v = self.mods[4].forward(v)
                v = self.mods[-1].forward(v)
                v = self.mods[-9].forward(v)
                return v

        x = torch.tensor(1)
        self.checkModule(M(), (x,))

        class MForward(torch.nn.Module):
            # [CLASS DOC] HELPER: Same as M but indexes modules and calls them directly.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a ModuleList of ten Sub modules with offsets
                # zero to nine.
                super().__init__()
                self.mods = nn.ModuleList([Sub(i) for i in range(10)])

            def forward(self, v):
                # [FUNCTION DOC] HELPER: Applies modules at positions four, last and ninth from
                # last in turn.
                v = self.mods[4](v)
                v = self.mods[-1](v)
                v = self.mods[-9](v)
                return v

        self.checkModule(MForward(), (torch.tensor(1),))

        class M2(M):
            # [CLASS DOC] HELPER: Subclass of M whose forward indexes with minus eleven, out
            # of range.
            # [FUNCTION DOC] HELPER: Indexes the module list with an out-of-range negative
            # literal; scripting must fail with a highlighted error.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, v):
                return self.mods[-11].forward(v)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Index -11 out of range", "self.mods[-11]"
        ):
            torch.jit.script(M2())

        class M3(M):
            # [CLASS DOC] HELPER: Subclass of M whose forward indexes with a local variable.
            # [FUNCTION DOC] HELPER: Sets a local variable to three and indexes the module list
            # with it; scripting must fail with a highlighted error.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, v):
                i = 3
                return self.mods[i].forward(v)

        with self.assertRaisesRegexWithHighlight(
            Exception, "Enumeration is supported", "self.mods[i]"
        ):
            torch.jit.script(M3())

        class M4(M):
            # [CLASS DOC] HELPER: Subclass of M identical in forward to the previous
            # failing variant.
            def forward(self, v):
                # [FUNCTION DOC] HELPER: Sets a local variable to three and indexes the
                # module list with it; scripting must fail with a message that the index is
                # not a literal. Same body as the previous failing case, only the expected
                # message differs.
                i = 3
                return self.mods[i].forward(v)

        with self.assertRaisesRegex(Exception, "will fail because i is not a literal"):
            torch.jit.script(M4())

    def test_module_interface_special_methods(self):
        # [FUNCTION DOC] Checks the special methods generated for scripted subclasses of
        # ModuleList, Sequential and ModuleDict that share a marker base class: getitem, length,
        # iteration, contains, keys, items and values. The asserts live in the scripted forward;
        # checkModule runs it.
        class CustomModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty marker base module class.
            pass

        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):
            # [CLASS DOC] HELPER: ModuleList that also derives from the marker class.
            def __init__(self, modules=None):
                # [FUNCTION DOC] HELPER: Initializes both bases explicitly, passing the modules
                # to the ModuleList.
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleList.__init__(self, modules)

        class CustomSequential(CustomModuleInterface, torch.nn.Sequential):
            # [CLASS DOC] HELPER: Sequential that also derives from the marker class.
            def __init__(self, modules=None):
                # [FUNCTION DOC] HELPER: Initializes both bases explicitly, passing the modules
                # to Sequential.
                CustomModuleInterface.__init__(self)
                torch.nn.Sequential.__init__(self, modules)

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):
            # [CLASS DOC] HELPER: ModuleDict that also derives from the marker class.
            def __init__(self, modules=None):
                # [FUNCTION DOC] HELPER: Initializes both bases explicitly, passing the modules
                # to the ModuleDict.
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds one pre-scripted relu inside each of the three
            # custom containers, and a forward that asserts the container special methods
            # behave as expected.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Scripts a relu up front (to avoid an aliasing
                # issue with identity checks) and places it in a custom list, sequential
                # and dictionary.
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])
                self.sequential = CustomSequential(self.submod)
                self.moduledict = CustomModuleDict({"submod": self.submod})

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Asserts getitem, length, iteration, contains, keys,
                # items and values on the three containers, then returns the input unchanged.
                assert self.modulelist[0] is self.submod, (  # noqa: S101
                    "__getitem__ failing for ModuleList"
                )
                assert len(self.modulelist) == 1, "__len__ failing for ModuleList"  # noqa: S101
                for module in self.modulelist:
                    assert module is self.submod, "__iter__ failing for ModuleList"  # noqa: S101

                assert self.sequential[0] is self.submod, (  # noqa: S101
                    "__getitem__ failing for Sequential"
                )
                assert len(self.sequential) == 1, "__len__ failing for Sequential"  # noqa: S101
                for module in self.sequential:
                    assert module is self.submod, "__iter__ failing for Sequential"  # noqa: S101

                assert self.moduledict["submod"] is self.submod, (  # noqa: S101
                    "__getitem__ failing for ModuleDict"
                )
                assert len(self.moduledict) == 1, "__len__ failing for ModuleDict"  # noqa: S101

                # note: unable to index moduledict with a string variable currently
                i = 0
                for _ in self.moduledict:
                    i += 1
                assert i == len(self.moduledict), "iteration failing for ModuleDict"  # noqa: S101

                assert "submod" in self.moduledict, "__contains__ fails for ModuleDict"  # noqa: S101

                for key in self.moduledict:
                    assert key == "submod", "keys() fails for ModuleDict"  # noqa: S101

                for item in self.moduledict.items():
                    assert item[0] == "submod", "items() fails for ModuleDict"  # noqa: S101
                    assert item[1] is self.submod, "items() fails for ModuleDict"  # noqa: S101

                for value in self.moduledict.values():
                    assert value is self.submod, "values() fails for ModuleDict"  # noqa: S101

                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_special_method_with_override(self):
        # [FUNCTION DOC] Checks that a Python-defined length method on a ModuleList subclass
        # takes precedence over the one the compiler generates: the scripted forward asserts
        # length two although the list has one element.
        class CustomModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty marker base module class.
            pass

        class CustomModuleList(CustomModuleInterface, torch.nn.ModuleList):
            # [CLASS DOC] HELPER: ModuleList subclass overriding the length method.
            def __init__(self, modules=None):
                # [FUNCTION DOC] HELPER: Initializes both bases explicitly.
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleList.__init__(self, modules)

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns two regardless of the real length.
                # this is arbitrary, just to check that the overridden py __len__ from
                # CustomModuleList takes precedence over the automatically generated
                # __len__ added by the jit compiler
                return 2

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a custom list with one pre-scripted relu; forward
            # asserts length two.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Scripts a relu and puts it in the custom list.
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.modulelist = CustomModuleList([self.submod])

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Asserts the container length is two and returns
                # the input.
                assert len(self.modulelist) == 2, "__len__ failing for ModuleList"  # noqa: S101
                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])
        torch.jit.script(m)

    def test_moduledict_getitem(self):
        # [FUNCTION DOC] Checks ModuleDict getitem with literal keys on a module whose forward
        # asserts the retrieved modules are the original scripted relu and tanh objects.
        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding scripted relu and tanh in a ModuleDict.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Scripts relu and tanh and stores them in a ModuleDict
                # under their names.
                super().__init__()
                self.relu = torch.jit.script(torch.nn.ReLU())
                self.tanh = torch.jit.script(torch.nn.Tanh())
                self.moduledict = torch.nn.ModuleDict(
                    {"relu": self.relu, "tanh": self.tanh}
                )

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Asserts literal-key lookups return the same module
                # objects; returns input.
                assert self.moduledict["relu"] is self.relu  # noqa: S101
                assert self.moduledict["tanh"] is self.tanh  # noqa: S101
                return input

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_moduledict_keyerror(self):
        # [FUNCTION DOC] Checks the errors when scripting a ModuleDict lookup with a missing
        # literal key (a key error mentioning the key) and with a variable key (an error that
        # the index must be a string literal). Both check the highlighted source snippet.
        class BadModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward looks up a missing key.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ModuleDict with two None entries.
                super().__init__()
                self.moduledict = torch.nn.ModuleDict({"foo": None, "bar": None})

            # [FUNCTION DOC] HELPER: Looks up a key that is not in the dictionary; scripting
            # must fail.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, input):
                assert self.moduledict["blah"] == "blah", "this is a keyerror"  # noqa: S101

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Key Error, blah", 'self.moduledict["blah"'
        ):
            b = BadModule()
            torch.jit.script(b)

        class AnotherBadModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward indexes the ModuleDict with a variable.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ModuleDict with two None entries.
                super().__init__()
                self.moduledict = torch.nn.ModuleDict({"foo": None, "bar": None})

            # [FUNCTION DOC] HELPER: Looks up a ModuleDict entry through a local string
            # variable; scripting must fail.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, input):
                idx = "blah"
                assert self.moduledict[idx] == "blah", "this is a string literal error"  # noqa: S101

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "Unable to extract string literal index. "
            "ModuleDict indexing is only supported with string literals. "
            "For example, 'i = \"a\"; self.layers\\[i\\]\\(x\\)' will fail "
            "because i is not a literal.",
            "self.moduledict[idx]",
        ):
            b = AnotherBadModule()
            torch.jit.script(b)

    def test_normal_list_attribute_with_modules_error(self):
        # [FUNCTION DOC] Checks scripting a module with an ordinary Python list attribute of
        # modules fails with an error that the list element type cannot be inferred,
        # highlighting the attribute access.
        """
        Test that an attempt to script a module with a regular list attribute
        containing other modules fails with a relevant error message.
        """

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a plain list attribute holding two relu modules.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a plain list of two relu modules.
                super().__init__()
                self.a = [torch.nn.ReLU(), torch.nn.ReLU()]

            # [FUNCTION DOC] HELPER: Returns the length of the list attribute; scripting it
            # must fail.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self):
                return len(self.a)

        error_msg = "Could not infer type of list element: Cannot infer concrete type of torch.nn.Module"
        with self.assertRaisesRegexWithHighlight(RuntimeError, error_msg, "self.a"):
            torch.jit.script(Mod())

    def test_empty_dict_override_contains(self):
        # [FUNCTION DOC] Checks that a ModuleDict subclass with no entries scripts, and that the
        # generated contains check reports a missing key as absent.
        class CustomModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty marker base module class.
            pass

        class CustomModuleDict(CustomModuleInterface, torch.nn.ModuleDict):
            # [CLASS DOC] HELPER: ModuleDict that also derives from the marker class.
            def __init__(self, modules=None):
                # [FUNCTION DOC] HELPER: Initializes both bases explicitly.
                CustomModuleInterface.__init__(self)
                torch.nn.ModuleDict.__init__(self, modules)

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds an empty custom dictionary; forward asserts a key is
            # not contained.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Scripts a relu (kept as attribute) and creates the
                # empty custom dictionary.
                super().__init__()
                # work around aliasing issue for 'is' operator by scripting ReLU up front
                self.submod = torch.jit.script(torch.nn.ReLU())
                self.moduledict = CustomModuleDict()

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Asserts the submodule key is absent from the empty
                # dictionary; returns input.
                assert "submod" not in self.moduledict, (  # noqa: S101
                    "__contains__ fails for ModuleDict"
                )
                return inputs

        m = MyModule()
        self.checkModule(m, [torch.randn(2, 2)])

    def test_typed_module_dict(self):
        # [FUNCTION DOC] Checks typed ModuleDict indexing with a non-literal key through an
        # interface annotation, both for a submodule and for the dictionary itself. A dictionary
        # holding a module that does not implement the interface must fail scripting with an
        # error that the attribute is not of the annotated type, highlighting the lookup.
        """
        Test that a type annotation can be provided for a ModuleDict that allows
        non-static indexing.
        """

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface with a forward taking and returning Any.
            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Interface forward declaration; placeholder body.
                pass

        class ImplementsInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Module satisfying the interface.
            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns the max along dimension zero (values and
                # indices) for tensors, and other inputs unchanged.
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        class DoesNotImplementInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward signature does not match the interface.
            def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns the max along dimension zero for a
                # tensor input.
                return torch.max(inp, dim=0)

        # Test annotation of submodule.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a ModuleDict with one conforming module; forward looks
            # up by variable key into an interface-typed local.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ModuleDict with the conforming module under a
                # module key.
                super().__init__()
                self.d = torch.nn.ModuleDict({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # [FUNCTION DOC] HELPER: Looks up the key, annotates the result as the
                # interface, and calls forward.
                value: ModuleInterface = self.d[key]
                return value.forward(x)

        m = Mod()
        self.checkModule(m, (torch.randn(2, 2), "module"))

        # Test annotation of self.
        class ModDict(torch.nn.ModuleDict):
            # [CLASS DOC] HELPER: ModuleDict subclass with the conforming module, indexing
            # itself by key.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the ModuleDict with the conforming module.
                super().__init__({"module": ImplementsInterface()})

            def forward(self, x: torch.Tensor, key: str) -> Any:
                # [FUNCTION DOC] HELPER: Looks up self by key, annotates as the interface, and
                # calls forward.
                submodule: ModuleInterface = self[key]
                return submodule.forward(x)

        m = ModDict()
        self.checkModule(m, (torch.randn(2, 2), "module"))

        # Test error message thrown when annotated attribute does not comply with the
        # annotation.
        class ModWithWrongAnnotation(torch.nn.ModuleDict):
            # [CLASS DOC] HELPER: ModuleDict subclass whose attribute holds a non-conforming
            # module; scripting must fail.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ModuleDict with the non-conforming module.
                super().__init__()
                self.d = torch.nn.ModuleDict({"module": DoesNotImplementInterface()})

            # [FUNCTION DOC] HELPER: Looks up the attribute by key, annotates as the interface
            # and calls forward.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, x: torch.Tensor, key: str) -> Any:
                submodule: ModuleInterface = self.d[key]
                return submodule.forward(x)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute module is not of annotated type", "self.d[key]"
        ):
            torch.jit.script(ModWithWrongAnnotation())

    def test_typed_module_list(self):
        # [FUNCTION DOC] List counterpart of the typed dictionary test: non-literal integer
        # indexing into a ModuleList through an interface annotation, for a submodule and for a
        # list subclass. A non- conforming module must fail scripting with an error that the
        # attribute is not of the annotated type, highlighting the indexing.
        """
        Test that a type annotation can be provided for a ModuleList that allows
        non-static indexing.
        """

        @torch.jit.interface
        class ModuleInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface with a forward taking and returning Any.
            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Interface forward declaration; placeholder body.
                pass

        class ImplementsInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Module satisfying the interface.
            def forward(self, inp: Any) -> Any:
                # [FUNCTION DOC] HELPER: Returns the max along dimension zero (values and
                # indices) for tensors, and other inputs unchanged.
                if isinstance(inp, torch.Tensor):
                    return torch.max(inp, dim=0)

                return inp

        class DoesNotImplementInterface(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward signature does not match the interface.
            def forward(self, inp: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
                # [FUNCTION DOC] HELPER: Returns the max along dimension zero for a
                # tensor input.
                return torch.max(inp, dim=0)

        # Test annotation of submodule.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Holds a ModuleList with one conforming module; forward indexes
            # by integer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ModuleList with the conforming module.
                super().__init__()
                self.l = torch.nn.ModuleList([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # [FUNCTION DOC] HELPER: Indexes the list, annotates the result as the
                # interface, calls forward.
                value: ModuleInterface = self.l[idx]
                return value.forward(x)

        m = Mod()
        self.checkModule(m, (torch.randn(2, 2), 0))

        # Test annotation of self.
        class ModList(torch.nn.ModuleList):
            # [CLASS DOC] HELPER: ModuleList subclass with the conforming module, indexing
            # itself by integer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initializes the ModuleList with the conforming module.
                super().__init__([ImplementsInterface()])

            def forward(self, x: torch.Tensor, idx: int) -> Any:
                # [FUNCTION DOC] HELPER: Indexes self, annotates as the interface,
                # calls forward.
                submodule: ModuleInterface = self[idx]
                return submodule.forward(x)

        m = ModList()
        self.checkModule(m, (torch.randn(2, 2), 0))

        # Test error message thrown when annotated attribute does not comply with the
        # annotation.
        class ModWithWrongAnnotation(torch.nn.ModuleList):
            # [CLASS DOC] HELPER: ModuleList subclass whose attribute holds a non-conforming
            # module; scripting must fail.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a ModuleList with the non-conforming module.
                super().__init__()
                self.l = torch.nn.ModuleList([DoesNotImplementInterface()])

            # [FUNCTION DOC] HELPER: Indexes the attribute list, annotates as the interface and
            # calls forward.
            # KNOWN ISSUE: asserted by a source highlight check, so this doc block sits above
            #     the def line.
            def forward(self, x: torch.Tensor, idx: int) -> Any:
                submodule: ModuleInterface = self.l[idx]
                return submodule.forward(x)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, r"Attribute 0 is not of annotated type", "self.l[idx]"
        ):
            torch.jit.script(ModWithWrongAnnotation())

    def test_module_properties(self):
        # [FUNCTION DOC] Checks scripting of modules with properties (with and without
        # setters, plus unused-property markers) using positive and negative inputs, and that
        # a property listed as unused is absent on the scripted module (attribute access
        # raises AttributeError).
        class ModuleWithProperties(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a property pair that reads and clamps a stored
            # integer, and properties excluded from scripting (one via the unused-properties
            # list, one via the unused decorator).
            __jit_unused_properties__ = ["ignored_attr"]

            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Stores the integer a.
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                # [FUNCTION DOC] HELPER: Assigns the sum of the arguments through the property
                # setter and returns the property value.
                self.attr = a + b
                return self.attr

            @property
            def attr(self):
                # [FUNCTION DOC] HELPER: Property getter returning the stored integer.
                return self.a

            @property
            def ignored_attr(self):
                # [FUNCTION DOC] HELPER: Property excluded from scripting; returns the sum of a
                # one-element list.
                return sum([self.a])

            @torch.jit.unused
            @property
            def ignored_attr_2(self):
                # [FUNCTION DOC] HELPER: Unused-marked property getter; returns the sum of a
                # one-element list.
                return sum([self.a])

            @ignored_attr_2.setter
            def ignored_attr_2(self, value):
                # [FUNCTION DOC] HELPER: Unused-marked property setter; replaces the stored
                # value with the sum of a one-element list holding it.
                self.a = sum([self.a])

            @attr.setter
            def attr(self, a: int):
                # [FUNCTION DOC] HELPER: Property setter that stores the value if positive and
                # zero otherwise.
                if a > 0:
                    self.a = a
                else:
                    self.a = 0

        class ModuleWithNoSetter(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a read-only property.
            def __init__(self, a: int):
                # [FUNCTION DOC] HELPER: Stores the integer a.
                super().__init__()
                self.a = a

            def forward(self, a: int, b: int):
                # [FUNCTION DOC] HELPER: Evaluates the property plus the two arguments and
                # discards the result; returns nothing.
                self.attr + a + b

            @property
            def attr(self):
                # [FUNCTION DOC] HELPER: Property getter returning the stored integer plus one.
                return self.a + 1

        self.checkModule(
            ModuleWithProperties(5),
            (
                5,
                6,
            ),
        )
        self.checkModule(
            ModuleWithProperties(5),
            (
                -5,
                -6,
            ),
        )
        self.checkModule(
            ModuleWithNoSetter(5),
            (
                5,
                6,
            ),
        )
        self.checkModule(
            ModuleWithNoSetter(5),
            (
                -5,
                -6,
            ),
        )

        mod = ModuleWithProperties(3)
        scripted_mod = torch.jit.script(mod)

        with self.assertRaisesRegex(AttributeError, "has no attribute"):
            scripted_mod.ignored_attr

    def test_module_inplace_construct(self):
        # [FUNCTION DOC] Scripts two different modules, then reconstructs the second from the
        # first's underlying compiled module. Checks both produce the same output on a random
        # input and that the ignored method of the second module is still callable and returns
        # its input.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer, an integer attribute, a parameter,
            # a method and an unused method.
            def __init__(self, start: int):
                # [FUNCTION DOC] HELPER: Creates a 3x3 linear layer, stores the start attribute
                # and a scalar float parameter.
                super().__init__()
                self.linear = nn.Linear(3, 3)
                self.attribute = start
                self.parameter = nn.Parameter(torch.tensor(3, dtype=torch.float))

            def method(self) -> int:
                # [FUNCTION DOC] HELPER: Returns the integer attribute.
                return self.attribute

            @torch.jit.unused
            def unused_method(self):
                # [FUNCTION DOC] HELPER: Unused-marked method returning twice the
                # integer attribute.
                return self.attribute + self.attribute

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer twice.
                return self.linear(self.linear(x))

        class N(nn.Module):
            # [CLASS DOC] HELPER: Module with a 4x4 linear layer and an ignored method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the linear layer.
                super().__init__()
                self.linear = nn.Linear(4, 4)

            @torch.jit.ignore
            def ignored_method(self, x):
                # [FUNCTION DOC] HELPER: Ignored-marked method returning its input.
                return x

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        m = torch.jit.script(M(3))
        n = torch.jit.script(N())

        n._reconstruct(m._c)

        inp = torch.rand((3))

        # Check that both modules produce the same output.
        with torch.no_grad():
            m_out = m(inp)
            n_out = n(inp)
            self.assertEqual(m_out, n_out)

        # Check that ignored method is still intact.
        self.assertEqual(inp, n.ignored_method(inp))

    def test_parameterlist_script_getitem(self):
        # [FUNCTION DOC] Scripts a module that indexes a ModuleList and a ParameterList with a
        # literal, and checks scripted equals eager.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Holds a ModuleList of ten linears and a ParameterList of
            # ten parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module list and the parameter list.
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                self.parameter_list = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(10)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Indexes both lists at position zero, discarding the
                # results; returns input.
                self.module_list[0]
                self.parameter_list[0]
                return x

        self.checkModule(MyModule(), (torch.zeros(1)))

    def test_parameterlist_script_iter(self):
        # [FUNCTION DOC] Scripts a module that iterates a ParameterList with enumerate and
        # checks scripted equals eager.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Holds a ModuleList of ten linears and a ParameterList of
            # ten parameters.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the module list and the parameter list.
                super().__init__()
                self.module_list = nn.ModuleList([nn.Linear(1, 1) for _ in range(10)])
                self.parameter_list = nn.ParameterList(
                    [nn.Parameter(torch.zeros(1)) for _ in range(10)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds each parameter and its index to the input and
                # returns the sum.
                r = x
                for i, p in enumerate(self.parameter_list):
                    r = r + p + i
                return r

        self.checkModule(MyModule(), (torch.zeros(1),))

    def test_parameterdict_script_getitem(self):
        # [FUNCTION DOC] Scripts a module that indexes a ParameterDict with literal keys and
        # checks scripted equals eager.
        class MyModule(nn.Module):
            # [CLASS DOC] HELPER: Holds a ParameterDict with three zero parameters keyed a, b
            # and c.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter dictionary.
                super().__init__()
                self.parameter_dict = nn.ParameterDict(
                    {k: nn.Parameter(torch.zeros(1)) for k in ["a", "b", "c"]}
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the a parameter times the input plus the
                # product of b and c.
                return (
                    self.parameter_dict["a"] * x
                    + self.parameter_dict["b"] * self.parameter_dict["c"]
                )

        self.checkModule(MyModule(), (torch.ones(1),))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
