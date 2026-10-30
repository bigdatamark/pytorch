# Owner(s): ["oncall: jit"]
# [MODULE DOC] JIT save/load tests. TestSaveLoad covers the standard serialization path (BytesIO
# buffers, files, pathlib paths, extra files, NamedTuple signatures,
# parameters/buffers/submodules, meta tensors, traced inputs, huge string attributes) and name
# collisions between independently compiled modules, functions, interfaces, classes and named
# tuples. TestSaveLoadFlatbuffer repeats the relevant checks through the flatbuffer
# lite-interpreter format. Importing appends the test directory to sys.path; direct execution
# redirects to test/test_jit.py.

import io
import os
import sys
import unittest
from pathlib import Path
from typing import NamedTuple, Optional

import torch
from torch import Tensor
from torch.testing._internal.common_cuda import SM120OrLater
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    raise_on_run_directly,
    skipIfTorchDynamo,
    TemporaryFileName,
)


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.jit_utils import clear_class_registry, JitTestCase


class TestSaveLoad(JitTestCase):
    # [CLASS DOC] Round-trip tests for torch.jit.save and torch.jit.load: each test scripts a
    # module, serializes it, reloads it and compares structure or behavior; collision tests clear
    # the class registry between compilations so that two different definitions share one
    # qualified name.
    def test_different_modules(self):
        # [FUNCTION DOC] Exercises two different module definitions with the same qualified name
        # in different compilation units on save/load. Saves the first Foo, clears the class
        # registry, defines and saves a second, different version, checks both script modules
        # share a qualified name, then loads both into one container module, scripts, saves and
        # reloads that container; passes if nothing raises.
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two Linear layers.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first then the second layer.
                x = self.foo(x)
                x = self.bar(x)
                return x

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Linear layer (same qualified name as the
            # earlier version).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the single Linear layer.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer.
                x = self.foo(x)
                return x

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first loaded module, then the second.
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)

    def test_different_functions(self):
        # [FUNCTION DOC] Same collision scenario with a module calling a free function
        # whose definition differs: it returns the argument in the first version and a
        # string in the second. Saves the first Foo, clears the class registry, defines
        # and saves a second, different version, checks both script modules share a
        # qualified name, then loads both into one container module, scripts, saves and
        # reloads that container; passes if nothing raises.
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        def lol(x):
            # [FUNCTION DOC] HELPER: Returns its argument.
            return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the free function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the free function on the input.
                return lol(x)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        def lol(x):  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefinition of the free function returning a string.
            return "hello"

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the redefined free function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the free function on the input.
                return lol(x)

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first loaded module, then the second.
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)

    def test_different_interfaces(self):
        # [FUNCTION DOC] Same collision scenario with an interface and a scripted class
        # implementing it: the first version declares a method named bar, the second a method
        # named not_bar, behind the same names. Saves the first Foo, clears the class registry,
        # defines and saves a second, different version, checks both script modules share a
        # qualified name, then loads both into one container module, scripts, saves and reloads
        # that container; passes if nothing raises.
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Interface declaring one method
            # taking a tensor and returning a tensor.
            def bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:
            # [CLASS DOC] HELPER: Scripted class implementing the
            # interface with an identity method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attribute is typed as the interface and holds the
            # implementing object.
            __annotations__ = {"interface": MyInterface}

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an instance of the implementing class.
                super().__init__()
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the interface method.
                return self.interface.bar(x)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Redefined interface whose method is named not_bar.
            def not_bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:  # noqa: F811
            # [CLASS DOC] HELPER: Redefined implementing class with the renamed method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def not_bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attribute is typed as the redefined interface.
            __annotations__ = {"interface": MyInterface}

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an instance of the implementing class.
                super().__init__()
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the renamed interface method.
                return self.interface.not_bar(x)

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(torch.jit.script(Foo()), second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first loaded module, then the second.
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)

    def test_many_collisions(self):
        # [FUNCTION DOC] Collision scenario combining all kinds of colliding definitions: a named
        # tuple with an int field versus a string field, an interface, an implementing class, a
        # free function and a module. The container module returns a number computed from both
        # loaded modules outputs. Saves the first Foo, clears the class registry, defines and
        # saves a second, different version, checks both script modules share a qualified name,
        # then loads both into one container module, scripts, saves and reloads that container;
        # passes if nothing raises.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one int field.
            a: int

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Interface declaring one method.
            def bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:
            # [CLASS DOC] HELPER: Scripted class implementing the interface.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        def lol(x):
            # [FUNCTION DOC] HELPER: Returns its argument.
            return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers, an interface attribute and a free
            # function call; returns the result and a named tuple.
            interface: MyInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers and the implementing object.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers, the free function and the interface
                # method, and returns the result with a named tuple holding five.
                x = self.foo(x)
                x = self.bar(x)
                x = lol(x)
                x = self.interface.bar(x)

                return x, MyCoolNamedTuple(a=5)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save(first_script_module, first_saved_module)
        first_saved_module.seek(0)

        clear_class_registry()

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Redefined interface with a method named not_bar.
            def not_bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:  # noqa: F811
            # [CLASS DOC] HELPER: Redefined implementing class.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def not_bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        def lol(x):  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefined free function returning a string.
            return "asdofij"

        class MyCoolNamedTuple(NamedTuple):  # noqa: F811
            # [CLASS DOC] HELPER: Redefined named tuple with one string field.
            a: str

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Redefined module with one Linear layer; returns the result and a
            # named tuple holding a string.
            interface: MyInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layer and the implementing object.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer, calls the interface method, applies
                # the free function, and returns the result with the named tuple.
                x = self.foo(x)
                self.interface.not_bar(x)
                x = lol(x)
                return x, MyCoolNamedTuple(a="hello")

        second_script_module = torch.jit.script(Foo())
        second_saved_module = io.BytesIO()
        torch.jit.save(second_script_module, second_saved_module)
        second_saved_module.seek(0)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Unpacks each loaded module output and combines the
                # lengths and named tuple fields into one number.
                x, named_tuple_1 = self.first(x)
                x, named_tuple_2 = self.second(x)
                return len(x + named_tuple_2.a) + named_tuple_1.a

        sm = torch.jit.script(ContainsBoth())
        contains_both = io.BytesIO()
        torch.jit.save(sm, contains_both)
        contains_both.seek(0)
        sm = torch.jit.load(contains_both)

    def test_save_load_with_extra_files(self):
        # [FUNCTION DOC] Saves a trivial ScriptModule with extra files (binary bytes and a string,
        # which is converted to bytes) to a file path, through the torch.jit API, and to buffers,
        # then loads each time and checks the requested extra files come back as bytes with the
        # saved contents. Loading a name that was not saved must raise a RuntimeError.
        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module with an identity forward.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns the input.
                return a

        # specifically test binary data
        value = b"bar\x00\xffbaz"

        expected_extra_files = {}
        expected_extra_files["foo"] = value
        # verify that str to bytes conversion also works
        expected_extra_files["foo2"] = "bar"
        m = MyMod()

        # Save to file.
        with TemporaryFileName() as fname:
            m.save(fname, _extra_files=expected_extra_files)
            # values don't matter
            extra_files = {"foo": "", "foo2": None}
            torch.jit.load(fname, _extra_files=extra_files)
            self.assertEqual(value, extra_files["foo"])
            # results come back always as bytes
            self.assertEqual(b"bar", extra_files["foo2"])

            # Use torch.jit API
            torch.jit.save(m, fname, _extra_files=expected_extra_files)
            extra_files["foo"] = ""
            torch.jit.load(fname, _extra_files=extra_files)
            self.assertEqual(value, extra_files["foo"])

        # Save to buffer.
        buffer = io.BytesIO(m.save_to_buffer(_extra_files=expected_extra_files))
        extra_files = {"foo": ""}
        torch.jit.load(buffer, _extra_files=extra_files)
        self.assertEqual(value, extra_files["foo"])

        # Use torch.jit API
        buffer = io.BytesIO()
        torch.jit.save(m, buffer, _extra_files=expected_extra_files)
        buffer.seek(0)
        extra_files = {"foo": ""}
        torch.jit.load(buffer, _extra_files=extra_files)
        self.assertEqual(value, extra_files["foo"])

        # Non-existent file 'bar'
        with self.assertRaises(RuntimeError):
            extra_files["bar"] = ""
            torch.jit.load(buffer, _extra_files=extra_files)

    def test_save_load_using_pathlib(self):
        # [FUNCTION DOC] Saves a ScriptModule to a pathlib path, reloads it from the path, and
        # checks both give the same results on a sample tensor.
        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module that doubles its input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns twice the input.
                return 2 * a

        m = MyMod()

        # Save then load.
        with TemporaryFileName() as fname:
            path = Path(fname)
            m.save(path)
            m2 = torch.jit.load(path)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(torch.equal(m(x), m2(x)))

    def test_save_nonexit_file(self):
        # [FUNCTION DOC] Saving a scripted module to a path in a nonexistent directory must raise
        # a RuntimeError.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module that doubles its input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns twice the input.
                return 2 * x

        script_module = torch.jit.script(Foo())
        with self.assertRaises(RuntimeError):
            script_module.save("NonExist/path/test.pt")

    def test_save_namedtuple_input_only(self):
        # [FUNCTION DOC] A named tuple used only as an input annotation survives export and import
        # (via the test helper); the loaded module returns the constant tensor three when called
        # with a tuple instance. The named tuple is declared global so the scripting frontend can
        # resolve it.
        """
        Even if a NamedTuple is only used as an input argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one int field.
            a: int

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes the named tuple and returns a
            # constant tensor.
            def forward(self, x: FooTuple) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Ignores the input and returns the tensor three.
                return torch.tensor(3)

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded(FooTuple(a=5))
        self.assertEqual(output, torch.tensor(3))

    def test_save_namedtuple_input_only_forwardref(self):
        # [FUNCTION DOC] Same as the input-only named tuple test, but the field type is a string
        # forward reference.
        """
        Even if a NamedTuple is only used as an input argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one field
            # annotated by a string forward reference.
            a: "int"

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes the named tuple and returns a
            # constant tensor.
            def forward(self, x: FooTuple) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Ignores the input and returns the tensor three.
                return torch.tensor(3)

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded(FooTuple(a=5))
        self.assertEqual(output, torch.tensor(3))

    def test_save_namedtuple_output_only(self):
        # [FUNCTION DOC] A named tuple used only in an output annotation survives export and
        # import; the loaded module returns None.
        """
        Even if a NamedTuple is only used as an output argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one int field.
            a: int

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns an optional named tuple, here
            # always None.
            def forward(self) -> Optional[FooTuple]:
                # [FUNCTION DOC] HELPER: Returns None.
                return None

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded()
        self.assertEqual(output, None)

    def test_save_load_params_buffers_submodules(self):
        # [FUNCTION DOC] Exports and imports a module with parameters, buffers, a non-buffer
        # tensor, and submodules registered in different ways, then checks the loaded copy has the
        # same submodule names, equal parameters and equal named buffers.
        """
        Check that parameters, buffers, and submodules are the same after loading.
        """

        class Submodule(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty module used as a submodule.
            pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering submodules, parameters and buffers through
            # several APIs plus a plain tensor attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodules, two parameters, two buffers and a
                # plain tensor.
                super().__init__()
                self.add_module("submodule_a", Submodule())
                self.register_parameter(
                    "parameter_a", torch.nn.Parameter(torch.randn(4))
                )
                self.buffer = torch.nn.Buffer(torch.randn(4))
                self.t = torch.rand(4)  # not buffer

                self.parameter_b = torch.nn.Parameter(torch.randn(4))
                self.submodule_b = Submodule()
                self.buffer_b = torch.nn.Buffer(torch.randn(4))

        m = TestModule()
        m_loaded = self.getExportImportCopy(torch.jit.script(m))

        # Check submodules.
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        for m_s, loaded_s in zip(m.named_modules(), m_loaded.named_modules()):
            m_name, _ = m_s
            loaded_name, _ = loaded_s
            self.assertEqual(m_name, loaded_name)

        # Check parameters.
        self.assertEqual(len(list(m.parameters())), len(list(m_loaded.parameters())))
        for m_p, loaded_p in zip(m.parameters(), m_loaded.parameters()):
            self.assertEqual(m_p, loaded_p)

        # Check buffers.
        self.assertEqual(
            len(list(m.named_buffers())), len(list(m_loaded.named_buffers()))
        )
        for m_b, loaded_b in zip(m.named_buffers(), m_loaded.named_buffers()):
            m_name, m_buffer = m_b
            loaded_name, loaded_buffer = loaded_b
            self.assertEqual(m_name, loaded_name)
            self.assertEqual(m_buffer, loaded_buffer)

    def test_save_load_meta_tensors(self):
        # [FUNCTION DOC] Round-trips a module with some parameters and a buffer on the meta device
        # and some on the default device, and checks submodule names, parameters and buffers are
        # equal and that the meta flags are preserved on each tensor.
        """
        Check that parameters, buffers, and submodules are the same after loading
        for a module with parameters and buffers that are meta tensors
        """

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a meta-device Linear, a regular Linear and a
            # meta-device buffer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers and the buffer.
                super().__init__()
                self.foo = torch.nn.Linear(2, 3, device="meta")
                self.bar = torch.nn.Linear(3, 4)
                self.buffer = torch.nn.Buffer(torch.randn(4, device="meta"))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the two layers in sequence.
                x = self.foo(x)
                x = self.bar(x)
                return x

        m = Foo()
        m_loaded = self.getExportImportCopy(torch.jit.script(m))
        # Check submodules.
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        self.assertEqual(
            {name for name, _ in m.named_modules()},
            {name for name, _ in m_loaded.named_modules()},
        )
        # Check parameters.
        m_params = dict(m.named_parameters())
        m_loaded_params = dict(m_loaded.named_parameters())
        self.assertEqual(len(m_params), len(m_loaded_params))
        self.assertEqual(m_params, m_loaded_params)
        # Check buffers.
        m_buffers = dict(m.named_buffers())
        m_loaded_buffers = dict(m_loaded.named_buffers())
        self.assertEqual(len(m_buffers), len(m_loaded_buffers))
        self.assertEqual(m_buffers, m_loaded_buffers)
        # Check params and buffers that are/are not meta tensors
        self.assertTrue(m_params["foo.weight"].is_meta)
        self.assertTrue(m_loaded_params["foo.weight"].is_meta)
        self.assertTrue(m_params["foo.bias"].is_meta)
        self.assertTrue(m_loaded_params["foo.bias"].is_meta)
        self.assertFalse(m_params["bar.weight"].is_meta)
        self.assertFalse(m_loaded_params["bar.weight"].is_meta)
        self.assertFalse(m_params["bar.bias"].is_meta)
        self.assertFalse(m_loaded_params["bar.bias"].is_meta)
        self.assertTrue(m_buffers["buffer"].is_meta)
        self.assertTrue(m_loaded_buffers["buffer"].is_meta)

    def test_save_load_meta_tensors_to_device(self):
        # [FUNCTION DOC] Round-trips a module with meta and regular parameters with a map location
        # of cpu: checks the meta tensors stay on meta while the regular parameters are on cpu.
        """
        Check that when loading a module with meta tensors to device, the meta tensors
        stay on meta, but non-meta tensors are set to the indicated device.
        """

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a meta-device Linear and a regular Linear.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two layers.
                super().__init__()
                self.foo = torch.nn.Linear(2, 3, device="meta")
                self.bar = torch.nn.Linear(3, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the two layers in sequence.
                x = self.foo(x)
                x = self.bar(x)
                return x

        m = Foo()

        m_loaded = self.getExportImportCopy(torch.jit.script(m), map_location="cpu")
        # Check submodules.
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        self.assertEqual(
            {name for name, _ in m.named_modules()},
            {name for name, _ in m_loaded.named_modules()},
        )
        # Check parameters.
        m_params = dict(m.named_parameters())
        m_loaded_params = dict(m_loaded.named_parameters())
        self.assertEqual(len(m_params), len(m_loaded_params))
        self.assertEqual(m_params, m_loaded_params)
        # Check params and buffers that are/are not meta tensors
        self.assertTrue(m_params["foo.weight"].is_meta)
        self.assertTrue(m_loaded_params["foo.weight"].is_meta)
        self.assertTrue(m_params["foo.bias"].is_meta)
        self.assertTrue(m_loaded_params["foo.bias"].is_meta)
        self.assertTrue(m_params["bar.weight"].is_cpu)
        self.assertTrue(m_loaded_params["bar.weight"].is_cpu)
        self.assertTrue(m_params["bar.bias"].is_cpu)
        self.assertTrue(m_loaded_params["bar.bias"].is_cpu)

    def test_save_load_with_saved_traced_inputs(self):
        # [FUNCTION DOC] Checks traced input shape storage: tracing a module stores its example
        # inputs, saving then loading with shape restoration reproduces the input type and sizes,
        # loading without it leaves sizes unknown, and tracing with input storage disabled stores
        # none. Also runs helper checks for dictionary, list and nested inputs.
        # KNOWN ISSUE: the helper below ignores its parameter and always traces the first
        # dictionary input, so the second to fourth input cases do not exercise their own inputs.
        """
        Check that saving and loading with traced inputs works as expected
        """

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns a tensor of one.
            def __init__(self) -> None:
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Ignores the input and
                # returns a one-element tensor of ones.
                return torch.ones(1)

        def get_loaded_inputs(inputs):
            # [FUNCTION DOC] HELPER: Traces the module on a captured example input, saves it to a
            # temporary path, prints both graphs, reloads it with shape restoration, and returns
            # the traced and loaded graph input lists.
            # KNOWN ISSUE: the parameter is unused; the traced example is the enclosing variable
            # input1 rather than the argument.
            traced_module = torch.jit.trace(module, input1)
            traced_inputs = list(traced_module.graph.inputs())
            with TemporaryFileName() as fname:
                path = Path(fname)
                traced_module.save(path)
                print(traced_module.graph)
                loaded_module = torch.jit.load(path, _restore_shapes=True)
                print(loaded_module.graph)
                return traced_inputs, list(loaded_module.graph.inputs())

        module = Module()
        input_tensor = torch.rand(1, 3, 24, 24)
        # Validate that with no input specified the traced inputs are stored
        traced_module = torch.jit.trace(module, input_tensor)
        traced_inputs = list(traced_module.graph.inputs())
        self.assertEqual(
            traced_module._c._retrieve_traced_inputs()["forward"], [input_tensor]
        )
        with TemporaryFileName() as fname:
            path = Path(fname)
            traced_module.save(path)
            loaded_module = torch.jit.load(path, _restore_shapes=True)
            loaded_inputs = list(loaded_module.graph.inputs())
            self.assertEqual(traced_inputs[1].type(), loaded_inputs[1].type())
            self.assertEqual(
                traced_inputs[1].type().sizes(), loaded_inputs[1].type().sizes()
            )
            # Validate that if no shapes are requested previous functionality remains
            loaded_module = torch.jit.load(path)
            loaded_inputs = list(loaded_module.graph.inputs())
            self.assertEqual(loaded_inputs[1].type().sizes(), None)

        # Validate that inputs aren't saved when requested not to
        traced_module = torch.jit.trace(module, input_tensor, _store_inputs=False)
        traced_inputs = list(traced_module.graph.inputs())
        self.assertEqual(len(traced_module._c._retrieve_traced_inputs()), 0)

        with TemporaryFileName() as fname:
            path = Path(fname)
            traced_module.save(path)
            loaded_module = torch.jit.load(path, _restore_shapes=True)
            loaded_inputs = list(loaded_module.graph.inputs())
            self.assertEqual(loaded_inputs[1].type().sizes(), None)
            # Validate that if no shapes are requested previous functionality remains
            loaded_module = torch.jit.load(path)
            loaded_inputs = list(loaded_module.graph.inputs())
            self.assertEqual(loaded_inputs[1].type().sizes(), None)

        # Validate that complex inputs work
        # Testing dict of list with empty tensors
        input1 = {
            "1000": (
                torch.tensor([0]),
                torch.tensor([], dtype=torch.int64),
                torch.tensor([]),
            )
        }
        traced_inputs, loaded_inputs = get_loaded_inputs(input1)
        self.assertEqual(traced_inputs[1].type(), loaded_inputs[1].type())

        # Testing dict of list
        input2 = {
            "1000": (
                torch.tensor([0]),
                torch.tensor([1500000, 1500004], dtype=torch.int64),
                torch.tensor([2.0, 3.0]),
            )
        }
        traced_inputs, loaded_inputs = get_loaded_inputs(input2)
        self.assertEqual(traced_inputs[1].type(), loaded_inputs[1].type())

        # Testing list
        input3 = [
            torch.tensor([0]),
            torch.tensor([1500000, 1500004], dtype=torch.int64),
            torch.tensor([2.0, 3.0]),
        ]

        traced_inputs, loaded_inputs = get_loaded_inputs(input3)
        self.assertEqual(traced_inputs[1].type(), loaded_inputs[1].type())

        # Testing list of dict of list
        input4 = [
            {
                "1000": (
                    torch.tensor([0]),
                    torch.tensor([1500000, 1500004], dtype=torch.int64),
                    torch.tensor([2.0, 3.0]),
                )
            }
        ]

        traced_inputs, loaded_inputs = get_loaded_inputs(input4)
        self.assertEqual(traced_inputs[1].type(), loaded_inputs[1].type())

    @unittest.skipIf(
        IS_WINDOWS and SM120OrLater,
        "Process crash in PyTorchStreamWriter on SM120+ Windows",
    )
    @skipIfTorchDynamo("too slow")
    def test_save_load_large_string_attribute(self):
        # [FUNCTION DOC] Builds a module with a string attribute longer than four gigabytes,
        # scripts and runs it, saves it to a buffer, reloads it and checks the output is
        # unchanged. Skipped unless enough memory is available (probed with psutil) and on some
        # Windows hardware; marked as slow under Dynamo.
        """
        Check if the model with string > 4GB can be loaded.
        """
        import psutil

        if psutil.virtual_memory().available < 60 * 1024 * 1024 * 1024:
            # Profiled the test execution, and got this number to be safe to run the test
            self.skipTest(
                "Doesn't have enough memory to run test_save_load_large_string_attribute"
            )

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a very large string attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a string of more than four gigabytes.
                super().__init__()
                self.x = "x" * (2**32 + 1)

            def forward(self, i) -> int:
                # [FUNCTION DOC] HELPER: Returns the string length plus the number of elements of
                # the input.
                return len(self.x) + i.numel()

        inp = torch.ones(0)
        ts = torch.jit.script(Model())
        ts_output = ts(inp)

        b = io.BytesIO(ts.save_to_buffer())
        del ts

        loaded_ts = torch.jit.load(b)
        del b
        loaded_output = loaded_ts(inp)
        self.assertEqual(ts_output, loaded_output)


def script_module_to_buffer(script_module):
    # [FUNCTION DOC] HELPER: Serializes a script module in the flatbuffer lite-interpreter
    # format into an in-memory buffer, rewinds it, and returns it for loading.
    module_buffer = io.BytesIO(
        script_module._save_to_buffer_for_lite_interpreter(_use_flatbuffer=True)
    )
    module_buffer.seek(0)
    return module_buffer


class TestSaveLoadFlatbuffer(JitTestCase):
    # [CLASS DOC] Round-trip tests that serialize through the flatbuffer
    # lite-interpreter format via script_module_to_buffer, mirroring the
    # main tests, plus a test of flatbuffer module info.
    def test_different_modules(self):
        # [FUNCTION DOC] Two different module definitions with the same qualified name saved as
        # flatbuffer, then both loaded into a container which is scripted, saved and reloaded.
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two Linear layers.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first then the second layer.
                x = self.foo(x)
                x = self.bar(x)
                return x

        first_script_module = torch.jit.script(Foo())
        first_saved_module = script_module_to_buffer(first_script_module)

        clear_class_registry()

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a single Linear layer (same qualified name as the
            # earlier version).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the single Linear layer.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer.
                x = self.foo(x)
                return x

        second_script_module = torch.jit.script(Foo())
        second_saved_module = script_module_to_buffer(second_script_module)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first loaded module, then the second.
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = script_module_to_buffer(sm)
        sm = torch.jit.load(contains_both)

    def test_different_functions(self):
        # [FUNCTION DOC] Flatbuffer variant of the free-function collision test: the free function
        # returns the argument in the first version and a string in the second.
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        def lol(x):
            # [FUNCTION DOC] HELPER: Returns its argument.
            return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the free function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the free function on the input.
                return lol(x)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = script_module_to_buffer(first_script_module)
        clear_class_registry()

        def lol(x):  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefined free function returning a string.
            return "hello"

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module calling the redefined free function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the free function on the input.
                return lol(x)

        second_script_module = torch.jit.script(Foo())
        second_saved_module = script_module_to_buffer(second_script_module)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first loaded module, then the second.
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = script_module_to_buffer(sm)
        sm = torch.jit.load(contains_both)

    def test_different_interfaces(self):
        # [FUNCTION DOC] Flatbuffer variant of the interface collision test, with methods bar
        # versus not_bar.
        """
        Exercise the situation where we have the same qualified name
        in two different CompilationUnits on save/load.
        """

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Interface declaring one method.
            def bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:
            # [CLASS DOC] HELPER: Scripted class implementing the interface.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attribute is typed as the interface.
            __annotations__ = {"interface": MyInterface}

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an implementing object.
                super().__init__()
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the interface method.
                return self.interface.bar(x)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = script_module_to_buffer(first_script_module)
        clear_class_registry()

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Redefined interface with a method named not_bar.
            def not_bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:  # noqa: F811
            # [CLASS DOC] HELPER: Redefined implementing class.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def not_bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose attribute is typed as the redefined interface.
            __annotations__ = {"interface": MyInterface}

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores an implementing object.
                super().__init__()
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the renamed interface method.
                return self.interface.not_bar(x)

        second_script_module = torch.jit.script(Foo())
        second_saved_module = script_module_to_buffer(second_script_module)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first loaded module, then the second.
                x = self.first(x)
                x = self.second(x)
                return x

        sm = torch.jit.script(ContainsBoth())
        contains_both = script_module_to_buffer(sm)
        sm = torch.jit.load(contains_both)

    def test_many_collisions(self):
        # [FUNCTION DOC] Flatbuffer variant of the many-collisions test with named tuples,
        # interface, class, free function and module definitions that differ between versions.
        class MyCoolNamedTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one int field.
            a: int

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Interface declaring one method.
            def bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:
            # [CLASS DOC] HELPER: Scripted class implementing the interface.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        def lol(x):
            # [FUNCTION DOC] HELPER: Returns its argument.
            return x

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers, an interface attribute and a free
            # function call; returns the result and a named tuple.
            interface: MyInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layers and the implementing object.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers, the free function and the interface
                # method, and returns the result with a named tuple holding five.
                x = self.foo(x)
                x = self.bar(x)
                x = lol(x)
                x = self.interface.bar(x)

                return x, MyCoolNamedTuple(a=5)

        first_script_module = torch.jit.script(Foo())
        first_saved_module = script_module_to_buffer(first_script_module)

        clear_class_registry()

        @torch.jit.interface
        class MyInterface:
            # [CLASS DOC] HELPER: Redefined interface with a method named not_bar.
            def not_bar(self, x: Tensor) -> Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration with no behavior.
                pass

        @torch.jit.script
        class ImplementInterface:  # noqa: F811
            # [CLASS DOC] HELPER: Redefined implementing class.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Does nothing.
                pass

            def not_bar(self, x):
                # [FUNCTION DOC] HELPER: Returns its argument.
                return x

        def lol(x):  # noqa: F811
            # [FUNCTION DOC] HELPER: Redefined free function returning a string.
            return "asdofij"

        class MyCoolNamedTuple(NamedTuple):  # noqa: F811
            # [CLASS DOC] HELPER: Redefined named tuple with one string field.
            a: str

        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Redefined module with one Linear layer; returns the result and a
            # named tuple holding a string.
            interface: MyInterface

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the layer and the implementing object.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.interface = ImplementInterface()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layer, calls the interface method, applies
                # the free function, and returns the result with the named tuple.
                x = self.foo(x)
                self.interface.not_bar(x)
                x = lol(x)
                return x, MyCoolNamedTuple(a="hello")

        second_script_module = torch.jit.script(Foo())
        second_saved_module = script_module_to_buffer(second_script_module)

        clear_class_registry()

        self.assertEqual(
            first_script_module._c.qualified_name,
            second_script_module._c.qualified_name,
        )

        class ContainsBoth(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding the two loaded modules as submodules named first
            # and second.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Loads both serialized
                # modules and attaches them as submodules.
                super().__init__()
                self.add_module("second", torch.jit.load(second_saved_module))
                self.add_module("first", torch.jit.load(first_saved_module))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Unpacks each loaded module output and combines the
                # lengths and named tuple fields into one number.
                x, named_tuple_1 = self.first(x)
                x, named_tuple_2 = self.second(x)
                return len(x + named_tuple_2.a) + named_tuple_1.a

        sm = torch.jit.script(ContainsBoth())
        contains_both = script_module_to_buffer(sm)
        sm = torch.jit.load(contains_both)

    def test_save_load_using_pathlib(self):
        # [FUNCTION DOC] Saves a ScriptModule as flatbuffer to a pathlib path, loads it back, and
        # checks both give the same results.
        class MyMod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module that doubles its input.
            @torch.jit.script_method
            def forward(self, a):
                # [FUNCTION DOC] HELPER: Returns twice the input.
                return 2 * a

        m = MyMod()

        # Save then load.
        with TemporaryFileName() as fname:
            path = Path(fname)
            torch.jit.save_jit_module_to_flatbuffer(m, path)
            m2 = torch.jit.load(path)

        x = torch.tensor([1.0, 2.0, 3.0, 4.0])
        self.assertTrue(torch.equal(m(x), m2(x)))

    def test_save_namedtuple_input_only(self):
        # [FUNCTION DOC] Copy of the input-only named tuple test in the flatbuffer class; it
        # uses the generic export/import helper, not the flatbuffer buffer.
        """
        Even if a NamedTuple is only used as an input argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one int field.
            a: int

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward takes the named tuple and returns a
            # constant tensor.
            def forward(self, x: FooTuple) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Ignores the input and returns the tensor three.
                return torch.tensor(3)

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded(FooTuple(a=5))
        self.assertEqual(output, torch.tensor(3))

    def test_save_namedtuple_output_only(self):
        # [FUNCTION DOC] Variant of the output-only named tuple
        # test; the loaded module returns None.
        """
        Even if a NamedTuple is only used as an output argument, saving and
        loading should work correctly.
        """
        global FooTuple  # see [local resolution in python]

        class FooTuple(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with one int field.
            a: int

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns an optional named tuple, here
            # always None.
            def forward(self) -> Optional[FooTuple]:
                # [FUNCTION DOC] HELPER: Returns None.
                return None

        m_loaded = self.getExportImportCopy(torch.jit.script(MyModule()))
        output = m_loaded()
        self.assertEqual(output, None)

    def test_module_info_flatbuffer(self):
        # [FUNCTION DOC] Saves a two-layer scripted module to flatbuffer and reads module info:
        # bytecode version nine, operator version one, no type names, one operator (linear, three
        # arguments), and one function name ending with forward.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with two Linear layers applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two layers.
                super().__init__()
                self.foo = torch.nn.Linear(2, 2)
                self.bar = torch.nn.Linear(2, 2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first then the second layer.
                x = self.foo(x)
                x = self.bar(x)
                return x

        first_script_module = torch.jit.script(Foo())
        first_saved_module = io.BytesIO()
        torch.jit.save_jit_module_to_flatbuffer(first_script_module, first_saved_module)
        first_saved_module.seek(0)
        ff_info = torch.jit._serialization.get_flatbuffer_module_info(
            first_saved_module
        )
        self.assertEqual(ff_info["bytecode_version"], 9)
        self.assertEqual(ff_info["operator_version"], 1)
        self.assertEqual(ff_info["type_names"], set())
        self.assertEqual(ff_info["opname_to_num_args"], {"aten::linear": 3})

        self.assertEqual(len(ff_info["function_names"]), 1)
        self.assertTrue(next(iter(ff_info["function_names"])).endswith("forward"))

    def test_save_load_params_buffers_submodules(self):
        # [FUNCTION DOC] Flatbuffer variant of the parameters, buffers and
        # submodules check: export/import copy must have the same submodule
        # names, equal parameters and equal buffers.
        """
        Check that parameters, buffers, and submodules are the same after loading.
        """

        class Submodule(torch.nn.Module):
            # [CLASS DOC] HELPER: Empty module used as a submodule.
            pass

        class TestModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module registering submodules, parameters and buffers through
            # several APIs plus a plain tensor attribute.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the submodules, two parameters, two buffers and a
                # plain tensor.
                super().__init__()
                self.add_module("submodule_a", Submodule())
                self.register_parameter(
                    "parameter_a", torch.nn.Parameter(torch.randn(4))
                )
                self.buffer = torch.nn.Buffer(torch.randn(4))
                self.t = torch.rand(4)  # not buffer

                self.parameter_b = torch.nn.Parameter(torch.randn(4))
                self.submodule_b = Submodule()
                self.buffer_b = torch.nn.Buffer(torch.randn(4))

        m = TestModule()
        m_loaded = self.getExportImportCopy(torch.jit.script(m))

        # Check submodules.
        self.assertEqual(
            len(list(m.named_modules())), len(list(m_loaded.named_modules()))
        )
        for m_s, loaded_s in zip(m.named_modules(), m_loaded.named_modules()):
            m_name, _ = m_s
            loaded_name, _ = loaded_s
            self.assertEqual(m_name, loaded_name)

        # Check parameters.
        self.assertEqual(len(list(m.parameters())), len(list(m_loaded.parameters())))
        for m_p, loaded_p in zip(m.parameters(), m_loaded.parameters()):
            self.assertEqual(m_p, loaded_p)

        # Check buffers.
        self.assertEqual(
            len(list(m.named_buffers())), len(list(m_loaded.named_buffers()))
        )
        for m_b, loaded_b in zip(m.named_buffers(), m_loaded.named_buffers()):
            m_name, m_buffer = m_b
            loaded_name, loaded_buffer = loaded_b
            self.assertEqual(m_name, loaded_name)
            self.assertEqual(m_buffer, loaded_buffer)

    def test_save_load_with_extra_files(self):
        # [FUNCTION DOC] Scripts an identity module, saves it as flatbuffer with an extra file,
        # reads the extra files back from the buffer with the C++ binding, and checks they equal
        # the saved ones.
        # KNOWN ISSUE: the docstring is copied from the parameters/buffers test and does not
        # describe this test.
        """
        Check that parameters, buffers, and submodules are the same after loading.
        """

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an identity forward.
            def forward(self, x: Tensor):
                # [FUNCTION DOC] HELPER: Returns the input.
                return x

        module = Module()
        script_module = torch.jit.script(module)

        extra_files = {"abc.json": b"[1,2,3]"}
        script_module_io = script_module._save_to_buffer_for_lite_interpreter(
            _extra_files=extra_files, _use_flatbuffer=True
        )

        re_extra_files = {}
        torch._C._get_model_extra_files_from_buffer(script_module_io, re_extra_files)

        self.assertEqual(extra_files, re_extra_files)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
