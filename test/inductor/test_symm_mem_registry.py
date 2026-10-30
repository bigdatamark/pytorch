# Owner(s): ["module: inductor"]
# mypy: allow-untyped-defs
# [MODULE DOC] Tests for the symmetric memory argument registry: registering which operator
# arguments need symmetric memory, querying them, library level registration and cleanup, compile
# behavior of functional ops with registered arguments, and the fallback kernel step that realizes
# such arguments as communication buffers. Registry entries whose keys begin with the test prefix
# are cleared around each test.
"""
Tests for symmetric memory argument registry.

This test suite validates the symm_mem argument registration system, which allows
operators to declare which arguments require symmetric memory allocation.
"""

import unittest
from unittest.mock import patch

import torch
from torch._library.simple_registry import singleton, SymmMemArgsHolder
from torch.library import Library  # noqa: SCOPED_LIBRARY
from torch.testing._internal.common_utils import run_tests, TestCase
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


def register_symm_mem_args(op, arg_names):
    # [FUNCTION DOC] HELPER: Test helper that registers the named arguments as symmetric memory
    # arguments for an operator. The operator may be a qualified name string, an operator overload
    # or an object with a qualified name; any other object raises a TypeError. It looks up the
    # singleton registry entry and registers the argument names on it.
    """Helper function for tests to register symm_mem args."""
    from torch._ops import OpOverload

    if isinstance(op, str):
        qualname = op
    elif isinstance(op, OpOverload):
        qualname = op.__qualname__
    else:
        if hasattr(op, "__qualname__"):
            qualname = op.__qualname__
        else:
            raise TypeError(f"Expected OpOverload or string, got {type(op)}")

    entry = singleton.find(qualname)
    entry.symm_mem_args.register(arg_names)


class TestSymmMemRegistry(TestCase):
    # [CLASS DOC] Tests of the core argument holder: registration with names, with several names,
    # empty lists, unregistered queries, overload objects, overwrite and isolation of holder
    # instances. Setup and teardown remove test entries from the singleton registry.
    """Test suite for SymmMemArgsHolder core functionality."""

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setup and deletes registry entries whose keys start with
        # the test prefix, leaving real registrations alone.
        """Clear test entries from registry before each test."""
        super().setUp()
        # Only clear test entries, preserve real registrations (e.g., symm_mem::*)
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]

    def tearDown(self):
        # [FUNCTION DOC] Deletes registry entries with the test prefix, then calls the parent
        # teardown.
        """Clear test entries from registry after each test."""
        # Only clear test entries, preserve real registrations (e.g., symm_mem::*)
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]
        super().tearDown()

    def test_basic_registration_with_string(self):
        # [FUNCTION DOC] Registers one argument by string name and checks the entry reports it
        # registered with exactly that argument.
        """Test basic registration using string qualname."""
        register_symm_mem_args(
            "test_namespace::test_op",
            ["input"],
        )

        entry = singleton.find("test_namespace::test_op")
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

    def test_multiple_args_registration(self):
        # [FUNCTION DOC] Registers three arguments and checks the stored set and that membership
        # queries are true for registered names and false for others.
        """Test registering multiple arguments for an operator."""
        register_symm_mem_args(
            "test_namespace::test_op",
            ["input", "output", "buffer"],
        )

        entry = singleton.find("test_namespace::test_op")
        args = entry.symm_mem_args.get()
        self.assertEqual(set(args), {"input", "output", "buffer"})
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("output"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("other"))

    def test_empty_arg_list_raises(self):
        # [FUNCTION DOC] Checks that registering an empty list raises a ValueError saying empty
        # argument names cannot be registered.
        """Test that registering empty arg list raises ValueError."""
        with self.assertRaisesRegex(ValueError, "Cannot register empty arg_names"):
            register_symm_mem_args("test_namespace::test_op", [])

    def test_unregistered_op_queries(self):
        # [FUNCTION DOC] Looks up an operator that was never registered: the entry is created but
        # reports not registered, returns no arguments and says no name is a symmetric memory
        # argument.
        """Test querying unregistered operators."""
        # Note: find() creates an entry, but symm_mem_args won't be registered
        entry = singleton.find("nonexistent::op")
        self.assertFalse(entry.symm_mem_args.is_registered())
        self.assertIsNone(entry.symm_mem_args.get())
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("input"))

    def test_registration_with_op_overload(self):
        # [FUNCTION DOC] Registers using the add tensor operator overload and checks the entry is
        # found by its qualified name and the argument is recognized.
        """Test registration with actual OpOverload object."""
        op = torch.ops.aten.add.Tensor

        # Register without validation (since we're using arbitrary arg names)
        register_symm_mem_args(op, ["self"])

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))

    def test_overwrite_registration(self):
        # [FUNCTION DOC] Registers an operator twice with different names and checks the second
        # registration replaces the first.
        """Test that re-registering an op overwrites previous registration."""
        qualname = "test_namespace::test_op"

        register_symm_mem_args(qualname, ["input"])
        entry = singleton.find(qualname)
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

        # Overwrite with different args
        register_symm_mem_args(qualname, ["output"])
        self.assertEqual(set(entry.symm_mem_args.get()), {"output"})

    def test_get_all_registered(self):
        # [FUNCTION DOC] Registers two operators with different arguments and checks each entry
        # holds only its own arguments.
        """Test getting all registered operators."""
        register_symm_mem_args("ns1::op1", ["a"])
        register_symm_mem_args("ns2::op2", ["b", "c"])

        # Check that both are registered
        entry1 = singleton.find("ns1::op1")
        entry2 = singleton.find("ns2::op2")
        self.assertEqual(set(entry1.symm_mem_args.get()), {"a"})
        self.assertEqual(set(entry2.symm_mem_args.get()), {"b", "c"})

    def test_holder_isolation(self):
        # [FUNCTION DOC] Checks that a separately constructed holder is not affected by a
        # registration in the singleton registry, while the registry entry for the same name is
        # registered.
        """Test that creating a new SymmMemArgsHolder instance is isolated."""
        new_holder = SymmMemArgsHolder("test::op")

        # Register in singleton
        register_symm_mem_args("test::op", ["input"])

        # New holder should be empty
        self.assertFalse(new_holder.is_registered())

        # But singleton should have it registered
        entry = singleton.find("test::op")
        self.assertTrue(entry.symm_mem_args.is_registered())


class TestLibraryIntegration(TestCase):
    # [CLASS DOC] Tests registering symmetric memory arguments through the library object,
    # including definition and fragment libraries, operator overloads and cleanup when the library
    # is deleted.
    """Test suite for Library.register_symm_mem_args integration."""

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setup and deletes registry entries whose keys start with
        # the test prefix.
        """Clear test entries from registry before each test."""
        super().setUp()
        # Only clear test entries, preserve real registrations (e.g., symm_mem::*)
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]

    def tearDown(self):
        # [FUNCTION DOC] Deletes registry entries with the test prefix, then calls the parent
        # teardown.
        """Clear test entries from registry after each test."""
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]
        super().tearDown()

    def test_library_def_registration(self):
        # [FUNCTION DOC] Creates a definition library with an operator, registers one input as a
        # symmetric memory argument and checks that argument is recognized but the group name is
        # not.
        """Test registration via Library.DEF."""
        lib = Library("test_lib_def", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("test_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("test_op", ["input"])

        qualname = "test_lib_def::test_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    def test_library_fragment_registration(self):
        # [FUNCTION DOC] Uses a fragment library on the aten namespace to register an argument on
        # the add tensor operator and checks it is registered.
        """Test registration via Library.FRAGMENT."""
        lib = Library("aten", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.register_symm_mem_args("add.Tensor", ["self"])

        qualname = "aten::add.Tensor"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))

    def test_library_multiple_args(self):
        # [FUNCTION DOC] Defines an operator with three tensor arguments and registers all three
        # through the library, then checks each is recognized.
        """Test registering multiple args via Library."""
        lib = Library("test_lib_multi", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("test_op(Tensor a, Tensor b, Tensor out) -> Tensor")
        lib.register_symm_mem_args("test_op", ["a", "b", "out"])

        qualname = "test_lib_multi::test_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("a"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("b"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("out"))

    def test_library_with_op_overload(self):
        # [FUNCTION DOC] Registers two arguments for the multiplication tensor overload through a
        # fragment library and checks both are recognized.
        """Test Library registration with OpOverload object."""
        op = torch.ops.aten.mul.Tensor
        lib = Library("aten", "FRAGMENT")  # noqa: SCOPED_LIBRARY
        lib.register_symm_mem_args(op, ["self", "other"])

        qualname = op.__qualname__
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("self"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("other"))

    def test_library_cleanup_on_delete(self):
        # [FUNCTION DOC] Registers an argument on a defined operator, deletes the library, and
        # checks the registry entry is no longer registered and returns no arguments.
        """Test that deleting a Library cleans up its symm_mem_args registration."""
        lib = Library("test_lib_cleanup", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input, str reduce_op, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        qualname = "test_lib_cleanup::my_op"
        entry = singleton.find(qualname)
        self.assertTrue(entry.symm_mem_args.is_registered())

        del lib

        self.assertFalse(entry.symm_mem_args.is_registered())
        self.assertIsNone(entry.symm_mem_args.get())


class TestFunctionalOpCompile(TestCase):
    # [CLASS DOC] Tests that functional operators with symmetric memory arguments compile with
    # Inductor and that the fallback kernel step for realizing such arguments only acts when
    # appropriate.
    """Test that functional ops with registered symm_mem_args work with torch.compile."""

    def setUp(self):
        # [FUNCTION DOC] Calls the parent setup and deletes registry entries with the test prefix.
        super().setUp()
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]

    def tearDown(self):
        # [FUNCTION DOC] Deletes registry entries with the test prefix, resets dynamo, then calls
        # the parent teardown.
        test_keys = [k for k in singleton._data if k.startswith("test_")]
        for key in test_keys:
            del singleton._data[key]
        torch._dynamo.reset()
        super().tearDown()

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "Requires CUDA and Triton")
    def test_functional_op_compiles_with_symm_mem_args(self):
        # [FUNCTION DOC] Requires CUDA and Triton. Defines a library operator with a symmetric
        # memory input, with meta and CUDA implementations (the latter adds one), compiles a
        # function calling it with full graph and checks the compiled result matches eager and the
        # input plus one, and that the registry reports the input but not the group name.
        """Test that a functional op with registered symm_mem_args compiles and runs."""
        lib = Library("test_func_symm", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_functional_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_functional_op", ["input"])

        @torch.library.impl(lib, "my_functional_op", "Meta")
        def meta_impl(input, group_name):
            # [FUNCTION DOC] HELPER: Meta implementation: returns an empty tensor shaped like the
            # input.
            return torch.empty_like(input)

        @torch.library.impl(lib, "my_functional_op", "CUDA")
        def cuda_impl(input, group_name):
            # [FUNCTION DOC] HELPER: CUDA implementation: returns the input plus one.
            return input + 1.0

        def f_eager(x):
            # [FUNCTION DOC] HELPER: Calls the test operator eagerly with a fixed group name.
            return torch.ops.test_func_symm.my_functional_op(x, "test_group")

        @torch.compile(backend="inductor", fullgraph=True)
        def f_compiled(x):
            # [FUNCTION DOC] HELPER: Compiled version that calls the test operator with a fixed
            # group name.
            return torch.ops.test_func_symm.my_functional_op(x, "test_group")

        x = torch.randn(4, 4, device="cuda")

        eager_result = f_eager(x.clone())
        compiled_result = f_compiled(x.clone())
        torch.testing.assert_close(compiled_result, eager_result)

        expected = x + 1.0
        torch.testing.assert_close(compiled_result, expected)

        # Verify the registration is visible in the registry
        entry = singleton.find("test_func_symm::my_functional_op")
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    @unittest.skipIf(not HAS_CUDA_AND_TRITON, "Requires CUDA and Triton")
    def test_functional_op_with_multiple_symm_mem_args(self):
        # [FUNCTION DOC] Requires CUDA and Triton. Same as the single argument compile test but
        # with two symmetric memory arguments (input and out) whose CUDA implementation adds them;
        # checks compiled and eager results equal the sum and both names are registered but the
        # group name is not.
        """Test that multiple symm_mem args are registered and visible during compilation."""
        lib = Library("test_func_multi", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define(
            "my_multi_arg_op(Tensor input, Tensor out, str group_name) -> Tensor"
        )
        lib.register_symm_mem_args("my_multi_arg_op", ["input", "out"])

        @torch.library.impl(lib, "my_multi_arg_op", "Meta")
        def meta_impl(input, out, group_name):
            # [FUNCTION DOC] HELPER: Meta implementation: returns an empty tensor shaped like the
            # input.
            return torch.empty_like(input)

        @torch.library.impl(lib, "my_multi_arg_op", "CUDA")
        def cuda_impl(input, out, group_name):
            # [FUNCTION DOC] HELPER: CUDA implementation: returns the sum of the two tensor
            # arguments.
            # use both symm_mem args to verify functionality
            return input + out

        def f_eager(x, y):
            # [FUNCTION DOC] HELPER: Calls the two-argument test operator eagerly with a fixed
            # group name.
            return torch.ops.test_func_multi.my_multi_arg_op(x, y, "test_group")

        @torch.compile(backend="inductor", fullgraph=True)
        def f_compiled(x, y):
            # [FUNCTION DOC] HELPER: Compiled version that calls the two-argument test operator
            # with a fixed group name.
            return torch.ops.test_func_multi.my_multi_arg_op(x, y, "test_group")

        x = torch.randn(4, 4, device="cuda")
        y = torch.randn(4, 4, device="cuda")

        eager_result = f_eager(x.clone(), y.clone())
        compiled_result = f_compiled(x.clone(), y.clone())
        torch.testing.assert_close(compiled_result, eager_result)

        expected = x + y
        torch.testing.assert_close(compiled_result, expected)

        # Verify both args are registered
        entry = singleton.find("test_func_multi::my_multi_arg_op")
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("input"))
        self.assertTrue(entry.symm_mem_args.is_symm_mem_arg("out"))
        self.assertFalse(entry.symm_mem_args.is_symm_mem_arg("group_name"))

    def test_unregistered_op_skips_realization(self):
        # [FUNCTION DOC] Calls the fallback kernel realization step for an operator that has no
        # symmetric memory registration with a raw tensor, and checks it returns without error.
        """Test that _maybe_realize_symm_mem_args returns early for unregistered ops."""
        from torch._inductor.ir import FallbackKernel

        # Use an op unlikely to be registered by other tests
        op = torch.ops.aten.abs.default

        # Verify the registry has no symm_mem_args for this op
        entry = singleton.get(op.__qualname__)
        if entry is not None:
            self.assertFalse(entry.symm_mem_args.is_registered())

        # Should return immediately without error or side effects
        FallbackKernel._maybe_realize_symm_mem_args(op, torch.randn(2))

    def test_maybe_realize_finds_registered_op(self):
        # [FUNCTION DOC] Registers an operator, checks the registry entry exists and holds the
        # argument, then calls the fallback realization step with raw tensor and string arguments;
        # it should find the registration but return early without error.
        """Test that _maybe_realize_symm_mem_args correctly looks up registered ops."""
        from torch._inductor.ir import FallbackKernel

        lib = Library("test_realize_lookup", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        op = torch.ops.test_realize_lookup.my_op.default

        # Verify the op is found in the registry
        entry = singleton._data.get(op.__qualname__)
        self.assertIsNotNone(entry)
        self.assertTrue(entry.symm_mem_args.is_registered())
        self.assertEqual(set(entry.symm_mem_args.get()), {"input"})

        # Call _maybe_realize_symm_mem_args directly. It should find the
        # registration but return early because group_name is a raw string
        # (not a graph-time value) and the tensor arg is a raw tensor
        # (not a TensorBox). No error expected.
        FallbackKernel._maybe_realize_symm_mem_args(op, torch.randn(4, 4), "test_group")

    def test_maybe_realize_skips_without_group_name(self):
        # [FUNCTION DOC] Registers an operator whose signature has no group name argument and
        # calls the realization step with only a tensor; it should return early without error.
        """Test that _maybe_realize_symm_mem_args returns early when group_name is absent."""
        from torch._inductor.ir import FallbackKernel

        lib = Library("test_no_group", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        op = torch.ops.test_no_group.my_op.default

        # No group_name argument → should return early without error
        FallbackKernel._maybe_realize_symm_mem_args(op, torch.randn(4, 4))

    def test_realize_called_for_tensorbox_with_registered_args(self):
        # [FUNCTION DOC] Registers an operator and calls the realization step with a mock of the
        # tensor box type while the can-realize check returns true and the realize function is
        # replaced by a recorder. Checks the recorder was called once with a symmetric memory
        # communication type and the given group name.
        """Test that _maybe_realize_symm_mem_args calls realize_as_comm_buffer for TensorBox args."""
        from unittest.mock import MagicMock

        from torch._inductor.ir import FallbackKernel, TensorBox

        lib = Library("test_realize_call", "DEF")  # noqa: SCOPED_LIBRARY
        lib.define("my_op(Tensor input, str group_name) -> Tensor")
        lib.register_symm_mem_args("my_op", ["input"])

        op = torch.ops.test_realize_call.my_op.default
        mock_tbox = MagicMock(spec=TensorBox)

        realize_log = []

        def mock_realize(t, comm_type, group_name):
            # [FUNCTION DOC] HELPER: Recorder replacing the realize function: appends the string
            # form of the communication type and the group name to a log list.
            realize_log.append((str(comm_type), group_name))

        with (
            patch(
                "torch._inductor.comm_lowering.can_realize_as_comm_buffer",
                return_value=True,
            ),
            patch(
                "torch._inductor.comm_lowering.realize_as_comm_buffer",
                side_effect=mock_realize,
            ),
        ):
            FallbackKernel._maybe_realize_symm_mem_args(op, mock_tbox, "test_group")

        self.assertEqual(len(realize_log), 1)
        self.assertIn("SYMM_MEM", realize_log[0][0])
        self.assertEqual(realize_log[0][1], "test_group")


if __name__ == "__main__":
    run_tests()
