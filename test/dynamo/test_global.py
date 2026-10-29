# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests of how Dynamo handles stores to module globals (global statement writes,
# in-place mutation of global containers and objects, creation of new globals, cross-file
# globals, globals changed in inlined functions) and of modules loaded without registration in
# the module registry. Defines module-level globals (a counter, list, dict, object, tensor), a
# name generator and helper class. Import time creates these globals; the utils module is
# imported from the package or directly.

import importlib.util
import os
import sys
import tempfile

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import same


try:
    from . import utils
except ImportError:
    import utils


class Pair:
    # [CLASS DOC] HELPER: Simple record with two public attributes, x and y, used as a global
    # object whose attribute is mutated.
    def __init__(self, x, y):
        # [FUNCTION DOC] HELPER: Store the two given values as attributes x and y.
        self.x = x
        self.y = y


def Foo():
    # [FUNCTION DOC] HELPER: Factory returning a new Pair with both attributes equal to 1;
    # used to create the global object.
    return Pair(1, 1)


g_counter = 1
g_list = [0, 1, 2]
g_dict = {"a": 0, "b": 1}
g_object = Foo()
g_tensor = torch.zeros(10)


_name: int = 0


def fresh_name() -> str:
    # [FUNCTION DOC] HELPER: Return a fresh unique name of the form v followed by a counter,
    # increment the module global counter used for naming.
    """create a new unique name for a variable: v0, v1, v2"""
    global _name
    r = f"v{_name}"
    _name += 1
    return r


def reset_name():
    # [FUNCTION DOC] HELPER: Reset the module global name counter to zero.
    global _name
    _name = 0


class TestGlobals(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for global variable stores under compile. Tests mutate shared module
    # globals, so values persist across tests.
    def test_store_global_1(self):
        # [FUNCTION DOC] Compiled function reads a global counter and increments it; after one
        # compiled and one eager call the results differ by one, showing the global was
        # updated by compiled code.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Declares the global counter, returns
            # the input plus its value, then increments it.
            global g_counter
            val = x + g_counter
            g_counter += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_2(self):
        # [FUNCTION DOC] Global counter incremented twice per call; running the compiled
        # function twice (recompiled between runs) leaves results differing by two.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input plus the global
            # counter then increments the counter twice.
            global g_counter
            val = x + g_counter
            g_counter += 1
            g_counter += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        """Wrap the second call with torch._dynamo as well"""
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x)
        self.assertTrue(same(res2 - res1, 2 * torch.ones(10)))

    def test_store_global_new(self):
        # [FUNCTION DOC] A function that assigns a global name that does not yet exist creates
        # it; compiled result equals the input doubled plus one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a new global holding the
            # input plus one and returns the input plus that global.
            # Test create a new global
            global g_counter_new
            g_counter_new = x + 1
            return x + g_counter_new

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        self.assertTrue(same(res1, x + x + 1))

    def test_store_global_list(self):
        # [FUNCTION DOC] Mutating an element of a global list within compiled code is applied
        # to the real list; results of compiled and eager calls differ by one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input plus element 1 of
            # the global list and increments that element in place.
            global g_list
            val = x + g_list[1]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_list[1] += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_list_2(self):
        # [FUNCTION DOC] Rebinding a global list to a new list (via comprehension adding one)
        # is applied; compiled and eager results differ by one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input plus element 1 of
            # the global list, then rebinds the global to a list with each element increased
            # by one.
            global g_list
            val = x + g_list[1]
            g_list = [x + 1 for x in g_list]
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict(self):
        # [FUNCTION DOC] Mutating a value in a global dict in place from compiled code is
        # applied; compiled and eager results differ by one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input plus the value for
            # key b of the global dict and increments that value in place.
            global g_dict
            val = x + g_dict["b"]
            """
            Strictly speaking, we are not testing STORE_GLOBAL
            here, since STORE_SUBSCR is actually used to store.
            """
            g_dict["b"] += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_dict_2(self):
        # [FUNCTION DOC] Rebinding a global dict to a new dict with each value increased by
        # one is applied; compiled and eager results differ by one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Rebinds the global dict to one with
            # all values plus one, then returns the input plus the value for key b.
            global g_dict
            g_dict = {key: value + 1 for key, value in g_dict.items()}
            val = x + g_dict["b"]
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_object(self):
        # [FUNCTION DOC] Mutating an attribute of a global object in compiled code is applied;
        # compiled and eager results differ by one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input plus the y
            # attribute of the global object, then increments the attribute.
            global g_object
            val = x + g_object.y
            g_object.y += 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_cross_file(self):
        # [FUNCTION DOC] Storing a module attribute of another module (utils tensor global)
        # from compiled code is applied; compiled and eager results differ by one.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the input plus the tensor
            # global of the utils module and rebinds that attribute to the tensor plus one.
            val = x + utils.g_tensor_export
            utils.g_tensor_export = utils.g_tensor_export + 1
            return val

        x = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res1 = opt_fn(x)
        res2 = fn(x)
        self.assertTrue(same(res2 - res1, torch.ones(10)))

    def test_store_global_inline_1(self):
        # [FUNCTION DOC] A class defined in the test whose constructor takes the default name
        # from the module-level name generator is inlined; compiled call yields names v0 and
        # v1. Resets the generator at the end.
        # Borrowed from test_python_autograd.py
        class Variable:
            # [CLASS DOC] HELPER: Holds a tensor value and a name, falling back to a freshly
            # generated name.
            def __init__(self, value: torch.Tensor, name: str | None = None):
                # [FUNCTION DOC] HELPER: Store the value, and the name or a fresh generated
                # name when none or empty is given.
                self.value = value
                self.name = name or fresh_name()

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Function under test. Wraps both tensors in the class and
            # returns the sum of values and the concatenated names.
            a = Variable(a)
            b = Variable(b)
            return a.value + b.value, a.name + b.name

        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v0, s0 = opt_fn(a, b)
        self.assertEqual(s0, "v0v1")
        reset_name()

    def test_store_global_inline_2(self):
        # [FUNCTION DOC] As for the previous test but the objects are created via a static
        # factory method; names are v0 and v1. Resets the generator at the end.
        # Borrowed from test_python_autograd.py
        class Variable:
            # [CLASS DOC] HELPER: Holds a tensor value and a name, with a static
            # factory method.
            def __init__(self, value: torch.Tensor, name: str | None = None):
                # [FUNCTION DOC] HELPER: Store the value, and the name or a fresh generated
                # name when none or empty is given.
                self.value = value
                self.name = name or fresh_name()

            @staticmethod
            def constant(value: torch.Tensor, name: str | None = None):
                # [FUNCTION DOC] HELPER: Static factory creating an instance from the value
                # and optional name.
                return Variable(value, name)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Function under test. Creates two instances through the
            # factory and returns the sum of values and the concatenated names.
            a = Variable.constant(a)
            b = Variable.constant(b)
            return a.value + b.value, a.name + b.name

        a = torch.randn(10)
        b = torch.randn(10)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        v0, s0 = opt_fn(a, b)
        self.assertEqual(s0, "v0v1")
        reset_name()

    def test_store_global_crossfile_inline(self):
        # [FUNCTION DOC] Stores to a global of a different module from inlined functions are
        # applied. One compiled function sets a flag true then false, another only true; flag
        # state is checked after each.
        try:
            from . import mock_store_global_crossfile_inline
        except ImportError:
            import mock_store_global_crossfile_inline

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Calls the mock module setters for
            # true then false and returns the input plus one.
            mock_store_global_crossfile_inline.set_flag_true()
            mock_store_global_crossfile_inline.set_flag_false()
            return x + 1

        @torch.compile(backend="eager")
        def fn_set_true(x):
            # [FUNCTION DOC] HELPER: Function under test. Calls the mock module setter for
            # true and returns the input plus one.
            mock_store_global_crossfile_inline.set_flag_true()
            return x + 1

        fn_set_true(torch.ones(2, 2))
        self.assertTrue(mock_store_global_crossfile_inline.global_flag)
        fn(torch.ones(2, 2))
        self.assertFalse(mock_store_global_crossfile_inline.global_flag)

    def test_unregistered_importlib_module_globals(self):
        # [FUNCTION DOC] A module loaded with importlib but not registered in the module table
        # can be compiled under fullgraph through a partial function that calls a helper;
        # result equals eager. Writes a temporary module file.
        module_name = "test_dynamo_unregistered_module_181243"
        self.assertNotIn(module_name, sys.modules)

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, f"{module_name}.py")
            with open(module_path, "w") as f:
                f.write(
                    """
import functools
import torch


def _helper(x, scale):
    return x.sin() * scale


def my_fn(x, scale):
    return _helper(x, scale)


fn = functools.partial(my_fn, scale=2)
"""
                )

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertNotIn(module_name, sys.modules)

            x = torch.randn(4, 4)
            compiled = torch.compile(mod.fn, backend="eager", fullgraph=True)
            self.assertTrue(same(compiled(x), mod.fn(x)))

    def test_store_global_in_unregistered_importlib_module(self):
        # [FUNCTION DOC] A function in an unregistered importlib module that stores a global
        # is unsupported under fullgraph (error mentions storing a global in non-module
        # globals), but works without fullgraph, updating the module flag to 2 and returning
        # the input plus 2. Writes a temporary module file.
        module_name = "test_dynamo_unregistered_store_global_181243"
        self.assertNotIn(module_name, sys.modules)

        with tempfile.TemporaryDirectory() as tmpdir:
            module_path = os.path.join(tmpdir, f"{module_name}.py")
            with open(module_path, "w") as f:
                f.write(
                    """
import functools


flag = 0


def my_fn(x, scale):
    global flag
    flag = scale
    return x + flag


fn = functools.partial(my_fn, scale=2)
"""
                )

            spec = importlib.util.spec_from_file_location(module_name, module_path)
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            mod = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(mod)
            self.assertNotIn(module_name, sys.modules)

            x = torch.randn(4, 4)
            with self.assertRaisesRegex(
                Unsupported, "STORE_GLOBAL in non-module globals"
            ):
                torch.compile(mod.fn, backend="eager", fullgraph=True)(x)

            mod.flag = 0
            compiled = torch.compile(mod.fn, backend="eager")
            self.assertTrue(same(compiled(x), x + 2))
            self.assertEqual(mod.flag, 2)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
