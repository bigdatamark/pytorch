# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests how dynamo traces the builtin id function. Four groups: sourceful objects
# (id equals eager id, and consuming it installs an identity-match guard so a different object
# recompiles), sourceless objects (a fake id that must graph break when it escapes into tensor
# ops, returns, branches or side effects), id used only as a dictionary key (must compile with
# a full graph), and argument-count errors. No import-time side effects.

import functools
import math
import unittest

import torch
import torch._dynamo.test_case
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import CompileCounter


class IdTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case holding all id tests plus three helpers that compile small
    # functions with the eager backend or a compile counter and check results, graph breaks or
    # recompilation counts.
    def _assert_id_equals(self, obj):
        # [FUNCTION DOC] HELPER: Compiles a function that returns the id of its second
        # argument (first argument is a dummy tensor) with the eager backend and a full graph,
        # and asserts the result equals the id computed in plain Python for obj.
        """Assert Dynamo's id(obj) matches eager Python's id(obj)."""
        expected = id(obj)

        def fn(_, o):
            # [FUNCTION DOC] HELPER: Ignores the first argument and returns the id of the
            # second.
            return id(o)

        result = torch.compile(fn, backend="eager", fullgraph=True)(
            torch.tensor(0), obj
        )
        self.assertEqual(result, expected)

    def _assert_id_graph_breaks(self, fn):
        # [FUNCTION DOC] HELPER: Asserts that compiling fn with the eager backend and a full
        # graph, called on a random 4-element tensor, raises dynamo's Unsupported error.
        """Assert fn graph-breaks under fullgraph=True due to FakeIdVariable."""
        with self.assertRaises(Unsupported):
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(4))

    # =====================================================================
    # Category 1: Sourceful objects — id() matches eager
    # =====================================================================

    def test_id_user_function(self):
        # [FUNCTION DOC] Checks the id of a user-defined local function is reproduced exactly
        # under full-graph compile.
        def target():
            # [FUNCTION DOC] HELPER: Empty function used only as the object whose id is taken.
            pass

        self._assert_id_equals(target)

    def test_id_lambda(self):
        # [FUNCTION DOC] Checks the id of a lambda matches eager under full-graph compile.
        self._assert_id_equals(lambda: None)

    def test_id_user_object(self):
        # [FUNCTION DOC] Checks the id of an instance of a user-defined class matches eager.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class used to create an object whose id is taken.
            pass

        self._assert_id_equals(MyObj())

    def test_id_module_import(self):
        # [FUNCTION DOC] Checks the id of the math module matches eager.
        self._assert_id_equals(math)

    def test_id_builtin_len(self):
        # [FUNCTION DOC] Checks the id of the builtin len function matches eager.
        self._assert_id_equals(len)

    def test_id_builtin_type_int(self):
        # [FUNCTION DOC] Checks the id of the builtin int type matches eager.
        self._assert_id_equals(int)

    def test_id_builtin_type_list(self):
        # [FUNCTION DOC] Checks the id of the builtin list type matches eager.
        self._assert_id_equals(list)

    def test_id_tensor(self):
        # [FUNCTION DOC] Checks the id of a tensor passed as an argument matches eager.
        self._assert_id_equals(torch.tensor([1.0, 2.0]))

    def test_id_nn_module(self):
        # [FUNCTION DOC] Checks the id of an nn.Linear module matches eager.
        self._assert_id_equals(torch.nn.Linear(2, 3))

    def test_id_user_defined_class(self):
        # [FUNCTION DOC] Checks the id of a user-defined class object (not an instance)
        # matches eager.
        class MyClass:
            # [CLASS DOC] HELPER: Empty class whose own id is taken.
            pass

        self._assert_id_equals(MyClass)

    def test_id_none(self):
        # [FUNCTION DOC] Checks that the id of the None constant inside a compiled function
        # equals the eager id of None.
        def fn(_):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the id of None.
            return id(None)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, id(None))

    def test_id_true(self):
        # [FUNCTION DOC] Checks that the id of the True constant inside a compiled function
        # equals the eager id of True.
        def fn(_):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the id of True.
            return id(True)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, id(True))

    def test_id_false(self):
        # [FUNCTION DOC] Checks that the id of the False constant inside a compiled function
        # equals the eager id of False.
        def fn(_):
            # [FUNCTION DOC] HELPER: Ignores its argument and returns the id of False.
            return id(False)

        result = torch.compile(fn, backend="eager", fullgraph=True)(torch.tensor(0))
        self.assertEqual(result, id(False))

    def test_id_functools_partial(self):
        # [FUNCTION DOC] Checks the id of a functools partial object (partial application of a
        # two-argument lambda) matches eager.
        p = functools.partial(lambda x, y: x + y, 1)
        self._assert_id_equals(p)

    def test_id_sourceful_dict(self):
        # [FUNCTION DOC] Checks the id of a dictionary argument matches eager.
        self._assert_id_equals({"a": 1, "b": 2})

    def test_id_sourceful_list(self):
        # [FUNCTION DOC] Checks the id of a list argument matches eager.
        self._assert_id_equals([1, 2, 3])

    def test_id_sourceful_tuple(self):
        # [FUNCTION DOC] Checks the id of a tuple argument matches eager.
        self._assert_id_equals((1, 2))

    def test_id_sourceful_int(self):
        # [FUNCTION DOC] Checks the id of an integer argument (42) matches eager.
        self._assert_id_equals(42)

    def test_id_sourceful_string(self):
        # [FUNCTION DOC] Checks the id of a string argument matches eager.
        self._assert_id_equals("hello")

    # =====================================================================
    # Category 2: Sourceful objects — ID_MATCH guard when consumed
    # =====================================================================

    def _assert_id_recompiles(self, fn, args1, args2):
        # [FUNCTION DOC] HELPER: Compiles fn with a compile-counting backend (default
        # settings, not full graph). Calls it with a dummy tensor plus the first argument list
        # and asserts one frame was compiled, then calls with the second argument list and
        # asserts the frame count rose to two, i.e. a recompile happened.
        """Assert fn(x, *args1) compiles, then fn(x, *args2) recompiles."""
        cnt = CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(torch.tensor(0), *args1)
        self.assertEqual(cnt.frame_count, 1)
        opt_fn(torch.tensor(0), *args2)
        self.assertEqual(cnt.frame_count, 2)

    def test_id_guards_graph_input(self):
        # [FUNCTION DOC] Checks that adding the id of an object argument to a tensor installs
        # a guard: a second distinct object triggers a recompile.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instances are passed as arguments.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the tensor plus the id of the object.
            return x + id(obj)

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_constant_op(self):
        # [FUNCTION DOC] Checks that using the id in constant arithmetic (doubling it) before
        # adding to a tensor still guards on the object and recompiles for a new instance.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instances are passed as arguments.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Doubles the id of the object and adds it to the tensor.
            k = id(obj) * 2
            return x + k

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_return(self):
        # [FUNCTION DOC] Checks that returning the id of an argument alongside a tensor result
        # guards on the object and recompiles for a new instance.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instances are passed as arguments.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Returns the tensor plus one and the id of the object.
            return x + 1.0, id(obj)

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_on_branch(self):
        # [FUNCTION DOC] Checks that branching on the parity of an object's id guards on the
        # object and recompiles for a new instance.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instances are passed as arguments.
            pass

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Adds one to the tensor if the id is even, otherwise adds
            # two.
            if id(obj) % 2 == 0:
                return x + 1
            return x + 2

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_guards_on_comparison(self):
        # [FUNCTION DOC] Checks that comparing the ids of two arguments guards: the first call
        # passes the same object twice, the second passes a different second object, and the
        # second call recompiles.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class used to make distinct objects.
            pass

        obj1 = MyObj()

        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: Adds one to the tensor if the two objects have equal ids,
            # otherwise adds two.
            if id(a) == id(b):
                return x + 1
            return x + 2

        self._assert_id_recompiles(fn, [obj1, obj1], [obj1, MyObj()])

    def test_id_guards_side_effect(self):
        # [FUNCTION DOC] Checks that appending an id to an outer-scope list (a side effect)
        # still guards on the object and recompiles for a new instance.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instances are passed as arguments.
            pass

        log = []

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Appends the object's id to the enclosing log list and
            # returns the tensor plus one.
            log.append(id(obj))
            return x + 1.0

        self._assert_id_recompiles(fn, [MyObj()], [MyObj()])

    def test_id_class_guard_uses_class_match(self):
        # [FUNCTION DOC] Checks that passing a different class object (id divided by 100000
        # added to a tensor) recompiles.
        class MyClass1:
            # [CLASS DOC] HELPER: Empty class passed as an argument.
            pass

        class MyClass2:
            # [CLASS DOC] HELPER: Second empty class passed as an argument to force a
            # recompile.
            pass

        def fn(x, cls):
            # [FUNCTION DOC] HELPER: Adds the integer-divided id of the class argument to the
            # tensor.
            return x + id(cls) // 100000

        self._assert_id_recompiles(fn, [MyClass1], [MyClass2])

    def test_id_nn_module_guard(self):
        # [FUNCTION DOC] Checks that comparing the id of the module itself against a passed-in
        # reference id (taken from a first module) recompiles when called through a second
        # module of the same type.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward multiplies the input by one if its own
            # id equals the supplied reference id, otherwise by zero.
            def forward(self, x, ref_id):
                # [FUNCTION DOC] HELPER: Returns the input multiplied by one if the module's
                # id equals ref_id, else multiplied by zero.
                if id(self) == ref_id:
                    return torch.mul(x, 1.0)
                return torch.mul(x, 0)

        m1 = M()
        m2 = M()

        def fn(x, m):
            # [FUNCTION DOC] HELPER: Calls the module with the id of the first module
            # (captured from the enclosing scope).
            return m(x, id(m1))

        self._assert_id_recompiles(fn, [m1], [m2])

    def test_id_tensor_guard(self):
        # [FUNCTION DOC] Checks that comparing the id of a tensor argument with the id of a
        # captured tensor guards on tensor identity: passing a different tensor recompiles.
        t1 = torch.ones(2)
        t2 = torch.zeros(2)

        def fn(x, t):
            # [FUNCTION DOC] HELPER: Multiplies the input by the captured first tensor if the
            # argument tensor is that same object, otherwise by the captured second tensor.
            if id(t) == id(t1):
                return torch.mul(x, t1)
            return torch.mul(x, t2)

        self._assert_id_recompiles(fn, [t1], [t2])

    def test_id_guarded_object(self):
        # [FUNCTION DOC] Checks that comparing an argument's id with a captured user object's
        # id guards on the object: passing a new instance recompiles.
        class UserDefinedObject:
            # [CLASS DOC] HELPER: Empty class whose instances are compared by id.
            pass

        obj1 = UserDefinedObject()

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Multiplies the tensor by one if the argument is the
            # captured object, otherwise by zero.
            if id(obj) == id(obj1):
                return torch.mul(x, 1.0)
            return torch.mul(x, 0)

        self._assert_id_recompiles(fn, [obj1], [UserDefinedObject()])

    def test_id_of_nn_module_op_count(self):
        # [FUNCTION DOC] Compiles an eval-mode module twice with fresh counters (resetting
        # dynamo in between), once passing the module's real id and once a wrong id, and
        # asserts the compiled graph has two operations when the ids match and one when they
        # do not, showing the id comparison is resolved at compile time.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward multiplies by one when its own id
            # equals the supplied id, then always adds one.
            def forward(self, x, ref_id):
                # [FUNCTION DOC] HELPER: Optionally multiplies by one (if id matches ref_id),
                # then adds one and returns the result.
                if id(self) == ref_id:
                    x = torch.mul(x, 1.0)
                x = torch.add(x, 1.0)
                return x

        m = M().eval()
        data = torch.randn(1)

        cnts = CompileCounter()
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, id(m))
        self.assertEqual(cnts.op_count, 2)

        torch._dynamo.reset()
        cnts = CompileCounter()
        opt_m = torch.compile(m, backend=cnts, fullgraph=True)
        opt_m(data, id(m) + 1)
        self.assertEqual(cnts.op_count, 1)

    # TODO: dict-key-only usage currently overguards (installs ID_MATCH eagerly).
    # Once LazyConstantVariable (PR #170644) lands, this test should pass with
    # frame_count == 1 (no recompile), and the expectedFailure can be removed.
    # Also add recompile tests for dict key usage once lazy constants are landed.
    @torch._dynamo.config.patch(error_on_recompile=False)
    @unittest.expectedFailure
    def test_id_dict_key_only_no_guard(self):
        # [FUNCTION DOC] Expected failure (recompile-on-error disabled via config patch): uses
        # the id of an object only as a dictionary key and expects a second distinct object
        # not to recompile, but currently a recompile happens so the final frame count check
        # fails.
        # KNOWN ISSUE: the test body asserts the desired behavior (no recompile) which the
        # code does not yet deliver; the expected-failure marker is meant to be removed once
        # lazy constants land.
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instances are passed as arguments.
            pass

        obj1 = MyObj()
        obj2 = MyObj()
        cnt = CompileCounter()

        def fn(x, obj):
            # [FUNCTION DOC] HELPER: Builds a dictionary keyed by the object's id and returns
            # the tensor plus the value looked up with the same id.
            d = {id(obj): 1}
            return x + d[id(obj)]

        opt_fn = torch.compile(fn, backend=cnt)
        opt_fn(torch.tensor(0), obj1)
        self.assertEqual(cnt.frame_count, 1)
        opt_fn(torch.tensor(0), obj2)
        # Dict-key-only usage should NOT recompile, but currently does
        self.assertEqual(cnt.frame_count, 1)

    # =====================================================================
    # Category 3: Sourceless objects — graph break when id escapes
    # =====================================================================

    def test_id_sourceless_return_graph_breaks(self):
        # [FUNCTION DOC] Checks that returning the id of a locally created list alongside a
        # tensor graph breaks (id of a sourceless object).
        self._assert_id_graph_breaks(lambda x: (x + 1.0, id([1, 2, 3])))

    def test_id_sourceless_branch_graph_breaks(self):
        # [FUNCTION DOC] Checks that branching on the parity of the id of a locally created
        # list graph breaks.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one or two to the tensor depending on whether the id
            # of a fresh list is even.
            if id([1, 2, 3]) % 2 == 0:
                return x + 1.0
            return x + 2.0

        self._assert_id_graph_breaks(fn)

    def test_id_sourceless_tensor_op_graph_breaks(self):
        # [FUNCTION DOC] Checks that adding the id of a locally created list to a tensor graph
        # breaks.
        self._assert_id_graph_breaks(lambda x: x + id([1, 2, 3]))

    def test_id_sourceless_side_effect_graph_breaks(self):
        # [FUNCTION DOC] Checks that appending the id of a locally created list to an outer
        # list graph breaks.
        log = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a list, appends its id to the enclosing log,
            # returns tensor plus one.
            obj = [1, 2, 3]
            log.append(id(obj))
            return x + 1.0

        self._assert_id_graph_breaks(fn)

    def test_id_sourceless_comparison(self):
        # [FUNCTION DOC] Checks that comparing the ids of two locally created lists compiles
        # (default non-fullgraph mode, falling back as needed) and gives the same result as
        # eager.
        """FakeIdVariable comparisons are not guaranteed to be sound (the
        compile-time fake id may not match runtime values), so we
        unconditionally graph-break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Creates two lists and adds one or two to the tensor
            # depending on whether their ids are equal.
            a = [1, 2]
            b = [3, 4]
            if id(a) == id(b):
                return x + 1.0
            return x + 2.0

        x = torch.randn(4)
        result = torch.compile(fn, backend="eager")(x)
        self.assertEqual(result, fn(x))

    # =====================================================================
    # Category 4: id() as dict key should not graph break
    # =====================================================================

    def _assert_id_dict_key_works(self, fn, *args):
        # [FUNCTION DOC] HELPER: Compiles fn with the eager backend and a full graph on the
        # given arguments, and asserts the compiled result equals the eager result.
        """Assert fn compiles with fullgraph=True and matches eager."""
        result = torch.compile(fn, backend="eager", fullgraph=True)(*args)
        self.assertEqual(result, fn(*args))

    def test_id_sourceless_int_as_key(self):
        # [FUNCTION DOC] Checks that the id of an integer literal used as a dictionary key and
        # membership test compiles with a full graph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dictionary keyed by an integer literal's id,
            # then adds one or two to the tensor depending on membership.
            d = {id(1000000): True}
            if id(1000000) in d:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_string_as_key(self):
        # [FUNCTION DOC] Checks that the id of a string literal as dictionary key and
        # membership test compiles with a full graph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Same as the integer variant but with a string literal.
            d = {id("hello"): True}
            if id("hello") in d:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_dict_as_key(self):
        # [FUNCTION DOC] Checks that the id of a locally created dictionary as a key in
        # another dictionary compiles with a full graph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Builds a dictionary, keys a memo by its id, and adds one
            # or two based on membership.
            d = {"a": 1, "b": 2}
            memo = {id(d): True}
            if id(d) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_partial_as_key(self):
        # [FUNCTION DOC] Checks that the id of a locally created functools partial object as a
        # dictionary key compiles with a full graph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a partial object, keys a dictionary by its id,
            # and adds one or two based on membership.
            p = functools.partial(lambda a, b: a + b, 1)
            d = {id(p): True}
            if id(p) in d:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceless_list_as_key(self):
        # [FUNCTION DOC] Checks that the id of a locally created list as a dictionary key
        # compiles with a full graph and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Creates a list, keys a memo by its id, and adds one or
            # two based on membership.
            lst = [1.0, 2.0]
            memo = {id(lst): True}
            if id(lst) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_function_as_dict_key(self):
        # [FUNCTION DOC] Checks that the id of a function passed as an argument can be a
        # dictionary key and looked up, with a full graph and matching eager results.
        def nothing():
            # [FUNCTION DOC] HELPER: Empty function passed as the argument whose id is used.
            pass

        def fn(x, f):
            # [FUNCTION DOC] HELPER: Keys a dictionary by the function's id and multiplies the
            # tensor by the looked-up value (3).
            d = {id(f): 3}
            return x * d[id(f)]

        self._assert_id_dict_key_works(fn, torch.randn(4), nothing)

    def test_id_functools_partial_as_dict_key(self):
        # [FUNCTION DOC] Checks that a captured functools partial (with keyword argument
        # bound) can be used as a dictionary key by id while also being called, with a full
        # graph and matching eager.
        def gn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return a + b

        partial_gn = functools.partial(gn, a=3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Keys a dictionary by the partial's id and returns the
            # partial called with the tensor times the looked-up value (5).
            d = {id(partial_gn): 5}
            return partial_gn(b=x) * d[id(partial_gn)]

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceful_dict_as_key(self):
        # [FUNCTION DOC] Checks that the id of a captured dictionary as a key and membership
        # test compiles with a full graph and matches eager.
        MY_DICT = {"a": 1, "b": 2}

        def fn(x):
            # [FUNCTION DOC] HELPER: Keys a memo by the id of the captured dictionary and adds
            # one or two based on membership.
            memo = {id(MY_DICT): True}
            if id(MY_DICT) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    def test_id_sourceful_list_as_key(self):
        # [FUNCTION DOC] Checks that the id of a captured list as a key and membership test
        # compiles with a full graph and matches eager.
        MY_LIST = [1.0, 2.0]

        def fn(x):
            # [FUNCTION DOC] HELPER: Keys a memo by the id of the captured list and adds one
            # or two based on membership.
            memo = {id(MY_LIST): True}
            if id(MY_LIST) in memo:
                return x + 1.0
            return x + 2.0

        self._assert_id_dict_key_works(fn, torch.randn(4))

    # =====================================================================
    # Category 5: Error handling
    # =====================================================================

    def test_id_wrong_arg_count(self):
        # [FUNCTION DOC] Checks that calling id with zero arguments and with two arguments
        # raises a TypeError that a wrapper catches inside the compiled function (returning
        # the input plus one); resets dynamo before each case and compares compiled to eager
        # output.
        def fn0(x):
            # [FUNCTION DOC] HELPER: Calls id with no arguments.
            return id()

        def fn2(x):
            # [FUNCTION DOC] HELPER: Calls id with two arguments.
            return id(x, x)

        for fn in [fn0, fn2]:
            torch._dynamo.reset()

            def wrapper(x):
                # [FUNCTION DOC] HELPER: Calls the selected failing function; on a TypeError
                # returns the input plus one.
                try:
                    return fn(x)
                except TypeError:
                    return x + 1.0

            x = torch.randn(4)
            result = torch.compile(wrapper, backend="eager", fullgraph=True)(x)
            self.assertEqual(result, wrapper(x))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
