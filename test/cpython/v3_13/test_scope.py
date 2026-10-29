# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_scope suite checking Python
# scoping rules: closures and free variables, nested functions and classes, global and
# nonlocal statements, cell handling, locals(), unbound-variable errors, exec/eval with
# free variables, trace functions and reference leaks. All tests live in one test case
# class. No import-time side effects besides class creation.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_scope.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests

# ======= END DYNAMO PATCH =======

import unittest
import weakref

from test.support import check_syntax_error, cpython_only
from test.support import gc_collect


class ScopeTests(CPythonTestCase):
    # [CLASS DOC] Test case holding all scoping tests. Most tests define small nested
    # functions or classes and assert on the values they return or the errors they raise;
    # some run code through exec strings and use check_syntax_error for compile-time errors.

    def testSimpleNesting(self):
        # [FUNCTION DOC] Closures: a function returned from a factory adds the captured value.

        def make_adder(x):
            # [FUNCTION DOC] HELPER: Return an inner function
            # adding the captured x to its argument.
            def adder(y):
                # [FUNCTION DOC] HELPER: Return x plus y.
                return x + y
            return adder

        inc = make_adder(1)
        plus10 = make_adder(10)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testExtraNesting(self):
        # [FUNCTION DOC] Free variables pass through an intermediate scope that does not use them.

        def make_adder2(x):
            # [FUNCTION DOC] HELPER: Return the adder built two levels down.
            def extra(): # check freevars passing through non-use scopes
                # [FUNCTION DOC] HELPER: Return an inner adder function closing over x.
                def adder(y):
                    # [FUNCTION DOC] HELPER: Return x plus y.
                    return x + y
                return adder
            return extra()

        inc = make_adder2(1)
        plus10 = make_adder2(10)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testSimpleAndRebinding(self):
        # [FUNCTION DOC] Reassigning the captured variable after the closure is defined is tracked
        # correctly.

        def make_adder3(x):
            # [FUNCTION DOC] HELPER: Define adder, then increment x, and return adder.
            def adder(y):
                # [FUNCTION DOC] HELPER: Return x plus y.
                return x + y
            x = x + 1 # check tracking of assignment to x in defining scope
            return adder

        inc = make_adder3(0)
        plus10 = make_adder3(9)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testNestingGlobalNoFree(self):
        # [FUNCTION DOC] Plain global variable reads work from deeply nested functions and see
        # updates to the global.

        def make_adder4(): # XXX add exta level of indirection
            # [FUNCTION DOC] HELPER: Return a three-level nested function adding global_x.
            def nest():
                # [FUNCTION DOC] HELPER: Middle nesting level; returns the next level's result.
                def nest():
                    # [FUNCTION DOC] HELPER: Inner nesting level; returns the adder.
                    def adder(y):
                        # [FUNCTION DOC] HELPER: Return global_x plus y.
                        return global_x + y # check that plain old globals work
                    return adder
                return nest()
            return nest()

        global_x = 1
        adder = make_adder4()
        self.assertEqual(adder(1), 2)

        global_x = 10
        self.assertEqual(adder(-2), 8)

    def testNestingThroughClass(self):
        # [FUNCTION DOC] Free variables are visible through a class body to its methods.

        def make_adder5(x):
            # [FUNCTION DOC] HELPER: Return an instance of a class whose call method adds x.
            class Adder:
                # [CLASS DOC] HELPER: Callable adding the captured x to its argument.
                def __call__(self, y):
                    # [FUNCTION DOC] HELPER: Return x plus y.
                    return x + y
            return Adder()

        inc = make_adder5(1)
        plus10 = make_adder5(10)

        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(-2), 8)

    def testNestingPlusFreeRefToGlobal(self):
        # [FUNCTION DOC] A function mixing a global declaration with a closure: all adders share
        # the single global.

        def make_adder6(x):
            # [FUNCTION DOC] HELPER: Set the global global_nest_x
            # to x and return a function reading it.
            global global_nest_x
            def adder(y):
                # [FUNCTION DOC] HELPER: Return the global global_nest_x plus y.
                return global_nest_x + y
            global_nest_x = x
            return adder

        inc = make_adder6(1)
        plus10 = make_adder6(10)

        self.assertEqual(inc(1), 11) # there's only one global
        self.assertEqual(plus10(-2), 8)

    def testNearestEnclosingScope(self):
        # [FUNCTION DOC] The nearest enclosing scope binding wins when names are shadowed.

        def f(x):
            # [FUNCTION DOC] HELPER: Return the result of g(2), a
            # function using the shadowed x of 42.
            def g(y):
                # [FUNCTION DOC] HELPER: Shadow x, then return inner function h.
                x = 42 # check that this masks binding in f()
                def h(z):
                    # [FUNCTION DOC] HELPER: Return x plus z.
                    return x + z
                return h
            return g(2)

        test_func = f(10)
        self.assertEqual(test_func(5), 47)

    def testMixedFreevarsAndCellvars(self):
        # [FUNCTION DOC] Mixed free variables and cell variables across nested scopes give the
        # right computed value.

        def identity(x):
            # [FUNCTION DOC] HELPER: Return its argument.
            return x

        def f(x, y, z):
            # [FUNCTION DOC] HELPER: Return function g, closing over x, y, z.
            def g(a, b, c):
                # [FUNCTION DOC] HELPER: Rebind a, define h, rebind y, return h.
                a = a + x # 3
                def h():
                    # [FUNCTION DOC] HELPER: Return identity of z times the sum of b and y.
                    # z * (4 + 9)
                    # 3 * 13
                    return identity(z * (b + y))
                y = c + z # 9
                return h
            return g

        g = f(1, 2, 3)
        h = g(2, 4, 6)
        self.assertEqual(h(), 39)

    def testFreeVarInMethod(self):
        # [FUNCTION DOC] Method names and class-level names do not hide enclosing-function
        # variables from methods, whether the class is nested in a function or at the top level.

        def test():
            # [FUNCTION DOC] HELPER: Build an instance of a class nested in a function.
            method_and_var = "var"
            class Test:
                # [CLASS DOC] HELPER: Class whose method names clash
                # with a free variable and builtin str.
                def method_and_var(self):
                    # [FUNCTION DOC] HELPER: Return the string method.
                    return "method"
                def test(self):
                    # [FUNCTION DOC] HELPER: Return the enclosing variable method_and_var.
                    return method_and_var
                def actual_global(self):
                    # [FUNCTION DOC] HELPER: Return the string global, using the builtin str.
                    return str("global")
                def str(self):
                    # [FUNCTION DOC] HELPER: Method named str that would
                    # recurse if the name were resolved to it.
                    return str(self)
            return Test()

        t = test()
        self.assertEqual(t.test(), "var")
        self.assertEqual(t.method_and_var(), "method")
        self.assertEqual(t.actual_global(), "global")

        method_and_var = "var"
        class Test:
            # [CLASS DOC] HELPER: Top-level (non-nested) variant
            # of the class above; different scoping
            # rules apply to its names.
            # this class is not nested, so the rules are different
            def method_and_var(self):
                # [FUNCTION DOC] HELPER: Return the string method.
                return "method"
            def test(self):
                # [FUNCTION DOC] HELPER: Return the global-level method_and_var.
                return method_and_var
            def actual_global(self):
                # [FUNCTION DOC] HELPER: Return the string global, using the builtin str.
                return str("global")
            def str(self):
                # [FUNCTION DOC] HELPER: Method named str that would
                # recurse if the name were resolved to it.
                return str(self)

        t = Test()
        self.assertEqual(t.test(), "var")
        self.assertEqual(t.method_and_var(), "method")
        self.assertEqual(t.actual_global(), "global")

    def testCellIsKwonlyArg(self):
        # [FUNCTION DOC] Issue 1409: a cell initialised from a keyword-only parameter works, with
        # default and explicit values.
        # Issue 1409: Initialisation of a cell value,
        # when it comes from a keyword-only parameter
        def foo(*, a=17):
            # [FUNCTION DOC] HELPER: Return the inner result plus 3.
            def bar():
                # [FUNCTION DOC] HELPER: Return a plus 5.
                return a + 5
            return bar() + 3

        self.assertEqual(foo(a=42), 50)
        self.assertEqual(foo(), 25)

    def testCellIsArgAndEscapes(self):
        # [FUNCTION DOC] A cell passed as an argument and captured by an inner function is wrapped
        # in a fresh cell; the function sees the original cell object, not the closure cell.
        # We need to be sure that a cell passed in as an arg still
        # gets wrapped in a new cell if the arg escapes into an
        # inner function (closure).

        def external():
            # [FUNCTION DOC] HELPER: Create a variable and return the cell from an inner closure.
            value = 42
            def inner():
                # [FUNCTION DOC] HELPER: Return the captured value.
                return value
            cell, = inner.__closure__
            return cell
        cell_ext = external()

        def spam(arg):
            # [FUNCTION DOC] HELPER: Return a function returning arg.
            def eggs():
                # [FUNCTION DOC] HELPER: Return the captured arg.
                return arg
            return eggs

        eggs = spam(cell_ext)
        cell_closure, = eggs.__closure__
        cell_eggs = eggs()

        self.assertIs(cell_eggs, cell_ext)
        self.assertIsNot(cell_eggs, cell_closure)

    def testCellIsLocalAndEscapes(self):
        # [FUNCTION DOC] Same as above but the cell is bound to a local variable instead of an arg.
        # We need to be sure that a cell bound to a local still
        # gets wrapped in a new cell if the local escapes into an
        # inner function (closure).

        def external():
            # [FUNCTION DOC] HELPER: Create a variable and return the cell from an inner closure.
            value = 42
            def inner():
                # [FUNCTION DOC] HELPER: Return the captured value.
                return value
            cell, = inner.__closure__
            return cell
        cell_ext = external()

        def spam(arg):
            # [FUNCTION DOC] HELPER: Return a function returning a local bound to arg.
            cell = arg
            def eggs():
                # [FUNCTION DOC] HELPER: Return the captured cell variable.
                return cell
            return eggs

        eggs = spam(cell_ext)
        cell_closure, = eggs.__closure__
        cell_eggs = eggs()

        self.assertIs(cell_eggs, cell_ext)
        self.assertIsNot(cell_eggs, cell_closure)

    def testRecursion(self):
        # [FUNCTION DOC] Recursion through a nested function works: factorial of 6 is 720.

        def f(x):
            # [FUNCTION DOC] HELPER: Return the factorial of x via inner recursive fact; raises
            # ValueError for negative x.
            def fact(n):
                # [FUNCTION DOC] HELPER: Recursive factorial referencing itself as a free variable.
                if n == 0:
                    return 1
                else:
                    return n * fact(n - 1)
            if x >= 0:
                return fact(x)
            else:
                raise ValueError("x must be >= 0")

        self.assertEqual(f(6), 720)


    def testUnoptimizedNamespaces(self):
        # [FUNCTION DOC] Checks that mixing a star import with free or global names in nested
        # functions is rejected at compile time (syntax error) in four code samples.

        check_syntax_error(self, """if 1:
            def unoptimized_clash1(strip):
                def f(s):
                    from sys import *
                    return getrefcount(s) # ambiguity: free or local
                return f
            """)

        check_syntax_error(self, """if 1:
            def unoptimized_clash2():
                from sys import *
                def f(s):
                    return getrefcount(s) # ambiguity: global or local
                return f
            """)

        check_syntax_error(self, """if 1:
            def unoptimized_clash2():
                from sys import *
                def g():
                    def f(s):
                        return getrefcount(s) # ambiguity: global or local
                    return f
            """)

        check_syntax_error(self, """if 1:
            def f():
                def g():
                    from sys import *
                    return getrefcount # global or local?
            """)

    def testLambdas(self):
        # [FUNCTION DOC] Checks nested lambdas close over variables and globals correctly.

        f1 = lambda x: lambda y: x + y
        inc = f1(1)
        plus10 = f1(10)
        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(5), 15)

        f2 = lambda x: (lambda : lambda y: x + y)()
        inc = f2(1)
        plus10 = f2(10)
        self.assertEqual(inc(1), 2)
        self.assertEqual(plus10(5), 15)

        f3 = lambda x: lambda y: global_x + y
        global_x = 1
        inc = f3(None)
        self.assertEqual(inc(2), 3)

        f8 = lambda x, y, z: lambda a, b, c: lambda : z * (b + y)
        g = f8(1, 2, 3)
        h = g(2, 4, 6)
        self.assertEqual(h(), 18)

    def testUnboundLocal(self):
        # [FUNCTION DOC] Reading a variable before assignment raises UnboundLocalError in the owner
        # scope and NameError when accessed from an inner function.

        def errorInOuter():
            # [FUNCTION DOC] HELPER: Prints y before its assignment; raises UnboundLocalError.
            print(y)
            def inner():
                # [FUNCTION DOC] HELPER: Closure returning y.
                return y
            y = 1

        def errorInInner():
            # [FUNCTION DOC] HELPER: Calls an inner function that reads y before it is assigned.
            def inner():
                # [FUNCTION DOC] HELPER: Closure returning y.
                return y
            inner()
            y = 1

        self.assertRaises(UnboundLocalError, errorInOuter)
        self.assertRaises(NameError, errorInInner)

    def testUnboundLocal_AfterDel(self):
        # [FUNCTION DOC] A cell variable may be deleted (issue 4617); later access raises
        # UnboundLocalError or NameError as appropriate.
        # #4617: It is now legal to delete a cell variable.
        # The following functions must obviously compile,
        # and give the correct error when accessing the deleted name.
        def errorInOuter():
            # [FUNCTION DOC] HELPER: Assign then delete y, then print it; raises UnboundLocalError.
            y = 1
            del y
            print(y)
            def inner():
                # [FUNCTION DOC] HELPER: Closure returning y.
                return y

        def errorInInner():
            # [FUNCTION DOC] HELPER: Calls an inner function reading y after it was deleted.
            def inner():
                # [FUNCTION DOC] HELPER: Closure returning y.
                return y
            y = 1
            del y
            inner()

        self.assertRaises(UnboundLocalError, errorInOuter)
        self.assertRaises(NameError, errorInInner)

    def testUnboundLocal_AugAssign(self):
        # [FUNCTION DOC] Bug 1501934: executes code where an augmented assignment to a name
        # declared only at top level fails with UnboundLocalError; fails the test otherwise.
        # Side effect: runs source text through exec in a throwaway namespace.
        # test for bug #1501934: incorrect LOAD/STORE_GLOBAL generation
        exec("""if 1:
            global_x = 1
            def f():
                global_x += 1
            try:
                f()
            except UnboundLocalError:
                pass
            else:
                fail('scope of global_x not correctly determined')
            """, {'fail': self.fail})

    def testComplexDefinitions(self):
        # [FUNCTION DOC] Closures over variable positional and keyword arguments return the
        # captured tuple and mapping.

        def makeReturner(*lst):
            # [FUNCTION DOC] HELPER: Return a function returning the captured positional args.
            def returner():
                # [FUNCTION DOC] HELPER: Return the captured tuple.
                return lst
            return returner

        self.assertEqual(makeReturner(1,2,3)(), (1,2,3))

        def makeReturner2(**kwargs):
            # [FUNCTION DOC] HELPER: Return a function returning the captured keyword args.
            def returner():
                # [FUNCTION DOC] HELPER: Return the captured mapping.
                return kwargs
            return returner

        self.assertEqual(makeReturner2(a=11)()['a'], 11)

    def testScopeOfGlobalStmt(self):
        # [FUNCTION DOC] Runs four classic examples of
        # global-statement scoping (Pedroni, 2001) plus
        # a class-body global case through exec, asserting the resulting values. Side effect:
        # exec of source text with the test case in scope.
        # Examples posted by Samuele Pedroni to python-dev on 3/1/2001

        exec("""if 1:
            # I
            x = 7
            def f():
                x = 1
                def g():
                    global x
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 7)
            self.assertEqual(x, 7)

            # II
            x = 7
            def f():
                x = 1
                def g():
                    x = 2
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 2)
            self.assertEqual(x, 7)

            # III
            x = 7
            def f():
                x = 1
                def g():
                    global x
                    x = 2
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 2)
            self.assertEqual(x, 2)

            # IV
            x = 7
            def f():
                x = 3
                def g():
                    global x
                    x = 2
                    def i():
                        def h():
                            return x
                        return h()
                    return i()
                return g()
            self.assertEqual(f(), 2)
            self.assertEqual(x, 2)

            # XXX what about global statements in class blocks?
            # do they affect methods?

            x = 12
            class Global:
                global x
                x = 13
                def set(self, val):
                    x = val
                def get(self):
                    return x

            g = Global()
            self.assertEqual(g.get(), 13)
            g.set(15)
            self.assertEqual(g.get(), 13)
            """)

    def testLeaks(self):
        # [FUNCTION DOC] Creates many objects captured by closures and checks they are all freed
        # (a live-instance counter returns to zero after a garbage collection).

        class Foo:
            # [CLASS DOC] HELPER: Counts live instances in the class attribute count.
            count = 0

            def __init__(self):
                # [FUNCTION DOC] HELPER: Increment the instance count.
                Foo.count += 1

            def __del__(self):
                # [FUNCTION DOC] HELPER: Decrement the instance count when the object is destroyed.
                Foo.count -= 1

        def f1():
            # [FUNCTION DOC] HELPER: Create a Foo captured by an
            # inner function, call it, and drop it.
            x = Foo()
            def f2():
                # [FUNCTION DOC] HELPER: Return the captured Foo.
                return x
            f2()

        for i in range(100):
            f1()

        gc_collect()  # For PyPy or other GCs.
        self.assertEqual(Foo.count, 0)

    def testClassAndGlobal(self):
        # [FUNCTION DOC] Runs code through exec checking a global declaration inside a class
        # body affects methods, and that implicit globals in a class body are looked up by name
        # lookup, not as true globals. Side effect: exec of source text.

        exec("""if 1:
            def test(x):
                class Foo:
                    global x
                    def __call__(self, y):
                        return x + y
                return Foo()

            x = 0
            self.assertEqual(test(6)(2), 8)
            x = -1
            self.assertEqual(test(3)(2), 5)

            looked_up_by_load_name = False
            class X:
                # Implicit globals inside classes are be looked up by LOAD_NAME, not
                # LOAD_GLOBAL.
                locals()['looked_up_by_load_name'] = True
                passed = looked_up_by_load_name

            self.assertTrue(X.passed)
            """)

    def testLocalsFunction(self):
        # [FUNCTION DOC] locals() in a nested function returns the parameters, locals and free
        # variables as a mapping, including the inner function h.

        def f(x):
            # [FUNCTION DOC] HELPER: Return function g whose result is the locals mapping.
            def g(y):
                # [FUNCTION DOC] HELPER: Rebind y and return the locals mapping.
                def h(z):
                    # [FUNCTION DOC] HELPER: Return y plus z.
                    return y + z
                w = x + y
                y += 3
                return locals()
            return g

        d = f(2)(4)
        self.assertIn('h', d)
        del d['h']
        self.assertEqual(d, {'x': 2, 'y': 7, 'w': 6})

    def testLocalsClass(self):
        # [FUNCTION DOC] Calling locals() in a class body must not add free variables to the class
        # namespace; checks both a plain free variable and a class variable derived from it.
        # This test verifies that calling locals() does not pollute
        # the local namespace of the class with free variables.  Old
        # versions of Python had a bug, where a free variable being
        # passed through a class namespace would be inserted into
        # locals() by locals() or exec or a trace function.
        #
        # The real bug lies in frame code that copies variables
        # between fast locals and the locals dict, e.g. when executing
        # a trace function.

        def f(x):
            # [FUNCTION DOC] HELPER: Return a class with
            # attribute x set to 12 that also calls locals.
            class C:
                # [CLASS DOC] HELPER: Class using locals() in its
                # body while a free variable shares a name.
                x = 12
                def m(self):
                    # [FUNCTION DOC] HELPER: Return the free variable x.
                    return x
                locals()
            return C

        self.assertEqual(f(1).x, 12)

        def f(x):
            # [FUNCTION DOC] HELPER: Return a class that records the list of its locals names in z.
            class C:
                # [CLASS DOC] HELPER: Class capturing the list of local names in z.
                y = x
                def m(self):
                    # [FUNCTION DOC] HELPER: Return the free variable x.
                    return x
                z = list(locals())
            return C

        varnames = f(1).z
        self.assertNotIn("x", varnames)
        self.assertIn("y", varnames)

    @cpython_only
    def testLocalsClass_WithTrace(self):
        # [FUNCTION DOC] Issue 23728: with a trace function installed, the locals update must not
        # break a free variable referenced from a class method. Restores the previous trace
        # function at cleanup. Only runs on CPython.
        # Issue23728: after the trace function returns, the locals()
        # dictionary is used to update all variables, this used to
        # include free variables. But in class statements, free
        # variables are not inserted...
        import sys
        self.addCleanup(sys.settrace, sys.gettrace())
        sys.settrace(lambda a,b,c:None)
        x = 12

        class C:
            # [CLASS DOC] HELPER: Class whose method references an enclosing variable.
            def f(self):
                # [FUNCTION DOC] HELPER: Return the free variable x.
                return x

        self.assertEqual(x, 12) # Used to raise UnboundLocalError

    def testBoundAndFree(self):
        # [FUNCTION DOC] A variable that is both bound in a class body and free in a method
        # works: the class attribute equals the method result.
        # var is bound and free in class

        def f(x):
            # [FUNCTION DOC] HELPER: Return a class with a method
            # returning x and attribute a set to x.
            class C:
                # [CLASS DOC] HELPER: Class with both a method
                # closing over x and an attribute set from x.
                def m(self):
                    # [FUNCTION DOC] HELPER: Return the free variable x.
                    return x
                a = x
            return C

        inst = f(3)()
        self.assertEqual(inst.a, inst.m())

    @cpython_only
    def testInteractionWithTraceFunc(self):
        # [FUNCTION DOC] Checks a trace function installed with settrace does not crash scope
        # handling while a getter adapter runs; also settrace without arguments raises
        # TypeError. Restores the previous trace function. Only runs on CPython.

        import sys
        def tracer(a,b,c):
            # [FUNCTION DOC] HELPER: Trace function that returns itself.
            return tracer

        def adaptgetter(name, klass, getter):
            # [FUNCTION DOC] HELPER: Build a getter function for
            # kind 1 descriptors with an empty name,
            # using a mangled name; returns None for other kinds.
            kind, des = getter
            if kind == 1:       # AV happens when stepping from this line to next
                if des == "":
                    des = "_%s__%s" % (klass.__name__, name)
                return lambda obj: getattr(obj, des)

        class TestClass:
            # [CLASS DOC] HELPER: Empty class used as the argument of the adapter.
            pass

        self.addCleanup(sys.settrace, sys.gettrace())
        sys.settrace(tracer)
        adaptgetter("foo", TestClass, (1, ""))
        sys.settrace(None)

        self.assertRaises(TypeError, sys.settrace)

    def testEvalExecFreeVars(self):
        # [FUNCTION DOC] Evaluating or executing code objects that have free variables must fail
        # with TypeError.

        def f(x):
            # [FUNCTION DOC] HELPER: Return a function closing over x.
            return lambda: x + 1

        g = f(3)
        self.assertRaises(TypeError, eval, g.__code__)

        try:
            exec(g.__code__, {})
        except TypeError:
            pass
        else:
            self.fail("exec should have failed, because code contained free vars")

    def testListCompLocalVars(self):
        # [FUNCTION DOC] A list comprehension variable does not leak out to the enclosing scope.

        try:
            print(bad)
        except NameError:
            pass
        else:
            print("bad should not be defined")

        def x():
            # [FUNCTION DOC] HELPER: Run a list comprehension that binds bad and s.
            [bad for s in 'a b' for bad in s.split()]

        x()
        try:
            print(bad)
        except NameError:
            pass

    def testEvalFreeVars(self):
        # [FUNCTION DOC] eval of an expression naming a free variable inside an inner function
        # works.

        def f(x):
            # [FUNCTION DOC] HELPER: Return function g which references x and evaluates a string.
            def g():
                # [FUNCTION DOC] HELPER: Refer to x, then eval the expression x plus 1.
                x
                eval("x + 1")
            return g

        f(4)()

    def testFreeingCell(self):
        # [FUNCTION DOC] Defines a class with a finalizer that calls an undefined helper; the
        # class is never instantiated, so the test only checks definition succeeds.
        # KNOWN ISSUE: Special is never instantiated and nestedcell_get is undefined, so the
        # test body checks nothing at runtime.
        # Test what happens when a finalizer accesses
        # the cell where the object was stored.
        class Special:
            # [CLASS DOC] HELPER: Class whose finalizer calls the undefined name nestedcell_get.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Finalizer that calls nestedcell_get (never executed).
                nestedcell_get()

    def testNonLocalFunction(self):
        # [FUNCTION DOC] nonlocal increments and decrements a shared captured counter.

        def f(x):
            # [FUNCTION DOC] HELPER: Return an increment and a decrement function sharing x.
            def inc():
                # [FUNCTION DOC] HELPER: Increase the captured x by one and return it.
                nonlocal x
                x += 1
                return x
            def dec():
                # [FUNCTION DOC] HELPER: Decrease the captured x by one and return it.
                nonlocal x
                x -= 1
                return x
            return inc, dec

        inc, dec = f(0)
        self.assertEqual(inc(), 1)
        self.assertEqual(inc(), 2)
        self.assertEqual(dec(), 1)
        self.assertEqual(dec(), 0)

    def testNonLocalMethod(self):
        # [FUNCTION DOC] nonlocal inside methods of a class defined in a function updates the
        # function's variable.
        def f(x):
            # [FUNCTION DOC] HELPER: Return an instance of a class with inc and dec methods on x.
            class c:
                # [CLASS DOC] HELPER: Class whose methods change the enclosing variable x.
                def inc(self):
                    # [FUNCTION DOC] HELPER: Increase the captured x by one and return it.
                    nonlocal x
                    x += 1
                    return x
                def dec(self):
                    # [FUNCTION DOC] HELPER: Decrease the captured x by one and return it.
                    nonlocal x
                    x -= 1
                    return x
            return c()
        c = f(0)
        self.assertEqual(c.inc(), 1)
        self.assertEqual(c.inc(), 2)
        self.assertEqual(c.dec(), 1)
        self.assertEqual(c.dec(), 0)

    def testGlobalInParallelNestedFunctions(self):
        # [FUNCTION DOC] A global statement in one nested function must not leak into a sibling
        # function. Executes source text in separate local and global namespaces and checks
        # results 9 and 2.
        # A symbol table bug leaked the global statement from one
        # function to other nested functions in the same block.
        # This test verifies that a global statement in the first
        # function does not affect the second function.
        local_ns = {}
        global_ns = {}
        exec("""if 1:
            def f():
                y = 1
                def g():
                    global y
                    return y
                def h():
                    return y + 1
                return g, h
            y = 9
            g, h = f()
            result9 = g()
            result2 = h()
            """, local_ns, global_ns)
        self.assertEqual(2, global_ns["result2"])
        self.assertEqual(9, global_ns["result9"])

    def testNonLocalClass(self):
        # [FUNCTION DOC] A nonlocal statement in a class body updates the enclosing variable
        # without creating a class attribute.

        def f(x):
            # [FUNCTION DOC] HELPER: Return an instance of a class that increments x in its body.
            class c:
                # [CLASS DOC] HELPER: Class whose body increments the enclosing x with nonlocal.
                nonlocal x
                x += 1
                def get(self):
                    # [FUNCTION DOC] HELPER: Return the free variable x.
                    return x
            return c()

        c = f(0)
        self.assertEqual(c.get(), 1)
        self.assertNotIn("x", c.__class__.__dict__)


    def testNonLocalGenerator(self):
        # [FUNCTION DOC] nonlocal works in a generator: yields increasing counter values.

        def f(x):
            # [FUNCTION DOC] HELPER: Return a generator function incrementing and yielding x.
            def g(y):
                # [FUNCTION DOC] HELPER: Generator yielding x after each of y increments.
                nonlocal x
                for i in range(y):
                    x += 1
                    yield x
            return g

        g = f(0)
        self.assertEqual(list(g(5)), [1, 2, 3, 4, 5])

    def testNestedNonLocal(self):
        # [FUNCTION DOC] Nested nonlocal statements across two
        # levels combine: 1 minus 2 plus 4 is 3.

        def f(x):
            # [FUNCTION DOC] HELPER: Return function g.
            def g():
                # [FUNCTION DOC] HELPER: Subtract 2 from x and return function h.
                nonlocal x
                x -= 2
                def h():
                    # [FUNCTION DOC] HELPER: Add 4 to x and return it.
                    nonlocal x
                    x += 4
                    return x
                return h
            return g

        g = f(1)
        h = g()
        self.assertEqual(h(), 3)

    def testTopIsNotSignificant(self):
        # [FUNCTION DOC] Issue 9997: a function-level name at
        # the top does not matter; checks that a
        # global statement for a name that is also a parameter of a sibling function compiles.
        # See #9997.
        def top(a):
            # [FUNCTION DOC] HELPER: Function with parameter a; does nothing.
            pass
        def b():
            # [FUNCTION DOC] HELPER: Function declaring a global named a; does nothing.
            global a

    def testClassNamespaceOverridesClosure(self):
        # [FUNCTION DOC] Issue 17853: assigning to the class namespace through locals overrides the
        # enclosing closure value, and deleting it hides it; the outer x is unchanged.
        # See #17853.
        x = 42
        class X:
            # [CLASS DOC] HELPER: Class that sets x to 43 in its namespace and copies it into y.
            locals()["x"] = 43
            y = x
        self.assertEqual(X.y, 43)
        class X:
            # [CLASS DOC] HELPER: Class that sets x in its namespace to 43 and then deletes it.
            locals()["x"] = 43
            del x
        self.assertFalse(hasattr(X, "x"))
        self.assertEqual(x, 42)

    @cpython_only
    def testCellLeak(self):
        # [FUNCTION DOC] Issue 17927: a method holding a cycle through a frame and a nested lambda
        # referencing self must still be collectable after setting self to None. Only runs on
        # CPython; uses a weak reference and garbage collection.
        # Issue 17927.
        #
        # The issue was that if self was part of a cycle involving the
        # frame of a method call, *and* the method contained a nested
        # function referencing self, thereby forcing 'self' into a
        # cell, setting self to None would not be enough to break the
        # frame -- the frame had another reference to the instance,
        # which could not be cleared by the code running in the frame
        # (though it will be cleared when the frame is collected).
        # Without the lambda, setting self to None is enough to break
        # the cycle.
        class Tester:
            # [CLASS DOC] HELPER: Class whose method creates a reference cycle through a caught
            # exception and closes over self.
            def dig(self):
                # [FUNCTION DOC] HELPER: Raise and catch
                # ZeroDivisionError, store it on self, then set the
                # local self to None to break the cycle.
                if 0:
                    lambda: self
                try:
                    1/0
                except Exception as exc:
                    self.exc = exc
                self = None  # Break the cycle
        tester = Tester()
        tester.dig()
        ref = weakref.ref(tester)
        del tester
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(ref())

    def test_multiple_nesting(self):
        # [FUNCTION DOC] Regression test for cpython issue 121863: name mangling of double-
        # underscore arguments in doubly nested classes; the call with a mangled keyword raises
        # TypeError both directly and via the returned bound method.
        # Regression test for https://github.com/python/cpython/issues/121863
        class MultiplyNested:
            # [CLASS DOC] HELPER: Class whose methods create
            # nested classes using a double-underscore
            # argument name.
            def f1(self):
                # [FUNCTION DOC] HELPER: Call the nested method with
                # the mangled keyword; raises TypeError.
                __arg = 1
                class D:
                    # [CLASS DOC] HELPER: Nested class with method g taking a mangled argument.
                    def g(self, __arg):
                        # [FUNCTION DOC] HELPER: Return its argument.
                        return __arg
                return D().g(_MultiplyNested__arg=2)

            def f2(self):
                # [FUNCTION DOC] HELPER: Return the bound method g of a nested class instance.
                __arg = 1
                class D:
                    # [CLASS DOC] HELPER: Nested class with method g taking a mangled argument.
                    def g(self, __arg):
                        # [FUNCTION DOC] HELPER: Return its argument.
                        return __arg
                return D().g

        inst = MultiplyNested()
        with self.assertRaises(TypeError):
            inst.f1()

        closure = inst.f2()
        with self.assertRaises(TypeError):
            closure(_MultiplyNested__arg=2)

if __name__ == "__main__":
    run_tests()
