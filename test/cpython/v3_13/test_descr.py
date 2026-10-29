# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Dynamo-patched copy of CPython 3.13 test_descr: tests for descriptors, the class
# and type model (operators, metaclasses, slots, properties, MRO, super, pickling), run under the
# CPythonTestCase harness. Importing defines the test classes and imports optional helpers
# (_testcapi, xxsubtype) falling back to None; run_tests is invoked only when run as a script.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_descr.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO

# ======= END DYNAMO PATCH =======

import builtins
import copyreg
import gc
import itertools
import math
import pickle
import random
import string
import sys
import types
import unittest
import warnings
import weakref

from copy import deepcopy
from contextlib import redirect_stdout
from test import support
from test.support.testcase import ExtraAssertions

try:
    import _testcapi
except ImportError:
    _testcapi = None

try:
    import xxsubtype
except ImportError:
    xxsubtype = None


class OperatorsTest(CPythonTestCase):
    # [CLASS DOC] Test case checking that operators on built-in and user-defined types dispatch to
    # the matching special methods; helper methods run an expression and its special-method form
    # and compare results.

    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: constructor. Arguments: *args, **kwargs. Stores the given values
        # on the instance, and for test cases initializes the base CPythonTestCase.
        CPythonTestCase.__init__(self, *args, **kwargs)
        self.binops = {
            'add': '+',
            'sub': '-',
            'mul': '*',
            'matmul': '@',
            'truediv': '/',
            'floordiv': '//',
            'divmod': 'divmod',
            'pow': '**',
            'lshift': '<<',
            'rshift': '>>',
            'and': '&',
            'xor': '^',
            'or': '|',
            'cmp': 'cmp',
            'lt': '<',
            'le': '<=',
            'eq': '==',
            'ne': '!=',
            'gt': '>',
            'ge': '>=',
        }

        for name, expr in list(self.binops.items()):
            if expr.islower():
                expr = expr + "(a, b)"
            else:
                expr = 'a %s b' % expr
            self.binops[name] = expr

        self.unops = {
            'pos': '+',
            'neg': '-',
            'abs': 'abs',
            'invert': '~',
            'int': 'int',
            'float': 'float',
        }

        for name, expr in list(self.unops.items()):
            if expr.islower():
                expr = expr + "(a)"
            else:
                expr = '%s a' % expr
            self.unops[name] = expr

    def unop_test(self, a, res, expr="len(a)", meth="__len__"):
        # [FUNCTION DOC] HELPER: unop_test of OperatorsTest. Arguments: a, res, expr, meth.
        # Supporting routine for tests; returns a computed value or None.
        d = {'a': a}
        self.assertEqual(eval(expr, d), res)
        t = type(a)
        m = getattr(t, meth)

        # Find method in parent class
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        self.assertEqual(m(a), res)
        bm = getattr(a, meth)
        self.assertEqual(bm(), res)

    def binop_test(self, a, b, res, expr="a+b", meth="__add__"):
        # [FUNCTION DOC] HELPER: binop_test of OperatorsTest. Arguments: a, b, res, expr, meth.
        # Supporting routine for tests; returns a computed value or None.
        d = {'a': a, 'b': b}

        self.assertEqual(eval(expr, d), res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        self.assertEqual(m(a, b), res)
        bm = getattr(a, meth)
        self.assertEqual(bm(b), res)

    def sliceop_test(self, a, b, c, res, expr="a[b:c]", meth="__getitem__"):
        # [FUNCTION DOC] HELPER: sliceop_test of OperatorsTest. Arguments: a, b, c, res, expr,
        # meth. Supporting routine for tests; returns a computed value or None.
        d = {'a': a, 'b': b, 'c': c}
        self.assertEqual(eval(expr, d), res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        self.assertEqual(m(a, slice(b, c)), res)
        bm = getattr(a, meth)
        self.assertEqual(bm(slice(b, c)), res)

    def setop_test(self, a, b, res, stmt="a+=b", meth="__iadd__"):
        # [FUNCTION DOC] HELPER: setop_test of OperatorsTest. Arguments: a, b, res, stmt, meth.
        # Supporting routine for tests; returns a computed value or None.
        d = {'a': deepcopy(a), 'b': b}
        exec(stmt, d)
        self.assertEqual(d['a'], res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        d['a'] = deepcopy(a)
        m(d['a'], b)
        self.assertEqual(d['a'], res)
        d['a'] = deepcopy(a)
        bm = getattr(d['a'], meth)
        bm(b)
        self.assertEqual(d['a'], res)

    def set2op_test(self, a, b, c, res, stmt="a[b]=c", meth="__setitem__"):
        # [FUNCTION DOC] HELPER: set2op_test of OperatorsTest. Arguments: a, b, c, res, stmt,
        # meth. Supporting routine for tests; returns a computed value or None.
        d = {'a': deepcopy(a), 'b': b, 'c': c}
        exec(stmt, d)
        self.assertEqual(d['a'], res)
        t = type(a)
        m = getattr(t, meth)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        d['a'] = deepcopy(a)
        m(d['a'], b, c)
        self.assertEqual(d['a'], res)
        d['a'] = deepcopy(a)
        bm = getattr(d['a'], meth)
        bm(b, c)
        self.assertEqual(d['a'], res)

    def setsliceop_test(self, a, b, c, d, res, stmt="a[b:c]=d", meth="__setitem__"):
        # [FUNCTION DOC] HELPER: setsliceop_test of OperatorsTest. Arguments: a, b, c, d, res,
        # stmt, meth. Supporting routine for tests; returns a computed value or None.
        dictionary = {'a': deepcopy(a), 'b': b, 'c': c, 'd': d}
        exec(stmt, dictionary)
        self.assertEqual(dictionary['a'], res)
        t = type(a)
        while meth not in t.__dict__:
            t = t.__bases__[0]
        m = getattr(t, meth)
        # in some implementations (e.g. PyPy), 'm' can be a regular unbound
        # method object; the getattr() below obtains its underlying function.
        self.assertEqual(getattr(m, 'im_func', m), t.__dict__[meth])
        dictionary['a'] = deepcopy(a)
        m(dictionary['a'], slice(b, c), d)
        self.assertEqual(dictionary['a'], res)
        dictionary['a'] = deepcopy(a)
        bm = getattr(dictionary['a'], meth)
        bm(slice(b, c), d)
        self.assertEqual(dictionary['a'], res)

    def test_lists(self):
        # [FUNCTION DOC] Test: checks lists behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing list operations...
        # Asserts are within individual test methods
        self.binop_test([1], [2], [1,2], "a+b", "__add__")
        self.binop_test([1,2,3], 2, 1, "b in a", "__contains__")
        self.binop_test([1,2,3], 4, 0, "b in a", "__contains__")
        self.binop_test([1,2,3], 1, 2, "a[b]", "__getitem__")
        self.sliceop_test([1,2,3], 0, 2, [1,2], "a[b:c]", "__getitem__")
        self.setop_test([1], [2], [1,2], "a+=b", "__iadd__")
        self.setop_test([1,2], 3, [1,2,1,2,1,2], "a*=b", "__imul__")
        self.unop_test([1,2,3], 3, "len(a)", "__len__")
        self.binop_test([1,2], 3, [1,2,1,2,1,2], "a*b", "__mul__")
        self.binop_test([1,2], 3, [1,2,1,2,1,2], "b*a", "__rmul__")
        self.set2op_test([1,2], 1, 3, [1,3], "a[b]=c", "__setitem__")
        self.setsliceop_test([1,2,3,4], 1, 3, [5,6], [1,5,6,4], "a[b:c]=d",
                        "__setitem__")

    def test_dicts(self):
        # [FUNCTION DOC] Test: checks dicts behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing dict operations...
        self.binop_test({1:2,3:4}, 1, 1, "b in a", "__contains__")
        self.binop_test({1:2,3:4}, 2, 0, "b in a", "__contains__")
        self.binop_test({1:2,3:4}, 1, 2, "a[b]", "__getitem__")

        d = {1:2, 3:4}
        l1 = []
        for i in list(d.keys()):
            l1.append(i)
        l = []
        for i in iter(d):
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in d.__iter__():
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in dict.__iter__(d):
            l.append(i)
        self.assertEqual(l, l1)
        d = {1:2, 3:4}
        self.unop_test(d, 2, "len(a)", "__len__")
        self.assertEqual(eval(repr(d), {}), d)
        self.assertEqual(eval(d.__repr__(), {}), d)
        self.set2op_test({1:2,3:4}, 2, 3, {1:2,2:3,3:4}, "a[b]=c",
                        "__setitem__")

    # Tests for unary and binary operators
    def number_operators(self, a, b, skip=[]):
        # [FUNCTION DOC] HELPER: number_operators of OperatorsTest. Arguments: a, b, skip.
        # Supporting routine for tests; returns a computed value or None.
        dict = {'a': a, 'b': b}

        for name, expr in self.binops.items():
            if name not in skip:
                name = "__%s__" % name
                if hasattr(a, name):
                    res = eval(expr, dict)
                    self.binop_test(a, b, res, expr, name)

        for name, expr in list(self.unops.items()):
            if name not in skip:
                name = "__%s__" % name
                if hasattr(a, name):
                    res = eval(expr, dict)
                    self.unop_test(a, res, expr, name)

    def test_ints(self):
        # [FUNCTION DOC] Test: checks ints behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing int operations...
        self.number_operators(100, 3)
        # The following crashes in Python 2.2
        self.assertEqual((1).__bool__(), 1)
        self.assertEqual((0).__bool__(), 0)
        # This returns 'NotImplemented' in Python 2.2
        class C(int):
            # [CLASS DOC] HELPER: local class C defined inside test_ints to exercise
            # descriptor/class-model behavior under test.
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: special method __add__ of C. Arguments: other. Implements
                # the protocol hook used by tests; returns the hook result.
                return NotImplemented
        self.assertEqual(C(5), 5)
        try:
            C() + ""
        except TypeError:
            pass
        else:
            self.fail("NotImplemented should have caused TypeError")

    def test_floats(self):
        # [FUNCTION DOC] Test: checks floats behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing float operations...
        self.number_operators(100.0, 3.0)

    def test_complexes(self):
        # [FUNCTION DOC] Test: checks complexes behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing complex operations...
        self.number_operators(100.0j, 3.0j, skip=['lt', 'le', 'gt', 'ge',
                                                  'int', 'float',
                                                  'floordiv', 'divmod', 'mod'])

        class Number(complex):
            # [CLASS DOC] HELPER: local class Number defined inside test_complexes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['prec']
            def __new__(cls, *args, **kwds):
                # [FUNCTION DOC] HELPER: special method __new__ of Number. Arguments: *args,
                # **kwds. Implements the protocol hook used by tests; returns the hook result.
                result = complex.__new__(cls, *args)
                result.prec = kwds.get('prec', 12)
                return result
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of Number. Implements the
                # protocol hook used by tests; returns the hook result.
                prec = self.prec
                if self.imag == 0.0:
                    return "%.*g" % (prec, self.real)
                if self.real == 0.0:
                    return "%.*gj" % (prec, self.imag)
                return "(%.*g+%.*gj)" % (prec, self.real, prec, self.imag)
            __str__ = __repr__

        a = Number(3.14, prec=6)
        self.assertEqual(repr(a), "3.14")
        self.assertEqual(a.prec, 6)

        a = Number(a, prec=2)
        self.assertEqual(repr(a), "3.1")
        self.assertEqual(a.prec, 2)

        a = Number(234.5)
        self.assertEqual(repr(a), "234.5")
        self.assertEqual(a.prec, 12)

    def test_explicit_reverse_methods(self):
        # [FUNCTION DOC] Test: checks explicit reverse methods behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # see issue 9930
        self.assertEqual(complex.__radd__(3j, 4.0), complex(4.0, 3.0))
        self.assertEqual(float.__rsub__(3.0, 1), -2.0)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_spam_lists(self):
        # [FUNCTION DOC] Test: checks spam lists behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing spamlist operations...
        import copy, xxsubtype as spam

        def spamlist(l, memo=None):
            # [FUNCTION DOC] HELPER: local function spamlist inside test_spam_lists. Arguments: l,
            # memo. Returns its computed value or None; used by the enclosing test.
            import xxsubtype as spam
            return spam.spamlist(l)

        # This is an ugly hack:
        copy._deepcopy_dispatch[spam.spamlist] = spamlist

        self.binop_test(spamlist([1]), spamlist([2]), spamlist([1,2]), "a+b",
                       "__add__")
        self.binop_test(spamlist([1,2,3]), 2, 1, "b in a", "__contains__")
        self.binop_test(spamlist([1,2,3]), 4, 0, "b in a", "__contains__")
        self.binop_test(spamlist([1,2,3]), 1, 2, "a[b]", "__getitem__")
        self.sliceop_test(spamlist([1,2,3]), 0, 2, spamlist([1,2]), "a[b:c]",
                          "__getitem__")
        self.setop_test(spamlist([1]), spamlist([2]), spamlist([1,2]), "a+=b",
                        "__iadd__")
        self.setop_test(spamlist([1,2]), 3, spamlist([1,2,1,2,1,2]), "a*=b",
                        "__imul__")
        self.unop_test(spamlist([1,2,3]), 3, "len(a)", "__len__")
        self.binop_test(spamlist([1,2]), 3, spamlist([1,2,1,2,1,2]), "a*b",
                        "__mul__")
        self.binop_test(spamlist([1,2]), 3, spamlist([1,2,1,2,1,2]), "b*a",
                        "__rmul__")
        self.set2op_test(spamlist([1,2]), 1, 3, spamlist([1,3]), "a[b]=c",
                         "__setitem__")
        self.setsliceop_test(spamlist([1,2,3,4]), 1, 3, spamlist([5,6]),
                             spamlist([1,5,6,4]), "a[b:c]=d", "__setitem__")
        # Test subclassing
        class C(spam.spamlist):
            # [FUNCTION DOC] HELPER: foo of C. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_spam_lists to exercise
            # descriptor/class-model behavior under test.
            def foo(self): return 1
        a = C()
        self.assertEqual(a, [])
        self.assertEqual(a.foo(), 1)
        a.append(100)
        self.assertEqual(a, [100])
        self.assertEqual(a.getstate(), 0)
        a.setstate(42)
        self.assertEqual(a.getstate(), 42)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_spam_dicts(self):
        # [FUNCTION DOC] Test: checks spam dicts behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing spamdict operations...
        import copy, xxsubtype as spam
        def spamdict(d, memo=None):
            # [FUNCTION DOC] HELPER: local function spamdict inside test_spam_dicts. Arguments: d,
            # memo. Returns its computed value or None; used by the enclosing test.
            import xxsubtype as spam
            sd = spam.spamdict()
            for k, v in list(d.items()):
                sd[k] = v
            return sd
        # This is an ugly hack:
        copy._deepcopy_dispatch[spam.spamdict] = spamdict

        self.binop_test(spamdict({1:2,3:4}), 1, 1, "b in a", "__contains__")
        self.binop_test(spamdict({1:2,3:4}), 2, 0, "b in a", "__contains__")
        self.binop_test(spamdict({1:2,3:4}), 1, 2, "a[b]", "__getitem__")
        d = spamdict({1:2,3:4})
        l1 = []
        for i in list(d.keys()):
            l1.append(i)
        l = []
        for i in iter(d):
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in d.__iter__():
            l.append(i)
        self.assertEqual(l, l1)
        l = []
        for i in type(spamdict({})).__iter__(d):
            l.append(i)
        self.assertEqual(l, l1)
        straightd = {1:2, 3:4}
        spamd = spamdict(straightd)
        self.unop_test(spamd, 2, "len(a)", "__len__")
        self.unop_test(spamd, repr(straightd), "repr(a)", "__repr__")
        self.set2op_test(spamdict({1:2,3:4}), 2, 3, spamdict({1:2,2:3,3:4}),
                   "a[b]=c", "__setitem__")
        # Test subclassing
        class C(spam.spamdict):
            # [FUNCTION DOC] HELPER: foo of C. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_spam_dicts to exercise
            # descriptor/class-model behavior under test.
            def foo(self): return 1
        a = C()
        self.assertEqual(list(a.items()), [])
        self.assertEqual(a.foo(), 1)
        a['foo'] = 'bar'
        self.assertEqual(list(a.items()), [('foo', 'bar')])
        self.assertEqual(a.getstate(), 0)
        a.setstate(100)
        self.assertEqual(a.getstate(), 100)

    def test_wrap_lenfunc_bad_cast(self):
        # [FUNCTION DOC] Test: checks wrap lenfunc bad cast behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        self.assertEqual(range(sys.maxsize).__len__(), sys.maxsize)


class ClassPropertiesAndMethods(CPythonTestCase, ExtraAssertions):
    # [CLASS DOC] Test case for the Python class model: metaclasses, slots, properties,
    # descriptors, MRO, super, attribute access, special methods and related edge cases; each test
    # method asserts via unittest assertions.

    def test_python_dicts(self):
        # [FUNCTION DOC] Test: checks python dicts behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing Python subclass of dict...
        self.assertTrue(issubclass(dict, dict))
        self.assertIsInstance({}, dict)
        d = dict()
        self.assertEqual(d, {})
        self.assertIs(d.__class__, dict)
        self.assertIsInstance(d, dict)
        class C(dict):
            # [CLASS DOC] HELPER: local class C defined inside test_python_dicts to exercise
            # descriptor/class-model behavior under test.
            state = -1
            def __init__(self_local, *a, **kw):
                # [FUNCTION DOC] HELPER: constructor. Arguments: self_local, *a, **kw. Stores the
                # given values on the instance, and for test cases initializes the base
                # CPythonTestCase.
                if a:
                    self.assertEqual(len(a), 1)
                    self_local.state = a[0]
                if kw:
                    for k, v in list(kw.items()):
                        self_local[v] = k
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: special method __getitem__ of C. Arguments: key.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.get(key, 0)
            def __setitem__(self_local, key, value):
                # [FUNCTION DOC] HELPER: special method __setitem__ of C. Arguments: self_local,
                # key, value. Implements the protocol hook used by tests; returns the hook result.
                self.assertIsInstance(key, int)
                dict.__setitem__(self_local, key, value)
            def setstate(self, state):
                # [FUNCTION DOC] HELPER: setstate of C. Arguments: state. Supporting routine for
                # tests; returns a computed value or None.
                self.state = state
            def getstate(self):
                # [FUNCTION DOC] HELPER: getstate of C. Supporting routine for tests; returns a
                # computed value or None.
                return self.state
        self.assertTrue(issubclass(C, dict))
        a1 = C(12)
        self.assertEqual(a1.state, 12)
        a2 = C(foo=1, bar=2)
        self.assertEqual(a2[1] == 'foo' and a2[2], 'bar')
        a = C()
        self.assertEqual(a.state, -1)
        self.assertEqual(a.getstate(), -1)
        a.setstate(0)
        self.assertEqual(a.state, 0)
        self.assertEqual(a.getstate(), 0)
        a.setstate(10)
        self.assertEqual(a.state, 10)
        self.assertEqual(a.getstate(), 10)
        self.assertEqual(a[42], 0)
        a[42] = 24
        self.assertEqual(a[42], 24)
        N = 50
        for i in range(N):
            a[i] = C()
            for j in range(N):
                a[i][j] = i*j
        for i in range(N):
            for j in range(N):
                self.assertEqual(a[i][j], i*j)

    def test_python_lists(self):
        # [FUNCTION DOC] Test: checks python lists behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing Python subclass of list...
        class C(list):
            # [CLASS DOC] HELPER: local class C defined inside test_python_lists to exercise
            # descriptor/class-model behavior under test.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: special method __getitem__ of C. Arguments: i. Implements
                # the protocol hook used by tests; returns the hook result.
                if isinstance(i, slice):
                    return i.start, i.stop
                return list.__getitem__(self, i) + 100
        a = C()
        a.extend([0,1,2])
        self.assertEqual(a[0], 100)
        self.assertEqual(a[1], 101)
        self.assertEqual(a[2], 102)
        self.assertEqual(a[100:200], (100,200))

    def test_metaclass(self):
        # [FUNCTION DOC] Test: checks metaclass behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing metaclasses...
        class C(metaclass=type):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.__state = 0
            def getstate(self):
                # [FUNCTION DOC] HELPER: getstate of C. Supporting routine for tests; returns a
                # computed value or None.
                return self.__state
            def setstate(self, state):
                # [FUNCTION DOC] HELPER: setstate of C. Arguments: state. Supporting routine for
                # tests; returns a computed value or None.
                self.__state = state
        a = C()
        self.assertEqual(a.getstate(), 0)
        a.setstate(10)
        self.assertEqual(a.getstate(), 10)
        class _metaclass(type):
            # [FUNCTION DOC] HELPER: myself of _metaclass. Supporting routine for tests; returns a
            # computed value or None.
            # [CLASS DOC] HELPER: local class _metaclass defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def myself(cls): return cls
        class D(metaclass=_metaclass):
            # [CLASS DOC] HELPER: local class D defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(D.myself(), D)
        d = D()
        self.assertEqual(d.__class__, D)
        class M1(type):
            # [CLASS DOC] HELPER: local class M1 defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls, name, bases, dict):
                # [FUNCTION DOC] HELPER: special method __new__ of M1. Arguments: name, bases,
                # dict. Implements the protocol hook used by tests; returns the hook result.
                dict['__spam__'] = 1
                return type.__new__(cls, name, bases, dict)
        class C(metaclass=M1):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(C.__spam__, 1)
        c = C()
        self.assertEqual(c.__spam__, 1)

        class _instance(object):
            # [CLASS DOC] HELPER: local class _instance defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        class M2(object):
            @staticmethod
            # [CLASS DOC] HELPER: local class M2 defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls, name, bases, dict):
                # [FUNCTION DOC] HELPER: special method __new__ of M2. Arguments: name, bases,
                # dict. Implements the protocol hook used by tests; returns the hook result.
                self = object.__new__(cls)
                self.name = name
                self.bases = bases
                self.dict = dict
                return self
            def __call__(self):
                # [FUNCTION DOC] HELPER: special method __call__ of M2. Implements the protocol
                # hook used by tests; returns the hook result.
                it = _instance()
                # Early binding of methods
                for key in self.dict:
                    if key.startswith("__"):
                        continue
                    setattr(it, key, self.dict[key].__get__(it, self))
                return it
        class C(metaclass=M2):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def spam(self):
                # [FUNCTION DOC] HELPER: spam of C. Supporting routine for tests; returns a
                # computed value or None.
                return 42
        self.assertEqual(C.name, 'C')
        self.assertEqual(C.bases, ())
        self.assertIn('spam', C.dict)
        c = C()
        self.assertEqual(c.spam(), 42)

        # More metaclass examples

        class autosuper(type):
            # [CLASS DOC] HELPER: local class autosuper defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            # Automatically add __super to the class
            # This trick only works for dynamic classes
            def __new__(metaclass, name, bases, dict):
                # [FUNCTION DOC] HELPER: special method __new__ of autosuper. Arguments:
                # metaclass, name, bases, dict. Implements the protocol hook used by tests;
                # returns the hook result.
                cls = super(autosuper, metaclass).__new__(metaclass,
                                                        name, bases, dict)
                # Name mangling for __super removes leading underscores
                while name[:1] == "_":
                    name = name[1:]
                if name:
                    name = "_%s__super" % name
                else:
                    name = "__super"
                setattr(cls, name, super(cls))
                return cls
        class A(metaclass=autosuper):
            # [CLASS DOC] HELPER: local class A defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of A. Supporting routine for tests; returns a
                # computed value or None.
                return "A"
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of B. Supporting routine for tests; returns a
                # computed value or None.
                return "B" + self.__super.meth()
        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of C. Supporting routine for tests; returns a
                # computed value or None.
                return "C" + self.__super.meth()
        class D(C, B):
            # [CLASS DOC] HELPER: local class D defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of D. Supporting routine for tests; returns a
                # computed value or None.
                return "D" + self.__super.meth()
        self.assertEqual(D().meth(), "DCBA")
        class E(B, C):
            # [CLASS DOC] HELPER: local class E defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of E. Supporting routine for tests; returns a
                # computed value or None.
                return "E" + self.__super.meth()
        self.assertEqual(E().meth(), "EBCA")

        class autoproperty(type):
            # [CLASS DOC] HELPER: local class autoproperty defined inside test_metaclass to
            # exercise descriptor/class-model behavior under test.
            # Automatically create property attributes when methods
            # named _get_x and/or _set_x are found
            def __new__(metaclass, name, bases, dict):
                # [FUNCTION DOC] HELPER: special method __new__ of autoproperty. Arguments:
                # metaclass, name, bases, dict. Implements the protocol hook used by tests;
                # returns the hook result.
                hits = {}
                for key, val in dict.items():
                    if key.startswith("_get_"):
                        key = key[5:]
                        get, set = hits.get(key, (None, None))
                        get = val
                        hits[key] = get, set
                    elif key.startswith("_set_"):
                        key = key[5:]
                        get, set = hits.get(key, (None, None))
                        set = val
                        hits[key] = get, set
                for key, (get, set) in hits.items():
                    dict[key] = property(get, set)
                return super(autoproperty, metaclass).__new__(metaclass,
                                                            name, bases, dict)
        class A(metaclass=autoproperty):
            # [CLASS DOC] HELPER: local class A defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def _get_x(self):
                # [FUNCTION DOC] HELPER: _get_x of A. Supporting routine for tests; returns a
                # computed value or None.
                return -self.__x
            def _set_x(self, x):
                # [FUNCTION DOC] HELPER: _set_x of A. Arguments: x. Supporting routine for tests;
                # returns a computed value or None.
                self.__x = -x
        a = A()
        self.assertNotHasAttr(a, "x")
        a.x = 12
        self.assertEqual(a.x, 12)
        self.assertEqual(a._A__x, -12)

        class multimetaclass(autoproperty, autosuper):
            # [CLASS DOC] HELPER: local class multimetaclass defined inside test_metaclass to
            # exercise descriptor/class-model behavior under test.
            # Merge of multiple cooperating metaclasses
            pass
        class A(metaclass=multimetaclass):
            # [CLASS DOC] HELPER: local class A defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def _get_x(self):
                # [FUNCTION DOC] HELPER: _get_x of A. Supporting routine for tests; returns a
                # computed value or None.
                return "A"
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def _get_x(self):
                # [FUNCTION DOC] HELPER: _get_x of B. Supporting routine for tests; returns a
                # computed value or None.
                return "B" + self.__super._get_x()
        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def _get_x(self):
                # [FUNCTION DOC] HELPER: _get_x of C. Supporting routine for tests; returns a
                # computed value or None.
                return "C" + self.__super._get_x()
        class D(C, B):
            # [CLASS DOC] HELPER: local class D defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def _get_x(self):
                # [FUNCTION DOC] HELPER: _get_x of D. Supporting routine for tests; returns a
                # computed value or None.
                return "D" + self.__super._get_x()
        self.assertEqual(D().x, "DCBA")

        # Make sure type(x) doesn't call x.__class__.__init__
        class T(type):
            # [CLASS DOC] HELPER: local class T defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            counter = 0
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: constructor. Arguments: *args. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                T.counter += 1
        class C(metaclass=T):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(T.counter, 1)
        a = C()
        self.assertEqual(type(a), C)
        self.assertEqual(T.counter, 1)

        # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
        # descriptor/class-model behavior under test.
        class C(object): pass
        c = C()
        try: c()
        except TypeError: pass
        else: self.fail("calling object w/o call method should raise "
                        "TypeError")

        # Testing code to find most derived baseclass
        class A(type):
            # [CLASS DOC] HELPER: local class A defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(*args, **kwargs):
                # [FUNCTION DOC] HELPER: special method __new__ of A. Arguments: *args, **kwargs.
                # Implements the protocol hook used by tests; returns the hook result.
                return type.__new__(*args, **kwargs)

        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass

        class C(object, metaclass=A):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass

        # The most derived metaclass of D is A rather than type.
        class D(B, C):
            # [CLASS DOC] HELPER: local class D defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(A, type(D))

        # issue1294232: correct metaclass calculation
        new_calls = []  # to check the order of __new__ calls
        class AMeta(type):
            @staticmethod
            # [CLASS DOC] HELPER: local class AMeta defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(mcls, name, bases, ns):
                # [FUNCTION DOC] HELPER: special method __new__ of AMeta. Arguments: mcls, name,
                # bases, ns. Implements the protocol hook used by tests; returns the hook result.
                new_calls.append('AMeta')
                return super().__new__(mcls, name, bases, ns)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: special method __prepare__ of AMeta. Arguments: mcls,
                # name, bases. Implements the protocol hook used by tests; returns the hook
                # result.
                return {}

        class BMeta(AMeta):
            @staticmethod
            # [CLASS DOC] HELPER: local class BMeta defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(mcls, name, bases, ns):
                # [FUNCTION DOC] HELPER: special method __new__ of BMeta. Arguments: mcls, name,
                # bases, ns. Implements the protocol hook used by tests; returns the hook result.
                new_calls.append('BMeta')
                return super().__new__(mcls, name, bases, ns)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: special method __prepare__ of BMeta. Arguments: mcls,
                # name, bases. Implements the protocol hook used by tests; returns the hook
                # result.
                ns = super().__prepare__(name, bases)
                ns['BMeta_was_here'] = True
                return ns

        class A(metaclass=AMeta):
            # [CLASS DOC] HELPER: local class A defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(['AMeta'], new_calls)
        new_calls.clear()

        class B(metaclass=BMeta):
            # [CLASS DOC] HELPER: local class B defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        # BMeta.__new__ calls AMeta.__new__ with super:
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()

        class C(A, B):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        # The most derived metaclass is BMeta:
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        # BMeta.__prepare__ should've been called:
        self.assertIn('BMeta_was_here', C.__dict__)

        # The order of the bases shouldn't matter:
        class C2(B, A):
            # [CLASS DOC] HELPER: local class C2 defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        self.assertIn('BMeta_was_here', C2.__dict__)

        # Check correct metaclass calculation when a metaclass is declared:
        class D(C, metaclass=type):
            # [CLASS DOC] HELPER: local class D defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        self.assertIn('BMeta_was_here', D.__dict__)

        class E(C, metaclass=AMeta):
            # [CLASS DOC] HELPER: local class E defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(['BMeta', 'AMeta'], new_calls)
        new_calls.clear()
        self.assertIn('BMeta_was_here', E.__dict__)

        # Special case: the given metaclass isn't a class,
        # so there is no metaclass calculation.
        marker = object()
        def func(*args, **kwargs):
            # [FUNCTION DOC] HELPER: local function func inside test_metaclass. Arguments: *args,
            # **kwargs. Returns its computed value or None; used by the enclosing test.
            return marker
        class X(metaclass=func):
            # [CLASS DOC] HELPER: local class X defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        class Y(object, metaclass=func):
            # [CLASS DOC] HELPER: local class Y defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        class Z(D, metaclass=func):
            # [CLASS DOC] HELPER: local class Z defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(marker, X)
        self.assertIs(marker, Y)
        self.assertIs(marker, Z)

        # The given metaclass is a class,
        # but not a descendant of type.
        prepare_calls = []  # to track __prepare__ calls
        class ANotMeta:
            # [CLASS DOC] HELPER: local class ANotMeta defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(mcls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: special method __new__ of ANotMeta. Arguments: mcls,
                # *args, **kwargs. Implements the protocol hook used by tests; returns the hook
                # result.
                new_calls.append('ANotMeta')
                return super().__new__(mcls)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: special method __prepare__ of ANotMeta. Arguments: mcls,
                # name, bases. Implements the protocol hook used by tests; returns the hook
                # result.
                prepare_calls.append('ANotMeta')
                return {}
        class BNotMeta(ANotMeta):
            # [CLASS DOC] HELPER: local class BNotMeta defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            def __new__(mcls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: special method __new__ of BNotMeta. Arguments: mcls,
                # *args, **kwargs. Implements the protocol hook used by tests; returns the hook
                # result.
                new_calls.append('BNotMeta')
                return super().__new__(mcls)
            @classmethod
            def __prepare__(mcls, name, bases):
                # [FUNCTION DOC] HELPER: special method __prepare__ of BNotMeta. Arguments: mcls,
                # name, bases. Implements the protocol hook used by tests; returns the hook
                # result.
                prepare_calls.append('BNotMeta')
                return super().__prepare__(name, bases)

        class A(metaclass=ANotMeta):
            # [CLASS DOC] HELPER: local class A defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(ANotMeta, type(A))
        self.assertEqual(['ANotMeta'], prepare_calls)
        prepare_calls.clear()
        self.assertEqual(['ANotMeta'], new_calls)
        new_calls.clear()

        class B(metaclass=BNotMeta):
            # [CLASS DOC] HELPER: local class B defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(BNotMeta, type(B))
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()

        class C(A, B):
            # [CLASS DOC] HELPER: local class C defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(BNotMeta, type(C))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        class C2(B, A):
            # [CLASS DOC] HELPER: local class C2 defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(BNotMeta, type(C2))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        # This is a TypeError, because of a metaclass conflict:
        # BNotMeta is neither a subclass, nor a superclass of type
        with self.assertRaises(TypeError):
            class D(C, metaclass=type):
                # [CLASS DOC] HELPER: local class D defined inside test_metaclass to exercise
                # descriptor/class-model behavior under test.
                pass

        class E(C, metaclass=ANotMeta):
            # [CLASS DOC] HELPER: local class E defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(BNotMeta, type(E))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        class F(object(), C):
            # [CLASS DOC] HELPER: local class F defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(BNotMeta, type(F))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        class F2(C, object()):
            # [CLASS DOC] HELPER: local class F2 defined inside test_metaclass to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertIs(BNotMeta, type(F2))
        self.assertEqual(['BNotMeta', 'ANotMeta'], new_calls)
        new_calls.clear()
        self.assertEqual(['BNotMeta', 'ANotMeta'], prepare_calls)
        prepare_calls.clear()

        # TypeError: BNotMeta is neither a
        # subclass, nor a superclass of int
        with self.assertRaises(TypeError):
            class X(C, int()):
                # [CLASS DOC] HELPER: local class X defined inside test_metaclass to exercise
                # descriptor/class-model behavior under test.
                pass
        with self.assertRaises(TypeError):
            class X(int(), C):
                # [CLASS DOC] HELPER: local class X defined inside test_metaclass to exercise
                # descriptor/class-model behavior under test.
                pass

    def test_module_subclasses(self):
        # [FUNCTION DOC] Test: checks module subclasses behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing Python subclass of module...
        log = []
        MT = type(sys)
        class MM(MT):
            # [CLASS DOC] HELPER: local class MM defined inside test_module_subclasses to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, name):
                # [FUNCTION DOC] HELPER: constructor. Arguments: name. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                MT.__init__(self, name)
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattribute__ of MM. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                log.append(("getattr", name))
                return MT.__getattribute__(self, name)
            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: special method __setattr__ of MM. Arguments: name, value.
                # Implements the protocol hook used by tests; returns the hook result.
                log.append(("setattr", name, value))
                MT.__setattr__(self, name, value)
            def __delattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __delattr__ of MM. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                log.append(("delattr", name))
                MT.__delattr__(self, name)
        a = MM("a")
        a.foo = 12
        x = a.foo
        del a.foo
        self.assertEqual(log, [("setattr", "foo", 12),
                               ("getattr", "foo"),
                               ("delattr", "foo")])

        # https://bugs.python.org/issue1174712
        try:
            class Module(types.ModuleType, str):
                # [CLASS DOC] HELPER: local class Module defined inside test_module_subclasses to
                # exercise descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("inheriting from ModuleType and str at the same time "
                    "should fail")

        # Issue 34805: Verify that definition order is retained
        def random_name():
            # [FUNCTION DOC] HELPER: local function random_name inside test_module_subclasses.
            # Returns its computed value or None; used by the enclosing test.
            return ''.join(random.choices(string.ascii_letters, k=10))
        class A:
            # [CLASS DOC] HELPER: local class A defined inside test_module_subclasses to exercise
            # descriptor/class-model behavior under test.
            pass
        subclasses = [type(random_name(), (A,), {}) for i in range(100)]
        self.assertEqual(A.__subclasses__(), subclasses)

    def test_multiple_inheritance(self):
        # [FUNCTION DOC] Test: checks multiple inheritance behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing multiple inheritance...
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.__state = 0
            def getstate(self):
                # [FUNCTION DOC] HELPER: getstate of C. Supporting routine for tests; returns a
                # computed value or None.
                return self.__state
            def setstate(self, state):
                # [FUNCTION DOC] HELPER: setstate of C. Arguments: state. Supporting routine for
                # tests; returns a computed value or None.
                self.__state = state
        a = C()
        self.assertEqual(a.getstate(), 0)
        a.setstate(10)
        self.assertEqual(a.getstate(), 10)
        class D(dict, C):
            # [CLASS DOC] HELPER: local class D defined inside test_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                dict.__init__(self)
                C.__init__(self)
        d = D()
        self.assertEqual(list(d.keys()), [])
        d["hello"] = "world"
        self.assertEqual(list(d.items()), [("hello", "world")])
        self.assertEqual(d["hello"], "world")
        self.assertEqual(d.getstate(), 0)
        d.setstate(10)
        self.assertEqual(d.getstate(), 10)
        self.assertEqual(D.__mro__, (D, dict, C, object))

        # SF bug #442833
        class Node(object):
            # [CLASS DOC] HELPER: local class Node defined inside test_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            def __int__(self):
                # [FUNCTION DOC] HELPER: special method __int__ of Node. Implements the protocol
                # hook used by tests; returns the hook result.
                return int(self.foo())
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of Node. Supporting routine for tests; returns a
                # computed value or None.
                return "23"
        class Frag(Node, list):
            # [CLASS DOC] HELPER: local class Frag defined inside test_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of Frag. Supporting routine for tests; returns a
                # computed value or None.
                return "42"
        self.assertEqual(Node().__int__(), 23)
        self.assertEqual(int(Node()), 23)
        self.assertEqual(Frag().__int__(), 42)
        self.assertEqual(int(Frag()), 42)

    def test_diamond_inheritance(self):
        # [FUNCTION DOC] Test: checks diamond inheritance behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing multiple inheritance special cases...
        class A(object):
            # [FUNCTION DOC] HELPER: spam of A. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class A defined inside test_diamond_inheritance to
            # exercise descriptor/class-model behavior under test.
            def spam(self): return "A"
        self.assertEqual(A().spam(), "A")
        class B(A):
            # [FUNCTION DOC] HELPER: boo of B. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class B defined inside test_diamond_inheritance to
            # exercise descriptor/class-model behavior under test.
            def boo(self): return "B"
            # [FUNCTION DOC] HELPER: spam of B. Supporting routine for tests; returns a computed
            # value or None.
            def spam(self): return "B"
        self.assertEqual(B().spam(), "B")
        self.assertEqual(B().boo(), "B")
        class C(A):
            # [FUNCTION DOC] HELPER: boo of C. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_diamond_inheritance to
            # exercise descriptor/class-model behavior under test.
            def boo(self): return "C"
        self.assertEqual(C().spam(), "A")
        self.assertEqual(C().boo(), "C")
        # [CLASS DOC] HELPER: local class D defined inside test_diamond_inheritance to exercise
        # descriptor/class-model behavior under test.
        class D(B, C): pass
        self.assertEqual(D().spam(), "B")
        self.assertEqual(D().boo(), "B")
        self.assertEqual(D.__mro__, (D, B, C, A, object))
        # [CLASS DOC] HELPER: local class E defined inside test_diamond_inheritance to exercise
        # descriptor/class-model behavior under test.
        class E(C, B): pass
        self.assertEqual(E().spam(), "B")
        self.assertEqual(E().boo(), "C")
        self.assertEqual(E.__mro__, (E, C, B, A, object))
        # MRO order disagreement
        try:
            # [CLASS DOC] HELPER: local class F defined inside test_diamond_inheritance to
            # exercise descriptor/class-model behavior under test.
            class F(D, E): pass
        except TypeError:
            pass
        else:
            self.fail("expected MRO order disagreement (F)")
        try:
            # [CLASS DOC] HELPER: local class G defined inside test_diamond_inheritance to
            # exercise descriptor/class-model behavior under test.
            class G(E, D): pass
        except TypeError:
            pass
        else:
            self.fail("expected MRO order disagreement (G)")

    # see thread python-dev/2002-October/029035.html
    def test_ex5_from_c3_switch(self):
        # [FUNCTION DOC] Test: checks ex5 from c3 switch behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing ex5 from C3 switch discussion...
        # [CLASS DOC] HELPER: local class A defined inside test_ex5_from_c3_switch to exercise
        # descriptor/class-model behavior under test.
        class A(object): pass
        # [CLASS DOC] HELPER: local class B defined inside test_ex5_from_c3_switch to exercise
        # descriptor/class-model behavior under test.
        class B(object): pass
        # [CLASS DOC] HELPER: local class C defined inside test_ex5_from_c3_switch to exercise
        # descriptor/class-model behavior under test.
        class C(object): pass
        # [CLASS DOC] HELPER: local class X defined inside test_ex5_from_c3_switch to exercise
        # descriptor/class-model behavior under test.
        class X(A): pass
        # [CLASS DOC] HELPER: local class Y defined inside test_ex5_from_c3_switch to exercise
        # descriptor/class-model behavior under test.
        class Y(A): pass
        # [CLASS DOC] HELPER: local class Z defined inside test_ex5_from_c3_switch to exercise
        # descriptor/class-model behavior under test.
        class Z(X,B,Y,C): pass
        self.assertEqual(Z.__mro__, (Z, X, B, Y, A, C, object))

    # see "A Monotonic Superclass Linearization for Dylan",
    # by Kim Barrett et al. (OOPSLA 1996)
    def test_monotonicity(self):
        # [FUNCTION DOC] Test: checks monotonicity behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing MRO monotonicity...
        # [CLASS DOC] HELPER: local class Boat defined inside test_monotonicity to exercise
        # descriptor/class-model behavior under test.
        class Boat(object): pass
        # [CLASS DOC] HELPER: local class DayBoat defined inside test_monotonicity to exercise
        # descriptor/class-model behavior under test.
        class DayBoat(Boat): pass
        # [CLASS DOC] HELPER: local class WheelBoat defined inside test_monotonicity to exercise
        # descriptor/class-model behavior under test.
        class WheelBoat(Boat): pass
        # [CLASS DOC] HELPER: local class EngineLess defined inside test_monotonicity to exercise
        # descriptor/class-model behavior under test.
        class EngineLess(DayBoat): pass
        # [CLASS DOC] HELPER: local class SmallMultihull defined inside test_monotonicity to
        # exercise descriptor/class-model behavior under test.
        class SmallMultihull(DayBoat): pass
        # [CLASS DOC] HELPER: local class PedalWheelBoat defined inside test_monotonicity to
        # exercise descriptor/class-model behavior under test.
        class PedalWheelBoat(EngineLess,WheelBoat): pass
        # [CLASS DOC] HELPER: local class SmallCatamaran defined inside test_monotonicity to
        # exercise descriptor/class-model behavior under test.
        class SmallCatamaran(SmallMultihull): pass
        # [CLASS DOC] HELPER: local class Pedalo defined inside test_monotonicity to exercise
        # descriptor/class-model behavior under test.
        class Pedalo(PedalWheelBoat,SmallCatamaran): pass

        self.assertEqual(PedalWheelBoat.__mro__,
              (PedalWheelBoat, EngineLess, DayBoat, WheelBoat, Boat, object))
        self.assertEqual(SmallCatamaran.__mro__,
              (SmallCatamaran, SmallMultihull, DayBoat, Boat, object))
        self.assertEqual(Pedalo.__mro__,
              (Pedalo, PedalWheelBoat, EngineLess, SmallCatamaran,
               SmallMultihull, DayBoat, WheelBoat, Boat, object))

    # see "A Monotonic Superclass Linearization for Dylan",
    # by Kim Barrett et al. (OOPSLA 1996)
    def test_consistency_with_epg(self):
        # [FUNCTION DOC] Test: checks consistency with epg behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing consistency with EPG...
        # [CLASS DOC] HELPER: local class Pane defined inside test_consistency_with_epg to
        # exercise descriptor/class-model behavior under test.
        class Pane(object): pass
        # [CLASS DOC] HELPER: local class ScrollingMixin defined inside test_consistency_with_epg
        # to exercise descriptor/class-model behavior under test.
        class ScrollingMixin(object): pass
        # [CLASS DOC] HELPER: local class EditingMixin defined inside test_consistency_with_epg to
        # exercise descriptor/class-model behavior under test.
        class EditingMixin(object): pass
        # [CLASS DOC] HELPER: local class ScrollablePane defined inside test_consistency_with_epg
        # to exercise descriptor/class-model behavior under test.
        class ScrollablePane(Pane,ScrollingMixin): pass
        # [CLASS DOC] HELPER: local class EditablePane defined inside test_consistency_with_epg to
        # exercise descriptor/class-model behavior under test.
        class EditablePane(Pane,EditingMixin): pass
        # [CLASS DOC] HELPER: local class EditableScrollablePane defined inside
        # test_consistency_with_epg to exercise descriptor/class-model behavior under test.
        class EditableScrollablePane(ScrollablePane,EditablePane): pass

        self.assertEqual(EditableScrollablePane.__mro__,
              (EditableScrollablePane, ScrollablePane, EditablePane, Pane,
                ScrollingMixin, EditingMixin, object))

    def test_mro_disagreement(self):
        # [FUNCTION DOC] Test: checks mro disagreement behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing error messages for MRO disagreement...
        mro_err_msg = ("Cannot create a consistent method resolution "
                       "order (MRO) for bases ")

        def raises(exc, expected, callable, *args):
            # [FUNCTION DOC] HELPER: local function raises inside test_mro_disagreement.
            # Arguments: exc, expected, callable, *args. Returns its computed value or None; used
            # by the enclosing test.
            try:
                callable(*args)
            except exc as msg:
                # the exact msg is generally considered an impl detail
                if support.check_impl_detail():
                    if not str(msg).startswith(expected):
                        self.fail("Message %r, expected %r" %
                                  (str(msg), expected))
            else:
                self.fail("Expected %s" % exc)

        # [CLASS DOC] HELPER: local class A defined inside test_mro_disagreement to exercise
        # descriptor/class-model behavior under test.
        class A(object): pass
        # [CLASS DOC] HELPER: local class B defined inside test_mro_disagreement to exercise
        # descriptor/class-model behavior under test.
        class B(A): pass
        # [CLASS DOC] HELPER: local class C defined inside test_mro_disagreement to exercise
        # descriptor/class-model behavior under test.
        class C(object): pass

        # Test some very simple errors
        raises(TypeError, "duplicate base class A",
               type, "X", (A, A), {})
        raises(TypeError, mro_err_msg,
               type, "X", (A, B), {})
        raises(TypeError, mro_err_msg,
               type, "X", (A, C, B), {})
        # Test a slightly more complex error
        # [CLASS DOC] HELPER: local class GridLayout defined inside test_mro_disagreement to
        # exercise descriptor/class-model behavior under test.
        class GridLayout(object): pass
        # [CLASS DOC] HELPER: local class HorizontalGrid defined inside test_mro_disagreement to
        # exercise descriptor/class-model behavior under test.
        class HorizontalGrid(GridLayout): pass
        # [CLASS DOC] HELPER: local class VerticalGrid defined inside test_mro_disagreement to
        # exercise descriptor/class-model behavior under test.
        class VerticalGrid(GridLayout): pass
        # [CLASS DOC] HELPER: local class HVGrid defined inside test_mro_disagreement to exercise
        # descriptor/class-model behavior under test.
        class HVGrid(HorizontalGrid, VerticalGrid): pass
        # [CLASS DOC] HELPER: local class VHGrid defined inside test_mro_disagreement to exercise
        # descriptor/class-model behavior under test.
        class VHGrid(VerticalGrid, HorizontalGrid): pass
        raises(TypeError, mro_err_msg,
               type, "ConfusedGrid", (HVGrid, VHGrid), {})

    def test_object_class(self):
        # [FUNCTION DOC] Test: checks object class behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing object class...
        a = object()
        self.assertEqual(a.__class__, object)
        self.assertEqual(type(a), object)
        b = object()
        self.assertNotEqual(a, b)
        self.assertNotHasAttr(a, "foo")
        try:
            a.foo = 12
        except (AttributeError, TypeError):
            pass
        else:
            self.fail("object() should not allow setting a foo attribute")
        self.assertNotHasAttr(object(), "__dict__")

        class Cdict(object):
            # [CLASS DOC] HELPER: local class Cdict defined inside test_object_class to exercise
            # descriptor/class-model behavior under test.
            pass
        x = Cdict()
        self.assertEqual(x.__dict__, {})
        x.foo = 1
        self.assertEqual(x.foo, 1)
        self.assertEqual(x.__dict__, {'foo': 1})

    def test_object_class_assignment_between_heaptypes_and_nonheaptypes(self):
        # [FUNCTION DOC] Test: checks object class assignment between heaptypes and nonheaptypes
        # behavior. Fails with AssertionError (via unittest assertions) if the observed behavior
        # differs from CPython semantics; no other side effects beyond temporary objects and
        # classes.
        class SubType(types.ModuleType):
            # [CLASS DOC] HELPER: local class SubType defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            a = 1

        m = types.ModuleType("m")
        self.assertTrue(m.__class__ is types.ModuleType)
        self.assertFalse(hasattr(m, "a"))

        m.__class__ = SubType
        self.assertTrue(m.__class__ is SubType)
        self.assertTrue(hasattr(m, "a"))

        m.__class__ = types.ModuleType
        self.assertTrue(m.__class__ is types.ModuleType)
        self.assertFalse(hasattr(m, "a"))

        # Make sure that builtin immutable objects don't support __class__
        # assignment, because the object instances may be interned.
        # We set __slots__ = () to ensure that the subclasses are
        # memory-layout compatible, and thus otherwise reasonable candidates
        # for __class__ assignment.

        # The following types have immutable instances, but are not
        # subclassable and thus don't need to be checked:
        #   NoneType, bool

        class MyInt(int):
            # [CLASS DOC] HELPER: local class MyInt defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            (1).__class__ = MyInt

        class MyFloat(float):
            # [CLASS DOC] HELPER: local class MyFloat defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            (1.0).__class__ = MyFloat

        class MyComplex(complex):
            # [CLASS DOC] HELPER: local class MyComplex defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            (1 + 2j).__class__ = MyComplex

        class MyStr(str):
            # [CLASS DOC] HELPER: local class MyStr defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            "a".__class__ = MyStr

        class MyBytes(bytes):
            # [CLASS DOC] HELPER: local class MyBytes defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            b"a".__class__ = MyBytes

        class MyTuple(tuple):
            # [CLASS DOC] HELPER: local class MyTuple defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            ().__class__ = MyTuple

        class MyFrozenSet(frozenset):
            # [CLASS DOC] HELPER: local class MyFrozenSet defined inside
            # test_object_class_assignment_between_heaptypes_and_nonheaptypes to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ()
        with self.assertRaises(TypeError):
            frozenset().__class__ = MyFrozenSet

    @unittest.expectedFailure
    def test_slots(self):
        # [FUNCTION DOC] Test: checks slots behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing __slots__...
        class C0(object):
            # [CLASS DOC] HELPER: local class C0 defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = []
        x = C0()
        self.assertNotHasAttr(x, "__dict__")
        self.assertNotHasAttr(x, "foo")

        class C1(object):
            # [CLASS DOC] HELPER: local class C1 defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['a']
        x = C1()
        self.assertNotHasAttr(x, "__dict__")
        self.assertNotHasAttr(x, "a")
        x.a = 1
        self.assertEqual(x.a, 1)
        x.a = None
        self.assertEqual(x.a, None)
        del x.a
        self.assertNotHasAttr(x, "a")

        class C3(object):
            # [CLASS DOC] HELPER: local class C3 defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['a', 'b', 'c']
        x = C3()
        self.assertNotHasAttr(x, "__dict__")
        self.assertNotHasAttr(x, 'a')
        self.assertNotHasAttr(x, 'b')
        self.assertNotHasAttr(x, 'c')
        x.a = 1
        x.b = 2
        x.c = 3
        self.assertEqual(x.a, 1)
        self.assertEqual(x.b, 2)
        self.assertEqual(x.c, 3)

        class C4(object):
            # [CLASS DOC] HELPER: local class C4 defined inside test_slots to exercise
            # descriptor/class-model behavior under test. Existing docstring says: Validate name
            # mangling
            """Validate name mangling"""
            __slots__ = ['__a']
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: constructor. Arguments: value. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.__a = value
            def get(self):
                # [FUNCTION DOC] HELPER: get of C4. Supporting routine for tests; returns a
                # computed value or None.
                return self.__a
        x = C4(5)
        self.assertNotHasAttr(x, '__dict__')
        self.assertNotHasAttr(x, '__a')
        self.assertEqual(x.get(), 5)
        try:
            x.__a = 6
        except AttributeError:
            pass
        else:
            self.fail("Double underscored names not mangled")

        # Make sure slot names are proper identifiers
        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                __slots__ = [None]
        except TypeError:
            pass
        else:
            self.fail("[None] slots not caught")
        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                __slots__ = ["foo bar"]
        except TypeError:
            pass
        else:
            self.fail("['foo bar'] slots not caught")
        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                __slots__ = ["foo\0bar"]
        except TypeError:
            pass
        else:
            self.fail("['foo\\0bar'] slots not caught")
        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                __slots__ = ["1"]
        except TypeError:
            pass
        else:
            self.fail("['1'] slots not caught")
        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                __slots__ = [""]
        except TypeError:
            pass
        else:
            self.fail("[''] slots not caught")

        class WithValidIdentifiers(object):
            # [CLASS DOC] HELPER: local class WithValidIdentifiers defined inside test_slots to
            # exercise descriptor/class-model behavior under test.
            __slots__ = ["a", "a_b", "_a", "A0123456789Z"]

        # Test a single string is not expanded as a sequence.
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = "abc"
        c = C()
        c.abc = 5
        self.assertEqual(c.abc, 5)

        # Test unicode slot names
        # Test a single unicode string is not expanded as a sequence.
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = "abc"
        c = C()
        c.abc = 5
        self.assertEqual(c.abc, 5)

        # _unicode_to_string used to modify slots in certain circumstances
        slots = ("foo", "bar")
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = slots
        x = C()
        x.foo = 5
        self.assertEqual(x.foo, 5)
        self.assertIs(type(slots[0]), str)
        # this used to leak references
        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                __slots__ = [chr(128)]
        except (TypeError, UnicodeEncodeError):
            pass
        else:
            self.fail("[chr(128)] slots not caught")

        # Test leaks
        class Counted(object):
            # [CLASS DOC] HELPER: local class Counted defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            counter = 0    # counts the number of instances alive
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                Counted.counter += 1
            def __del__(self):
                # [FUNCTION DOC] HELPER: special method __del__ of Counted. Implements the
                # protocol hook used by tests; returns the hook result.
                Counted.counter -= 1
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['a', 'b', 'c']
        x = C()
        x.a = Counted()
        x.b = Counted()
        x.c = Counted()
        self.assertEqual(Counted.counter, 3)
        del x
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            pass
        x = D()
        x.a = Counted()
        x.z = Counted()
        self.assertEqual(Counted.counter, 2)
        del x
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)
        class E(D):
            # [CLASS DOC] HELPER: local class E defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['e']
        x = E()
        x.a = Counted()
        x.z = Counted()
        x.e = Counted()
        self.assertEqual(Counted.counter, 3)
        del x
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)

        # Test cyclical leaks [SF bug 519621]
        class F(object):
            # [CLASS DOC] HELPER: local class F defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['a', 'b']
        s = F()
        s.a = [Counted(), s]
        self.assertEqual(Counted.counter, 1)
        s = None
        support.gc_collect()
        self.assertEqual(Counted.counter, 0)

        # Test lookup leaks [SF bug 572567]
        if hasattr(gc, 'get_objects'):
            class G(object):
                # [CLASS DOC] HELPER: local class G defined inside test_slots to exercise
                # descriptor/class-model behavior under test.
                def __eq__(self, other):
                    # [FUNCTION DOC] HELPER: special method __eq__ of G. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    return False
            g = G()
            orig_objects = len(gc.get_objects())
            for i in range(10):
                g==g
            new_objects = len(gc.get_objects())
            self.assertEqual(orig_objects, new_objects)

        class H(object):
            # [CLASS DOC] HELPER: local class H defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['a', 'b']
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.a = 1
                self.b = 2
            def __del__(self_):
                # [FUNCTION DOC] HELPER: special method __del__ of H. Arguments: self_. Implements
                # the protocol hook used by tests; returns the hook result.
                self.assertEqual(self_.a, 1)
                self.assertEqual(self_.b, 2)
        with support.captured_output('stderr') as s:
            h = H()
            del h
        self.assertEqual(s.getvalue(), '')

        class X(object):
            # [CLASS DOC] HELPER: local class X defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = "a"
        with self.assertRaises(AttributeError):
            del X().a

        # Inherit from object on purpose to check some backwards compatibility paths
        class X(object):
            # [CLASS DOC] HELPER: local class X defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = "a"
        with self.assertRaisesRegex(AttributeError, "'test.test_descr.ClassPropertiesAndMethods.test_slots.<locals>.X' object has no attribute 'a'"):
            X().a

        # Test string subclass in `__slots__`, see gh-98783
        class SubStr(str):
            # [CLASS DOC] HELPER: local class SubStr defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            pass
        class X(object):
            # [CLASS DOC] HELPER: local class X defined inside test_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = (SubStr('x'),)
        X().x = 1
        with self.assertRaisesRegex(AttributeError, "'X' object has no attribute 'a'"):
            X().a

    def test_slots_special(self):
        # [FUNCTION DOC] Test: checks slots special behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing __dict__ and __weakref__ in __slots__...
        class D(object):
            # [CLASS DOC] HELPER: local class D defined inside test_slots_special to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__dict__"]
        a = D()
        self.assertHasAttr(a, "__dict__")
        self.assertNotHasAttr(a, "__weakref__")
        a.foo = 42
        self.assertEqual(a.__dict__, {"foo": 42})

        class W(object):
            # [CLASS DOC] HELPER: local class W defined inside test_slots_special to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__weakref__"]
        a = W()
        self.assertHasAttr(a, "__weakref__")
        self.assertNotHasAttr(a, "__dict__")
        try:
            a.foo = 42
        except AttributeError:
            pass
        else:
            self.fail("shouldn't be allowed to set a.foo")

        class C1(W, D):
            # [CLASS DOC] HELPER: local class C1 defined inside test_slots_special to exercise
            # descriptor/class-model behavior under test.
            __slots__ = []
        a = C1()
        self.assertHasAttr(a, "__dict__")
        self.assertHasAttr(a, "__weakref__")
        a.foo = 42
        self.assertEqual(a.__dict__, {"foo": 42})

        class C2(D, W):
            # [CLASS DOC] HELPER: local class C2 defined inside test_slots_special to exercise
            # descriptor/class-model behavior under test.
            __slots__ = []
        a = C2()
        self.assertHasAttr(a, "__dict__")
        self.assertHasAttr(a, "__weakref__")
        a.foo = 42
        self.assertEqual(a.__dict__, {"foo": 42})

    def test_slots_special2(self):
        # [FUNCTION DOC] Test: checks slots special2 behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing __qualname__ and __classcell__ in __slots__
        class Meta(type):
            # [CLASS DOC] HELPER: local class Meta defined inside test_slots_special2 to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls, name, bases, namespace, attr):
                # [FUNCTION DOC] HELPER: special method __new__ of Meta. Arguments: name, bases,
                # namespace, attr. Implements the protocol hook used by tests; returns the hook
                # result.
                self.assertIn(attr, namespace)
                return super().__new__(cls, name, bases, namespace)

        class C1:
            # [CLASS DOC] HELPER: local class C1 defined inside test_slots_special2 to exercise
            # descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.b = 42
        class C2(C1, metaclass=Meta, attr="__classcell__"):
            # [CLASS DOC] HELPER: local class C2 defined inside test_slots_special2 to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__classcell__"]
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                super().__init__()
        self.assertIsInstance(C2.__dict__["__classcell__"],
                              types.MemberDescriptorType)
        c = C2()
        self.assertEqual(c.b, 42)
        self.assertNotHasAttr(c, "__classcell__")
        c.__classcell__ = 42
        self.assertEqual(c.__classcell__, 42)
        with self.assertRaises(TypeError):
            class C3:
                # [CLASS DOC] HELPER: local class C3 defined inside test_slots_special2 to
                # exercise descriptor/class-model behavior under test.
                __classcell__ = 42
                __slots__ = ["__classcell__"]

        class Q1(metaclass=Meta, attr="__qualname__"):
            # [CLASS DOC] HELPER: local class Q1 defined inside test_slots_special2 to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__qualname__"]
        self.assertEqual(Q1.__qualname__, C1.__qualname__[:-2] + "Q1")
        self.assertIsInstance(Q1.__dict__["__qualname__"],
                              types.MemberDescriptorType)
        q = Q1()
        self.assertNotHasAttr(q, "__qualname__")
        q.__qualname__ = "q"
        self.assertEqual(q.__qualname__, "q")
        with self.assertRaises(TypeError):
            class Q2:
                # [CLASS DOC] HELPER: local class Q2 defined inside test_slots_special2 to
                # exercise descriptor/class-model behavior under test.
                __qualname__ = object()
                __slots__ = ["__qualname__"]

    def test_slots_descriptor(self):
        # [FUNCTION DOC] Test: checks slots descriptor behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Issue2115: slot descriptors did not correctly check
        # the type of the given object
        import abc
        class MyABC(metaclass=abc.ABCMeta):
            # [CLASS DOC] HELPER: local class MyABC defined inside test_slots_descriptor to
            # exercise descriptor/class-model behavior under test.
            __slots__ = "a"

        class Unrelated(object):
            # [CLASS DOC] HELPER: local class Unrelated defined inside test_slots_descriptor to
            # exercise descriptor/class-model behavior under test.
            pass
        MyABC.register(Unrelated)

        u = Unrelated()
        self.assertIsInstance(u, MyABC)

        # This used to crash
        self.assertRaises(TypeError, MyABC.a.__set__, u, 3)

    def test_dynamics(self):
        # [FUNCTION DOC] Test: checks dynamics behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing class attribute propagation...
        class D(object):
            # [CLASS DOC] HELPER: local class D defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        class E(D):
            # [CLASS DOC] HELPER: local class E defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        class F(D):
            # [CLASS DOC] HELPER: local class F defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        D.foo = 1
        self.assertEqual(D.foo, 1)
        # Test that dynamic attributes are inherited
        self.assertEqual(E.foo, 1)
        self.assertEqual(F.foo, 1)
        # Test dynamic instances
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        a = C()
        self.assertNotHasAttr(a, "foobar")
        C.foobar = 2
        self.assertEqual(a.foobar, 2)
        C.method = lambda self: 42
        self.assertEqual(a.method(), 42)
        C.__repr__ = lambda self: "C()"
        self.assertEqual(repr(a), "C()")
        C.__int__ = lambda self: 100
        self.assertEqual(int(a), 100)
        self.assertEqual(a.foobar, 2)
        self.assertNotHasAttr(a, "spam")
        def mygetattr(self, name):
            # [FUNCTION DOC] HELPER: local function mygetattr inside test_dynamics. Arguments:
            # name. Returns its computed value or None; used by the enclosing test.
            if name == "spam":
                return "spam"
            raise AttributeError
        C.__getattr__ = mygetattr
        self.assertEqual(a.spam, "spam")
        a.new = 12
        self.assertEqual(a.new, 12)
        def mysetattr(self, name, value):
            # [FUNCTION DOC] HELPER: local function mysetattr inside test_dynamics. Arguments:
            # name, value. Returns its computed value or None; used by the enclosing test.
            if name == "spam":
                raise AttributeError
            return object.__setattr__(self, name, value)
        C.__setattr__ = mysetattr
        with self.assertRaises(AttributeError):
            a.spam = "not spam"

        self.assertEqual(a.spam, "spam")
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        d = D()
        d.foo = 1
        self.assertEqual(d.foo, 1)

        # Test handling of int*seq and seq*int
        class I(int):
            # [CLASS DOC] HELPER: local class I defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual("a"*I(2), "aa")
        self.assertEqual(I(2)*"a", "aa")
        self.assertEqual(2*I(3), 6)
        self.assertEqual(I(3)*2, 6)
        self.assertEqual(I(3)*I(2), 6)

        # Test comparison of classes with dynamic metaclasses
        class dynamicmetaclass(type):
            # [CLASS DOC] HELPER: local class dynamicmetaclass defined inside test_dynamics to
            # exercise descriptor/class-model behavior under test.
            pass
        class someclass(metaclass=dynamicmetaclass):
            # [CLASS DOC] HELPER: local class someclass defined inside test_dynamics to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertNotEqual(someclass, object)

    def test_errors(self):
        # [FUNCTION DOC] Test: checks errors behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing errors...
        try:
            class C(list, dict):
                # [CLASS DOC] HELPER: local class C defined inside test_errors to exercise
                # descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("inheritance from both list and dict should be illegal")

        try:
            class C(object, None):
                # [CLASS DOC] HELPER: local class C defined inside test_errors to exercise
                # descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("inheritance from non-type should be illegal")
        class Classic:
            # [CLASS DOC] HELPER: local class Classic defined inside test_errors to exercise
            # descriptor/class-model behavior under test.
            pass

        try:
            class C(type(len)):
                # [CLASS DOC] HELPER: local class C defined inside test_errors to exercise
                # descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("inheritance from CFunction should be illegal")

        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_errors to exercise
                # descriptor/class-model behavior under test.
                __slots__ = 1
        except TypeError:
            pass
        else:
            self.fail("__slots__ = 1 should be illegal")

        try:
            class C(object):
                # [CLASS DOC] HELPER: local class C defined inside test_errors to exercise
                # descriptor/class-model behavior under test.
                __slots__ = [1]
        except TypeError:
            pass
        else:
            self.fail("__slots__ = [1] should be illegal")

        class M1(type):
            # [CLASS DOC] HELPER: local class M1 defined inside test_errors to exercise
            # descriptor/class-model behavior under test.
            pass
        class M2(type):
            # [CLASS DOC] HELPER: local class M2 defined inside test_errors to exercise
            # descriptor/class-model behavior under test.
            pass
        class A1(object, metaclass=M1):
            # [CLASS DOC] HELPER: local class A1 defined inside test_errors to exercise
            # descriptor/class-model behavior under test.
            pass
        class A2(object, metaclass=M2):
            # [CLASS DOC] HELPER: local class A2 defined inside test_errors to exercise
            # descriptor/class-model behavior under test.
            pass
        try:
            class B(A1, A2):
                # [CLASS DOC] HELPER: local class B defined inside test_errors to exercise
                # descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("finding the most derived metaclass should have failed")

    def test_classmethods(self):
        # [FUNCTION DOC] Test: checks classmethods behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing class methods...
        class C(object):
            # [FUNCTION DOC] HELPER: foo of C. Arguments: *a. Supporting routine for tests;
            # returns a computed value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_classmethods to exercise
            # descriptor/class-model behavior under test.
            def foo(*a): return a
            goo = classmethod(foo)
        c = C()
        self.assertEqual(C.goo(1), (C, 1))
        self.assertEqual(c.goo(1), (C, 1))
        self.assertEqual(c.foo(1), (c, 1))
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_classmethods to exercise
            # descriptor/class-model behavior under test.
            pass
        d = D()
        self.assertEqual(D.goo(1), (D, 1))
        self.assertEqual(d.goo(1), (D, 1))
        self.assertEqual(d.foo(1), (d, 1))
        self.assertEqual(D.foo(d, 1), (d, 1))
        # Test for a specific crash (SF bug 528132)
        def f(cls, arg):
            # [FUNCTION DOC] HELPER: local function f inside test_classmethods. Arguments: arg.
            # Returns its computed value or None; used by the enclosing test.
            "f docstring"
            return (cls, arg)
        ff = classmethod(f)
        self.assertEqual(ff.__get__(0, int)(42), (int, 42))
        self.assertEqual(ff.__get__(0)(42), (int, 42))

        # Test super() with classmethods (SF bug 535444)
        self.assertEqual(C.goo.__self__, C)
        self.assertEqual(D.goo.__self__, D)
        self.assertEqual(super(D,D).goo.__self__, D)
        self.assertEqual(super(D,d).goo.__self__, D)
        self.assertEqual(super(D,D).goo(), (D,))
        self.assertEqual(super(D,d).goo(), (D,))

        # Verify that a non-callable will raise
        meth = classmethod(1).__get__(1)
        self.assertRaises(TypeError, meth)

        # Verify that classmethod() doesn't allow keyword args
        try:
            classmethod(f, kw=1)
        except TypeError:
            pass
        else:
            self.fail("classmethod shouldn't accept keyword args")

        cm = classmethod(f)
        cm_dict = {'__annotations__': {},
                   '__doc__': (
                       "f docstring"
                       if support.HAVE_PY_DOCSTRINGS
                       else None
                    ),
                   '__module__': __name__,
                   '__name__': 'f',
                   '__qualname__': f.__qualname__}
        self.assertEqual(cm.__dict__, cm_dict)

        cm.x = 42
        self.assertEqual(cm.x, 42)
        self.assertEqual(cm.__dict__, {"x" : 42, **cm_dict})
        del cm.x
        self.assertNotHasAttr(cm, "x")

    @support.refcount_test
    def test_refleaks_in_classmethod___init__(self):
        # [FUNCTION DOC] Test: checks refleaks in classmethod   init behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        cm = classmethod(None)
        refs_before = gettotalrefcount()
        for i in range(100):
            cm.__init__(None)
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_classmethods_in_c(self):
        # [FUNCTION DOC] Test: checks classmethods in c behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing C-based class methods...
        import xxsubtype as spam
        a = (1, 2, 3)
        d = {'abc': 123}
        x, a1, d1 = spam.spamlist.classmeth(*a, **d)
        self.assertEqual(x, spam.spamlist)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)
        x, a1, d1 = spam.spamlist().classmeth(*a, **d)
        self.assertEqual(x, spam.spamlist)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)
        spam_cm = spam.spamlist.__dict__['classmeth']
        x2, a2, d2 = spam_cm(spam.spamlist, *a, **d)
        self.assertEqual(x2, spam.spamlist)
        self.assertEqual(a2, a1)
        self.assertEqual(d2, d1)
        # [CLASS DOC] HELPER: local class SubSpam defined inside test_classmethods_in_c to
        # exercise descriptor/class-model behavior under test.
        class SubSpam(spam.spamlist): pass
        x2, a2, d2 = spam_cm(SubSpam, *a, **d)
        self.assertEqual(x2, SubSpam)
        self.assertEqual(a2, a1)
        self.assertEqual(d2, d1)

        with self.assertRaises(TypeError) as cm:
            spam_cm()
        self.assertEqual(
            str(cm.exception),
            "descriptor 'classmeth' of 'xxsubtype.spamlist' "
            "object needs an argument")

        with self.assertRaises(TypeError) as cm:
            spam_cm(spam.spamlist())
        self.assertEqual(
            str(cm.exception),
            "descriptor 'classmeth' for type 'xxsubtype.spamlist' "
            "needs a type, not a 'xxsubtype.spamlist' as arg 2")

        with self.assertRaises(TypeError) as cm:
            spam_cm(list)
        expected_errmsg = (
            "descriptor 'classmeth' requires a subtype of 'xxsubtype.spamlist' "
            "but received 'list'")
        self.assertEqual(str(cm.exception), expected_errmsg)

        with self.assertRaises(TypeError) as cm:
            spam_cm.__get__(None, list)
        self.assertEqual(str(cm.exception), expected_errmsg)

    def test_staticmethods(self):
        # [FUNCTION DOC] Test: checks staticmethods behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing static methods...
        class C(object):
            # [FUNCTION DOC] HELPER: foo of C. Arguments: *a. Supporting routine for tests;
            # returns a computed value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_staticmethods to exercise
            # descriptor/class-model behavior under test.
            def foo(*a): return a
            goo = staticmethod(foo)
        c = C()
        self.assertEqual(C.goo(1), (1,))
        self.assertEqual(c.goo(1), (1,))
        self.assertEqual(c.foo(1), (c, 1,))
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_staticmethods to exercise
            # descriptor/class-model behavior under test.
            pass
        d = D()
        self.assertEqual(D.goo(1), (1,))
        self.assertEqual(d.goo(1), (1,))
        self.assertEqual(d.foo(1), (d, 1))
        self.assertEqual(D.foo(d, 1), (d, 1))
        sm = staticmethod(None)
        self.assertEqual(sm.__dict__, {'__doc__': None.__doc__})
        sm.x = 42
        self.assertEqual(sm.x, 42)
        self.assertEqual(sm.__dict__, {"x" : 42, '__doc__': None.__doc__})
        del sm.x
        self.assertNotHasAttr(sm, "x")

    @support.refcount_test
    def test_refleaks_in_staticmethod___init__(self):
        # [FUNCTION DOC] Test: checks refleaks in staticmethod   init behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        gettotalrefcount = support.get_attribute(sys, 'gettotalrefcount')
        sm = staticmethod(None)
        refs_before = gettotalrefcount()
        for i in range(100):
            sm.__init__(None)
        self.assertAlmostEqual(gettotalrefcount() - refs_before, 0, delta=10)

    @support.impl_detail("the module 'xxsubtype' is internal")
    @unittest.skipIf(xxsubtype is None, "requires xxsubtype module")
    def test_staticmethods_in_c(self):
        # [FUNCTION DOC] Test: checks staticmethods in c behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing C-based static methods...
        import xxsubtype as spam
        a = (1, 2, 3)
        d = {"abc": 123}
        x, a1, d1 = spam.spamlist.staticmeth(*a, **d)
        self.assertEqual(x, None)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)
        x, a1, d2 = spam.spamlist().staticmeth(*a, **d)
        self.assertEqual(x, None)
        self.assertEqual(a, a1)
        self.assertEqual(d, d1)

    def test_classic(self):
        # [FUNCTION DOC] Test: checks classic behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing classic classes...
        class C:
            # [FUNCTION DOC] HELPER: foo of C. Arguments: *a. Supporting routine for tests;
            # returns a computed value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_classic to exercise
            # descriptor/class-model behavior under test.
            def foo(*a): return a
            goo = classmethod(foo)
        c = C()
        self.assertEqual(C.goo(1), (C, 1))
        self.assertEqual(c.goo(1), (C, 1))
        self.assertEqual(c.foo(1), (c, 1))
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_classic to exercise
            # descriptor/class-model behavior under test.
            pass
        d = D()
        self.assertEqual(D.goo(1), (D, 1))
        self.assertEqual(d.goo(1), (D, 1))
        self.assertEqual(d.foo(1), (d, 1))
        self.assertEqual(D.foo(d, 1), (d, 1))
        class E: # *not* subclassing from C
            # [CLASS DOC] HELPER: local class E defined inside test_classic to exercise
            # descriptor/class-model behavior under test.
            foo = C.foo
        self.assertEqual(E().foo.__func__, C.foo) # i.e., unbound
        self.assertTrue(repr(C.foo.__get__(C())).startswith("<bound method "))

    def test_compattr(self):
        # [FUNCTION DOC] Test: checks compattr behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing computed attributes...
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_compattr to exercise
            # descriptor/class-model behavior under test.
            class computed_attribute(object):
                # [CLASS DOC] HELPER: nested helper class computed_attribute inside class C used
                # by tests.
                def __init__(self, get, set=None, delete=None):
                    # [FUNCTION DOC] HELPER: constructor. Arguments: get, set, delete. Stores the
                    # given values on the instance, and for test cases initializes the base
                    # CPythonTestCase.
                    self.__get = get
                    self.__set = set
                    self.__delete = delete
                def __get__(self, obj, type=None):
                    # [FUNCTION DOC] HELPER: special method __get__ of computed_attribute.
                    # Arguments: obj, type. Implements the protocol hook used by tests; returns
                    # the hook result.
                    return self.__get(obj)
                def __set__(self, obj, value):
                    # [FUNCTION DOC] HELPER: special method __set__ of computed_attribute.
                    # Arguments: obj, value. Implements the protocol hook used by tests; returns
                    # the hook result.
                    return self.__set(obj, value)
                def __delete__(self, obj):
                    # [FUNCTION DOC] HELPER: special method __delete__ of computed_attribute.
                    # Arguments: obj. Implements the protocol hook used by tests; returns the hook
                    # result.
                    return self.__delete(obj)
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.__x = 0
            def __get_x(self):
                # [FUNCTION DOC] HELPER: __get_x of C. Supporting routine for tests; returns a
                # computed value or None.
                x = self.__x
                self.__x = x+1
                return x
            def __set_x(self, x):
                # [FUNCTION DOC] HELPER: __set_x of C. Arguments: x. Supporting routine for tests;
                # returns a computed value or None.
                self.__x = x
            def __delete_x(self):
                # [FUNCTION DOC] HELPER: __delete_x of C. Supporting routine for tests; returns a
                # computed value or None.
                del self.__x
            x = computed_attribute(__get_x, __set_x, __delete_x)
        a = C()
        self.assertEqual(a.x, 0)
        self.assertEqual(a.x, 1)
        a.x = 10
        self.assertEqual(a.x, 10)
        self.assertEqual(a.x, 11)
        del a.x
        self.assertNotHasAttr(a, 'x')

    def test_newslots(self):
        # [FUNCTION DOC] Test: checks newslots behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing __new__ slot override...
        class C(list):
            # [CLASS DOC] HELPER: local class C defined inside test_newslots to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls):
                # [FUNCTION DOC] HELPER: special method __new__ of C. Implements the protocol hook
                # used by tests; returns the hook result.
                self = list.__new__(cls)
                self.foo = 1
                return self
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.foo = self.foo + 2
        a = C()
        self.assertEqual(a.foo, 3)
        self.assertEqual(a.__class__, C)
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_newslots to exercise
            # descriptor/class-model behavior under test.
            pass
        b = D()
        self.assertEqual(b.foo, 3)
        self.assertEqual(b.__class__, D)

    @unittest.expectedFailure
    def test_bad_new(self):
        # [FUNCTION DOC] Test: checks bad new behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        self.assertRaises(TypeError, object.__new__)
        self.assertRaises(TypeError, object.__new__, '')
        self.assertRaises(TypeError, list.__new__, object)
        self.assertRaises(TypeError, object.__new__, list)
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_bad_new to exercise
            # descriptor/class-model behavior under test.
            __new__ = list.__new__
        self.assertRaises(TypeError, C)
        class C(list):
            # [CLASS DOC] HELPER: local class C defined inside test_bad_new to exercise
            # descriptor/class-model behavior under test.
            __new__ = object.__new__
        self.assertRaises(TypeError, C)

    def test_object_new(self):
        # [FUNCTION DOC] Test: checks object new behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_object_new to exercise
            # descriptor/class-model behavior under test.
            pass
        object.__new__(A)
        self.assertRaises(TypeError, object.__new__, A, 5)
        object.__init__(A())
        self.assertRaises(TypeError, object.__init__, A(), 5)

        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_object_new to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: constructor. Arguments: foo. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.foo = foo
        object.__new__(A)
        object.__new__(A, 5)
        object.__init__(A(3))
        self.assertRaises(TypeError, object.__init__, A(3), 5)

        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_object_new to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls, foo):
                # [FUNCTION DOC] HELPER: special method __new__ of A. Arguments: foo. Implements
                # the protocol hook used by tests; returns the hook result.
                return object.__new__(cls)
        object.__new__(A)
        self.assertRaises(TypeError, object.__new__, A, 5)
        object.__init__(A(3))
        object.__init__(A(3), 5)

        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_object_new to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls, foo):
                # [FUNCTION DOC] HELPER: special method __new__ of A. Arguments: foo. Implements
                # the protocol hook used by tests; returns the hook result.
                return object.__new__(cls)
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: constructor. Arguments: foo. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.foo = foo
        object.__new__(A)
        self.assertRaises(TypeError, object.__new__, A, 5)
        object.__init__(A(3))
        self.assertRaises(TypeError, object.__init__, A(3), 5)

    @unittest.expectedFailure
    def test_restored_object_new(self):
        # [FUNCTION DOC] Test: checks restored object new behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_restored_object_new to
            # exercise descriptor/class-model behavior under test.
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: special method __new__ of A. Arguments: *args, **kwargs.
                # Implements the protocol hook used by tests; returns the hook result.
                raise AssertionError
        self.assertRaises(AssertionError, A)
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_restored_object_new to
            # exercise descriptor/class-model behavior under test.
            __new__ = object.__new__
            def __init__(self, foo):
                # [FUNCTION DOC] HELPER: constructor. Arguments: foo. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.foo = foo
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            b = B(3)
        self.assertEqual(b.foo, 3)
        self.assertEqual(b.__class__, B)
        del B.__new__
        self.assertRaises(AssertionError, B)
        del A.__new__
        with warnings.catch_warnings():
            warnings.simplefilter('error', DeprecationWarning)
            b = B(3)
        self.assertEqual(b.foo, 3)
        self.assertEqual(b.__class__, B)

    def test_altmro(self):
        # [FUNCTION DOC] Test: checks altmro behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing mro() and overriding it...
        class A(object):
            # [FUNCTION DOC] HELPER: f of A. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class A defined inside test_altmro to exercise
            # descriptor/class-model behavior under test.
            def f(self): return "A"
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_altmro to exercise
            # descriptor/class-model behavior under test.
            pass
        class C(A):
            # [FUNCTION DOC] HELPER: f of C. Supporting routine for tests; returns a computed
            # value or None.
            # [CLASS DOC] HELPER: local class C defined inside test_altmro to exercise
            # descriptor/class-model behavior under test.
            def f(self): return "C"
        class D(B, C):
            # [CLASS DOC] HELPER: local class D defined inside test_altmro to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(A.mro(), [A, object])
        self.assertEqual(A.__mro__, (A, object))
        self.assertEqual(B.mro(), [B, A, object])
        self.assertEqual(B.__mro__, (B, A, object))
        self.assertEqual(C.mro(), [C, A, object])
        self.assertEqual(C.__mro__, (C, A, object))
        self.assertEqual(D.mro(), [D, B, C, A, object])
        self.assertEqual(D.__mro__, (D, B, C, A, object))
        self.assertEqual(D().f(), "C")

        class PerverseMetaType(type):
            # [CLASS DOC] HELPER: local class PerverseMetaType defined inside test_altmro to
            # exercise descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of PerverseMetaType. Supporting routine for tests;
                # returns a computed value or None.
                L = type.mro(cls)
                L.reverse()
                return L
        class X(D,B,C,A, metaclass=PerverseMetaType):
            # [CLASS DOC] HELPER: local class X defined inside test_altmro to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(X.__mro__, (object, A, C, B, D, X))
        self.assertEqual(X().f(), "A")

        try:
            class _metaclass(type):
                # [CLASS DOC] HELPER: local class _metaclass defined inside test_altmro to
                # exercise descriptor/class-model behavior under test.
                def mro(self):
                    # [FUNCTION DOC] HELPER: mro of _metaclass. Supporting routine for tests;
                    # returns a computed value or None.
                    return [self, dict, object]
            class X(object, metaclass=_metaclass):
                # [CLASS DOC] HELPER: local class X defined inside test_altmro to exercise
                # descriptor/class-model behavior under test.
                pass
            # In CPython, the class creation above already raises
            # TypeError, as a protection against the fact that
            # instances of X would segfault it.  In other Python
            # implementations it would be ok to let the class X
            # be created, but instead get a clean TypeError on the
            # __setitem__ below.
            x = object.__new__(X)
            x[5] = 6
        except TypeError:
            pass
        else:
            self.fail("devious mro() return not caught")

        try:
            class _metaclass(type):
                # [CLASS DOC] HELPER: local class _metaclass defined inside test_altmro to
                # exercise descriptor/class-model behavior under test.
                def mro(self):
                    # [FUNCTION DOC] HELPER: mro of _metaclass. Supporting routine for tests;
                    # returns a computed value or None.
                    return [1]
            class X(object, metaclass=_metaclass):
                # [CLASS DOC] HELPER: local class X defined inside test_altmro to exercise
                # descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("non-class mro() return not caught")

        try:
            class _metaclass(type):
                # [CLASS DOC] HELPER: local class _metaclass defined inside test_altmro to
                # exercise descriptor/class-model behavior under test.
                def mro(self):
                    # [FUNCTION DOC] HELPER: mro of _metaclass. Supporting routine for tests;
                    # returns a computed value or None.
                    return 1
            class X(object, metaclass=_metaclass):
                # [CLASS DOC] HELPER: local class X defined inside test_altmro to exercise
                # descriptor/class-model behavior under test.
                pass
        except TypeError:
            pass
        else:
            self.fail("non-sequence mro() return not caught")

    def test_overloading(self):
        # [FUNCTION DOC] Test: checks overloading behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing operator overloading...

        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside test_overloading to exercise
            # descriptor/class-model behavior under test. Existing docstring says: Intermediate
            # class because object doesn't have a __setattr__
            "Intermediate class because object doesn't have a __setattr__"

        class C(B):
            # [CLASS DOC] HELPER: local class C defined inside test_overloading to exercise
            # descriptor/class-model behavior under test.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattr__ of C. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                if name == "foo":
                    return ("getattr", name)
                else:
                    raise AttributeError
            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: special method __setattr__ of C. Arguments: name, value.
                # Implements the protocol hook used by tests; returns the hook result.
                if name == "foo":
                    self.setattr = (name, value)
                else:
                    return B.__setattr__(self, name, value)
            def __delattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __delattr__ of C. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                if name == "foo":
                    self.delattr = name
                else:
                    return B.__delattr__(self, name)

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: special method __getitem__ of C. Arguments: key.
                # Implements the protocol hook used by tests; returns the hook result.
                return ("getitem", key)
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: special method __setitem__ of C. Arguments: key, value.
                # Implements the protocol hook used by tests; returns the hook result.
                self.setitem = (key, value)
            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: special method __delitem__ of C. Arguments: key.
                # Implements the protocol hook used by tests; returns the hook result.
                self.delitem = key

        a = C()
        self.assertEqual(a.foo, ("getattr", "foo"))
        a.foo = 12
        self.assertEqual(a.setattr, ("foo", 12))
        del a.foo
        self.assertEqual(a.delattr, "foo")

        self.assertEqual(a[12], ("getitem", 12))
        a[12] = 21
        self.assertEqual(a.setitem, (12, 21))
        del a[12]
        self.assertEqual(a.delitem, 12)

        self.assertEqual(a[0:10], ("getitem", slice(0, 10)))
        a[0:10] = "foo"
        self.assertEqual(a.setitem, (slice(0, 10), "foo"))
        del a[0:10]
        self.assertEqual(a.delitem, (slice(0, 10)))

    def test_load_attr_extended_arg(self):
        # [FUNCTION DOC] Test: checks load attr extended arg behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # https://github.com/python/cpython/issues/91625
        class Numbers:
            # [CLASS DOC] HELPER: local class Numbers defined inside test_load_attr_extended_arg
            # to exercise descriptor/class-model behavior under test.
            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: special method __getattr__ of Numbers. Arguments: attr.
                # Implements the protocol hook used by tests; returns the hook result.
                return int(attr.lstrip("_"))
        attrs = ", ".join(f"Z._{n:03d}" for n in range(280))
        code = f"def number_attrs(Z):\n    return [ {attrs} ]"
        ns = {}
        with torch._dynamo.error_on_graph_break(False):
            exec(code, ns)
        number_attrs = ns["number_attrs"]
        # Warm up the function for quickening (PEP 659)
        for _ in range(30):
            self.assertEqual(number_attrs(Numbers()), list(range(280)))

    def test_methods(self):
        # [FUNCTION DOC] Test: checks methods behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing methods...
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_methods to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: constructor. Arguments: x. Stores the given values on the
                # instance, and for test cases initializes the base CPythonTestCase.
                self.x = x
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of C. Supporting routine for tests; returns a
                # computed value or None.
                return self.x
        c1 = C(1)
        self.assertEqual(c1.foo(), 1)
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_methods to exercise
            # descriptor/class-model behavior under test.
            boo = C.foo
            goo = c1.foo
        d2 = D(2)
        self.assertEqual(d2.foo(), 2)
        self.assertEqual(d2.boo(), 2)
        self.assertEqual(d2.goo(), 1)
        class E(object):
            # [CLASS DOC] HELPER: local class E defined inside test_methods to exercise
            # descriptor/class-model behavior under test.
            foo = C.foo
        self.assertEqual(E().foo.__func__, C.foo) # i.e., unbound
        self.assertTrue(repr(C.foo.__get__(C(1))).startswith("<bound method "))

    @support.impl_detail("testing error message from implementation")
    def test_methods_in_c(self):
        # [FUNCTION DOC] Test: checks methods in c behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # This test checks error messages in builtin method descriptor.
        # It is allowed that other Python implementations use
        # different error messages.
        set_add = set.add

        expected_errmsg = "unbound method set.add() needs an argument"

        with self.assertRaises(TypeError) as cm:
            set_add()
        self.assertEqual(cm.exception.args[0], expected_errmsg)

        expected_errmsg = "descriptor 'add' for 'set' objects doesn't apply to a 'int' object"

        with self.assertRaises(TypeError) as cm:
            set_add(0)
        self.assertEqual(cm.exception.args[0], expected_errmsg)

        with self.assertRaises(TypeError) as cm:
            set_add.__get__(0)
        self.assertEqual(cm.exception.args[0], expected_errmsg)

    def test_special_method_lookup(self):
        # [FUNCTION DOC] Test: checks special method lookup behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # The lookup of special methods bypasses __getattr__ and
        # __getattribute__, but they still can be descriptors.

        def run_context(manager):
            # [FUNCTION DOC] HELPER: local function run_context inside test_special_method_lookup.
            # Arguments: manager. Returns its computed value or None; used by the enclosing test.
            with manager:
                pass
        def iden(self):
            # [FUNCTION DOC] HELPER: local function iden inside test_special_method_lookup.
            # Returns its computed value or None; used by the enclosing test.
            return self
        def hello(self):
            # [FUNCTION DOC] HELPER: local function hello inside test_special_method_lookup.
            # Returns its computed value or None; used by the enclosing test.
            return b"hello"
        def empty_seq(self):
            # [FUNCTION DOC] HELPER: local function empty_seq inside test_special_method_lookup.
            # Returns its computed value or None; used by the enclosing test.
            return []
        def zero(self):
            # [FUNCTION DOC] HELPER: local function zero inside test_special_method_lookup.
            # Returns its computed value or None; used by the enclosing test.
            return 0
        def complex_num(self):
            # [FUNCTION DOC] HELPER: local function complex_num inside test_special_method_lookup.
            # Returns its computed value or None; used by the enclosing test.
            return 1j
        def stop(self):
            # [FUNCTION DOC] HELPER: local function stop inside test_special_method_lookup.
            # Returns its computed value or None; used by the enclosing test.
            raise StopIteration
        def return_true(self, thing=None):
            # [FUNCTION DOC] HELPER: local function return_true inside test_special_method_lookup.
            # Arguments: thing. Returns its computed value or None; used by the enclosing test.
            return True
        def do_isinstance(obj):
            # [FUNCTION DOC] HELPER: local function do_isinstance inside
            # test_special_method_lookup. Arguments: obj. Returns its computed value or None; used
            # by the enclosing test.
            return isinstance(int, obj)
        def do_issubclass(obj):
            # [FUNCTION DOC] HELPER: local function do_issubclass inside
            # test_special_method_lookup. Arguments: obj. Returns its computed value or None; used
            # by the enclosing test.
            return issubclass(int, obj)
        def do_dict_missing(checker):
            # [FUNCTION DOC] HELPER: local function do_dict_missing inside
            # test_special_method_lookup. Arguments: checker. Returns its computed value or None;
            # used by the enclosing test.
            class DictSub(checker.__class__, dict):
                # [CLASS DOC] HELPER: local class DictSub defined inside do_dict_missing to
                # exercise descriptor/class-model behavior under test.
                pass
            self.assertEqual(DictSub()["hi"], 4)
        def some_number(self_, key):
            # [FUNCTION DOC] HELPER: local function some_number inside test_special_method_lookup.
            # Arguments: self_, key. Returns its computed value or None; used by the enclosing
            # test.
            self.assertEqual(key, "hi")
            return 4
        # [FUNCTION DOC] HELPER: local function swallow inside test_special_method_lookup.
        # Arguments: *args. Returns its computed value or None; used by the enclosing test.
        def swallow(*args): pass
        def format_impl(self, spec):
            # [FUNCTION DOC] HELPER: local function format_impl inside test_special_method_lookup.
            # Arguments: spec. Returns its computed value or None; used by the enclosing test.
            return "hello"

        # It would be nice to have every special method tested here, but I'm
        # only listing the ones I can remember outside of typeobject.c, since it
        # does it right.
        specials = [
            ("__bytes__", bytes, hello, set(), {}),
            ("__reversed__", reversed, empty_seq, set(), {}),
            ("__length_hint__", list, zero, set(),
             {"__iter__" : iden, "__next__" : stop}),
            ("__sizeof__", sys.getsizeof, zero, set(), {}),
            ("__instancecheck__", do_isinstance, return_true, set(), {}),
            ("__missing__", do_dict_missing, some_number,
             set(("__class__",)), {}),
            ("__subclasscheck__", do_issubclass, return_true,
             set(("__bases__",)), {}),
            ("__enter__", run_context, iden, set(), {"__exit__" : swallow}),
            ("__exit__", run_context, swallow, set(), {"__enter__" : iden}),
            ("__complex__", complex, complex_num, set(), {}),
            ("__format__", format, format_impl, set(), {}),
            ("__floor__", math.floor, zero, set(), {}),
            ("__trunc__", math.trunc, zero, set(), {}),
            ("__ceil__", math.ceil, zero, set(), {}),
            ("__dir__", dir, empty_seq, set(), {}),
            ("__round__", round, zero, set(), {}),
            ]

        class Checker(object):
            # [CLASS DOC] HELPER: local class Checker defined inside test_special_method_lookup to
            # exercise descriptor/class-model behavior under test.
            def __getattr__(self, attr, test=self):
                # [FUNCTION DOC] HELPER: special method __getattr__ of Checker. Arguments: attr,
                # test. Implements the protocol hook used by tests; returns the hook result.
                test.fail("__getattr__ called with {0}".format(attr))
            def __getattribute__(self, attr, test=self):
                # [FUNCTION DOC] HELPER: special method __getattribute__ of Checker. Arguments:
                # attr, test. Implements the protocol hook used by tests; returns the hook result.
                if attr not in ok:
                    test.fail("__getattribute__ called with {0}".format(attr))
                return object.__getattribute__(self, attr)
        class SpecialDescr(object):
            # [CLASS DOC] HELPER: local class SpecialDescr defined inside
            # test_special_method_lookup to exercise descriptor/class-model behavior under test.
            def __init__(self, impl):
                # [FUNCTION DOC] HELPER: constructor. Arguments: impl. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.impl = impl
            def __get__(self, obj, owner):
                # [FUNCTION DOC] HELPER: special method __get__ of SpecialDescr. Arguments: obj,
                # owner. Implements the protocol hook used by tests; returns the hook result.
                record.append(1)
                return self.impl.__get__(obj, owner)
        class MyException(Exception):
            # [CLASS DOC] HELPER: local class MyException defined inside
            # test_special_method_lookup to exercise descriptor/class-model behavior under test.
            pass
        class ErrDescr(object):
            # [CLASS DOC] HELPER: local class ErrDescr defined inside test_special_method_lookup
            # to exercise descriptor/class-model behavior under test.
            def __get__(self, obj, owner):
                # [FUNCTION DOC] HELPER: special method __get__ of ErrDescr. Arguments: obj,
                # owner. Implements the protocol hook used by tests; returns the hook result.
                raise MyException

        for name, runner, meth_impl, ok, env in specials:
            class X(Checker):
                # [CLASS DOC] HELPER: local class X defined inside test_special_method_lookup to
                # exercise descriptor/class-model behavior under test.
                pass
            for attr, obj in env.items():
                setattr(X, attr, obj)
            setattr(X, name, meth_impl)
            runner(X())

            record = []
            class X(Checker):
                # [CLASS DOC] HELPER: local class X defined inside test_special_method_lookup to
                # exercise descriptor/class-model behavior under test.
                pass
            for attr, obj in env.items():
                setattr(X, attr, obj)
            setattr(X, name, SpecialDescr(meth_impl))
            runner(X())
            self.assertEqual(record, [1], name)

            class X(Checker):
                # [CLASS DOC] HELPER: local class X defined inside test_special_method_lookup to
                # exercise descriptor/class-model behavior under test.
                pass
            for attr, obj in env.items():
                setattr(X, attr, obj)
            setattr(X, name, ErrDescr())
            self.assertRaises(MyException, runner, X())

    def test_specials(self):
        # [FUNCTION DOC] Test: checks specials behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing special operators...
        # Test operators like __hash__ for which a built-in default exists

        # Test the default behavior for static classes
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_specials to exercise
            # descriptor/class-model behavior under test.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: special method __getitem__ of C. Arguments: i. Implements
                # the protocol hook used by tests; returns the hook result.
                if 0 <= i < 10: return i
                raise IndexError
        c1 = C()
        c2 = C()
        self.assertFalse(not c1)
        self.assertNotEqual(id(c1), id(c2))
        hash(c1)
        hash(c2)
        self.assertEqual(c1, c1)
        self.assertTrue(c1 != c2)
        self.assertFalse(c1 != c1)
        self.assertFalse(c1 == c2)
        # Note that the module name appears in str/repr, and that varies
        # depending on whether this test is run standalone or from a framework.
        self.assertGreaterEqual(str(c1).find('C object at '), 0)
        self.assertEqual(str(c1), repr(c1))
        self.assertNotIn(-1, c1)
        for i in range(10):
            self.assertIn(i, c1)
        self.assertNotIn(10, c1)
        # Test the default behavior for dynamic classes
        class D(object):
            # [CLASS DOC] HELPER: local class D defined inside test_specials to exercise
            # descriptor/class-model behavior under test.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: special method __getitem__ of D. Arguments: i. Implements
                # the protocol hook used by tests; returns the hook result.
                if 0 <= i < 10: return i
                raise IndexError
        d1 = D()
        d2 = D()
        self.assertFalse(not d1)
        self.assertNotEqual(id(d1), id(d2))
        hash(d1)
        hash(d2)
        self.assertEqual(d1, d1)
        self.assertNotEqual(d1, d2)
        self.assertFalse(d1 != d1)
        self.assertFalse(d1 == d2)
        # Note that the module name appears in str/repr, and that varies
        # depending on whether this test is run standalone or from a framework.
        self.assertGreaterEqual(str(d1).find('D object at '), 0)
        self.assertEqual(str(d1), repr(d1))
        self.assertNotIn(-1, d1)
        for i in range(10):
            self.assertIn(i, d1)
        self.assertNotIn(10, d1)
        # Test overridden behavior
        class Proxy(object):
            # [CLASS DOC] HELPER: local class Proxy defined inside test_specials to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: constructor. Arguments: x. Stores the given values on the
                # instance, and for test cases initializes the base CPythonTestCase.
                self.x = x
            def __bool__(self):
                # [FUNCTION DOC] HELPER: special method __bool__ of Proxy. Implements the protocol
                # hook used by tests; returns the hook result.
                return not not self.x
            def __hash__(self):
                # [FUNCTION DOC] HELPER: special method __hash__ of Proxy. Implements the protocol
                # hook used by tests; returns the hook result.
                return hash(self.x)
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: special method __eq__ of Proxy. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.x == other
            def __ne__(self, other):
                # [FUNCTION DOC] HELPER: special method __ne__ of Proxy. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.x != other
            def __ge__(self, other):
                # [FUNCTION DOC] HELPER: special method __ge__ of Proxy. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.x >= other
            def __gt__(self, other):
                # [FUNCTION DOC] HELPER: special method __gt__ of Proxy. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.x > other
            def __le__(self, other):
                # [FUNCTION DOC] HELPER: special method __le__ of Proxy. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.x <= other
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: special method __lt__ of Proxy. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.x < other
            def __str__(self):
                # [FUNCTION DOC] HELPER: special method __str__ of Proxy. Implements the protocol
                # hook used by tests; returns the hook result.
                return "Proxy:%s" % self.x
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of Proxy. Implements the protocol
                # hook used by tests; returns the hook result.
                return "Proxy(%r)" % self.x
            def __contains__(self, value):
                # [FUNCTION DOC] HELPER: special method __contains__ of Proxy. Arguments: value.
                # Implements the protocol hook used by tests; returns the hook result.
                return value in self.x
        p0 = Proxy(0)
        p1 = Proxy(1)
        p_1 = Proxy(-1)
        self.assertFalse(p0)
        self.assertFalse(not p1)
        self.assertEqual(hash(p0), hash(0))
        self.assertEqual(p0, p0)
        self.assertNotEqual(p0, p1)
        self.assertFalse(p0 != p0)
        self.assertEqual(not p0, p1)
        self.assertTrue(p0 < p1)
        self.assertTrue(p0 <= p1)
        self.assertTrue(p1 > p0)
        self.assertTrue(p1 >= p0)
        self.assertEqual(str(p0), "Proxy:0")
        self.assertEqual(repr(p0), "Proxy(0)")
        p10 = Proxy(range(10))
        self.assertNotIn(-1, p10)
        for i in range(10):
            self.assertIn(i, p10)
        self.assertNotIn(10, p10)

    def test_weakrefs(self):
        # [FUNCTION DOC] Test: checks weakrefs behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing weak references...
        import weakref
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_weakrefs to exercise
            # descriptor/class-model behavior under test.
            pass
        c = C()
        r = weakref.ref(c)
        self.assertEqual(r(), c)
        del c
        support.gc_collect()
        self.assertEqual(r(), None)
        del r
        class NoWeak(object):
            # [CLASS DOC] HELPER: local class NoWeak defined inside test_weakrefs to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['foo']
        no = NoWeak()
        try:
            weakref.ref(no)
        except TypeError as msg:
            self.assertIn("weak reference", str(msg))
        else:
            self.fail("weakref.ref(no) should be illegal")
        class Weak(object):
            # [CLASS DOC] HELPER: local class Weak defined inside test_weakrefs to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['foo', '__weakref__']
        yes = Weak()
        r = weakref.ref(yes)
        self.assertEqual(r(), yes)
        del yes
        support.gc_collect()
        self.assertEqual(r(), None)
        del r

    def test_properties(self):
        # [FUNCTION DOC] Test: checks properties behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing property...
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_properties to exercise
            # descriptor/class-model behavior under test.
            def getx(self):
                # [FUNCTION DOC] HELPER: getx of C. Supporting routine for tests; returns a
                # computed value or None.
                return self.__x
            def setx(self, value):
                # [FUNCTION DOC] HELPER: setx of C. Arguments: value. Supporting routine for
                # tests; returns a computed value or None.
                self.__x = value
            def delx(self):
                # [FUNCTION DOC] HELPER: delx of C. Supporting routine for tests; returns a
                # computed value or None.
                del self.__x
            x = property(getx, setx, delx, doc="I'm the x property.")
        a = C()
        self.assertNotHasAttr(a, "x")
        a.x = 42
        self.assertEqual(a._C__x, 42)
        self.assertEqual(a.x, 42)
        del a.x
        self.assertNotHasAttr(a, "x")
        self.assertNotHasAttr(a, "_C__x")
        C.x.__set__(a, 100)
        self.assertEqual(C.x.__get__(a), 100)
        C.x.__delete__(a)
        self.assertNotHasAttr(a, "x")

        raw = C.__dict__['x']
        self.assertIsInstance(raw, property)

        attrs = dir(raw)
        self.assertIn("__doc__", attrs)
        self.assertIn("fget", attrs)
        self.assertIn("fset", attrs)
        self.assertIn("fdel", attrs)

        self.assertEqual(raw.__doc__, "I'm the x property.")
        self.assertIs(raw.fget, C.__dict__['getx'])
        self.assertIs(raw.fset, C.__dict__['setx'])
        self.assertIs(raw.fdel, C.__dict__['delx'])

        for attr in "fget", "fset", "fdel":
            try:
                setattr(raw, attr, 42)
            except AttributeError as msg:
                if str(msg).find('readonly') < 0:
                    self.fail("when setting readonly attr %r on a property, "
                              "got unexpected AttributeError msg %r" % (attr, str(msg)))
            else:
                self.fail("expected AttributeError from trying to set readonly %r "
                          "attr on a property" % attr)

        raw.__doc__ = 42
        self.assertEqual(raw.__doc__, 42)

        class D(object):
            # [CLASS DOC] HELPER: local class D defined inside test_properties to exercise
            # descriptor/class-model behavior under test.
            __getitem__ = property(lambda s: 1/0)

        d = D()
        try:
            for i in d:
                str(i)
        except ZeroDivisionError:
            pass
        else:
            self.fail("expected ZeroDivisionError from bad property")

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_properties_doc_attrib(self):
        # [FUNCTION DOC] Test: checks properties doc attrib behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        class E(object):
            # [CLASS DOC] HELPER: local class E defined inside test_properties_doc_attrib to
            # exercise descriptor/class-model behavior under test.
            def getter(self):
                # [FUNCTION DOC] HELPER: getter of E. Supporting routine for tests; returns a
                # computed value or None. Docstring: getter method
                "getter method"
                return 0
            def setter(self_, value):
                # [FUNCTION DOC] HELPER: setter of E. Arguments: self_, value. Supporting routine
                # for tests; returns a computed value or None. Docstring: setter method
                "setter method"
                pass
            prop = property(getter)
            self.assertEqual(prop.__doc__, "getter method")
            prop2 = property(fset=setter)
            self.assertEqual(prop2.__doc__, None)

    @support.cpython_only
    def test_testcapi_no_segfault(self):
        # [FUNCTION DOC] Test: checks testcapi no segfault behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # this segfaulted in 2.5b2
        try:
            import _testcapi
        except ImportError:
            pass
        else:
            class X(object):
                # [CLASS DOC] HELPER: local class X defined inside test_testcapi_no_segfault to
                # exercise descriptor/class-model behavior under test.
                p = property(_testcapi.test_with_docstring)

    def test_properties_plus(self):
        # [FUNCTION DOC] Test: checks properties plus behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_properties_plus to exercise
            # descriptor/class-model behavior under test.
            foo = property(doc="hello")
            @foo.getter
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of C. Supporting routine for tests; returns a
                # computed value or None.
                return self._foo
            @foo.setter
            def foo(self, value):
                # [FUNCTION DOC] HELPER: foo of C. Arguments: value. Supporting routine for tests;
                # returns a computed value or None.
                self._foo = abs(value)
            @foo.deleter
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of C. Supporting routine for tests; returns a
                # computed value or None.
                del self._foo
        c = C()
        self.assertEqual(C.foo.__doc__, "hello")
        self.assertNotHasAttr(c, "foo")
        c.foo = -42
        self.assertHasAttr(c, '_foo')
        self.assertEqual(c._foo, 42)
        self.assertEqual(c.foo, 42)
        del c.foo
        self.assertNotHasAttr(c, '_foo')
        self.assertNotHasAttr(c, "foo")

        class D(C):
            @C.foo.deleter
            # [CLASS DOC] HELPER: local class D defined inside test_properties_plus to exercise
            # descriptor/class-model behavior under test.
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of D. Supporting routine for tests; returns a
                # computed value or None.
                try:
                    del self._foo
                except AttributeError:
                    pass
        d = D()
        d.foo = 24
        self.assertEqual(d.foo, 24)
        del d.foo
        del d.foo

        class E(object):
            @property
            # [CLASS DOC] HELPER: local class E defined inside test_properties_plus to exercise
            # descriptor/class-model behavior under test.
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of E. Supporting routine for tests; returns a
                # computed value or None.
                return self._foo
            @foo.setter
            def foo(self, value):
                # [FUNCTION DOC] HELPER: foo of E. Arguments: value. Supporting routine for tests;
                # returns a computed value or None.
                raise RuntimeError
            @foo.setter
            def foo(self, value):
                # [FUNCTION DOC] HELPER: foo of E. Arguments: value. Supporting routine for tests;
                # returns a computed value or None.
                self._foo = abs(value)
            @foo.deleter
            def foo(self, value=None):
                # [FUNCTION DOC] HELPER: foo of E. Arguments: value. Supporting routine for tests;
                # returns a computed value or None.
                del self._foo

        e = E()
        e.foo = -42
        self.assertEqual(e.foo, 42)
        del e.foo

        class F(E):
            @E.foo.deleter
            # [CLASS DOC] HELPER: local class F defined inside test_properties_plus to exercise
            # descriptor/class-model behavior under test.
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of F. Supporting routine for tests; returns a
                # computed value or None.
                del self._foo
            @foo.setter
            def foo(self, value):
                # [FUNCTION DOC] HELPER: foo of F. Arguments: value. Supporting routine for tests;
                # returns a computed value or None.
                self._foo = max(0, value)
        f = F()
        f.foo = -10
        self.assertEqual(f.foo, 0)
        del f.foo

    def test_dict_constructors(self):
        # [FUNCTION DOC] Test: checks dict constructors behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing dict constructor ...
        d = dict()
        self.assertEqual(d, {})
        d = dict({})
        self.assertEqual(d, {})
        d = dict({1: 2, 'a': 'b'})
        self.assertEqual(d, {1: 2, 'a': 'b'})
        self.assertEqual(d, dict(list(d.items())))
        self.assertEqual(d, dict(iter(d.items())))
        d = dict({'one':1, 'two':2})
        self.assertEqual(d, dict(one=1, two=2))
        self.assertEqual(d, dict(**d))
        self.assertEqual(d, dict({"one": 1}, two=2))
        self.assertEqual(d, dict([("two", 2)], one=1))
        self.assertEqual(d, dict([("one", 100), ("two", 200)], **d))
        self.assertEqual(d, dict(**d))

        for badarg in 0, 0, 0j, "0", [0], (0,):
            try:
                dict(badarg)
            except TypeError:
                pass
            except ValueError:
                if badarg == "0":
                    # It's a sequence, and its elements are also sequences (gotta
                    # love strings <wink>), but they aren't of length 2, so this
                    # one seemed better as a ValueError than a TypeError.
                    pass
                else:
                    self.fail("no TypeError from dict(%r)" % badarg)
            else:
                self.fail("no TypeError from dict(%r)" % badarg)

        with self.assertRaises(TypeError):
            dict({}, {})

        class Mapping:
            # [CLASS DOC] HELPER: local class Mapping defined inside test_dict_constructors to
            # exercise descriptor/class-model behavior under test.
            # Lacks a .keys() method; will be added later.
            dict = {1:2, 3:4, 'a':1j}

        try:
            dict(Mapping())
        except TypeError:
            pass
        else:
            self.fail("no TypeError from dict(incomplete mapping)")

        Mapping.keys = lambda self: list(self.dict.keys())
        Mapping.__getitem__ = lambda self, i: self.dict[i]
        d = dict(Mapping())
        self.assertEqual(d, Mapping.dict)

        # Init from sequence of iterable objects, each producing a 2-sequence.
        class AddressBookEntry:
            # [CLASS DOC] HELPER: local class AddressBookEntry defined inside
            # test_dict_constructors to exercise descriptor/class-model behavior under test.
            def __init__(self, first, last):
                # [FUNCTION DOC] HELPER: constructor. Arguments: first, last. Stores the given
                # values on the instance, and for test cases initializes the base CPythonTestCase.
                self.first = first
                self.last = last
            def __iter__(self):
                # [FUNCTION DOC] HELPER: special method __iter__ of AddressBookEntry. Implements
                # the protocol hook used by tests; returns the hook result.
                return iter([self.first, self.last])

        d = dict([AddressBookEntry('Tim', 'Warsaw'),
                  AddressBookEntry('Barry', 'Peters'),
                  AddressBookEntry('Tim', 'Peters'),
                  AddressBookEntry('Barry', 'Warsaw')])
        self.assertEqual(d, {'Barry': 'Warsaw', 'Tim': 'Peters'})

        d = dict(zip(range(4), range(1, 5)))
        self.assertEqual(d, dict([(i, i+1) for i in range(4)]))

        # Bad sequence lengths.
        for bad in [('tooshort',)], [('too', 'long', 'by 1')]:
            try:
                dict(bad)
            except ValueError:
                pass
            else:
                self.fail("no ValueError from dict(%r)" % bad)

    def test_dir(self):
        # [FUNCTION DOC] Test: checks dir behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing dir() ...
        junk = 12
        self.assertEqual(dir(), ['junk', 'self'])
        del junk

        # Just make sure these don't blow up!
        for arg in 2, 2, 2j, 2e0, [2], "2", b"2", (2,), {2:2}, type, self.test_dir:
            dir(arg)

        # Test dir on new-style classes.  Since these have object as a
        # base class, a lot more gets sucked in.
        def interesting(strings):
            # [FUNCTION DOC] HELPER: local function interesting inside test_dir. Arguments:
            # strings. Returns its computed value or None; used by the enclosing test.
            return [s for s in strings if not s.startswith('_')]

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_dir to exercise
            # descriptor/class-model behavior under test.
            Cdata = 1
            # [FUNCTION DOC] HELPER: Cmethod of C. Supporting routine for tests; returns a
            # computed value or None.
            def Cmethod(self): pass

        cstuff = ['Cdata', 'Cmethod']
        self.assertEqual(interesting(dir(C)), cstuff)

        c = C()
        self.assertEqual(interesting(dir(c)), cstuff)
        ## self.assertIn('__self__', dir(C.Cmethod))

        c.cdata = 2
        c.cmethod = lambda self: 0
        self.assertEqual(interesting(dir(c)), cstuff + ['cdata', 'cmethod'])
        ## self.assertIn('__self__', dir(c.Cmethod))

        class A(C):
            # [CLASS DOC] HELPER: local class A defined inside test_dir to exercise
            # descriptor/class-model behavior under test.
            Adata = 1
            # [FUNCTION DOC] HELPER: Amethod of A. Supporting routine for tests; returns a
            # computed value or None.
            def Amethod(self): pass

        astuff = ['Adata', 'Amethod'] + cstuff
        self.assertEqual(interesting(dir(A)), astuff)
        ## self.assertIn('__self__', dir(A.Amethod))
        a = A()
        self.assertEqual(interesting(dir(a)), astuff)
        a.adata = 42
        a.amethod = lambda self: 3
        self.assertEqual(interesting(dir(a)), astuff + ['adata', 'amethod'])
        ## self.assertIn('__self__', dir(a.Amethod))

        # Try a module subclass.
        class M(type(sys)):
            # [CLASS DOC] HELPER: local class M defined inside test_dir to exercise
            # descriptor/class-model behavior under test.
            pass
        minstance = M("m")
        minstance.b = 2
        minstance.a = 1
        default_attributes = ['__name__', '__doc__', '__package__',
                              '__loader__', '__spec__']
        names = [x for x in dir(minstance) if x not in default_attributes]
        self.assertEqual(names, ['a', 'b'])

        class M2(M):
            # [CLASS DOC] HELPER: local class M2 defined inside test_dir to exercise
            # descriptor/class-model behavior under test.
            def getdict(self):
                # [FUNCTION DOC] HELPER: getdict of M2. Supporting routine for tests; returns a
                # computed value or None.
                return "Not a dict!"
            __dict__ = property(getdict)

        m2instance = M2("m2")
        m2instance.b = 2
        m2instance.a = 1
        self.assertEqual(m2instance.__dict__, "Not a dict!")
        with self.assertRaises(TypeError):
            dir(m2instance)

        # Two essentially featureless objects, (Ellipsis just inherits stuff
        # from object.
        self.assertEqual(dir(object()), dir(Ellipsis))

        # Nasty test case for proxied objects
        class Wrapper(object):
            # [CLASS DOC] HELPER: local class Wrapper defined inside test_dir to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: constructor. Arguments: obj. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.__obj = obj
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of Wrapper. Implements the
                # protocol hook used by tests; returns the hook result.
                return "Wrapper(%s)" % repr(self.__obj)
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: special method __getitem__ of Wrapper. Arguments: key.
                # Implements the protocol hook used by tests; returns the hook result.
                return Wrapper(self.__obj[key])
            def __len__(self):
                # [FUNCTION DOC] HELPER: special method __len__ of Wrapper. Implements the
                # protocol hook used by tests; returns the hook result.
                return len(self.__obj)
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattr__ of Wrapper. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                return Wrapper(getattr(self.__obj, name))

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_dir to exercise
            # descriptor/class-model behavior under test.
            def __getclass(self):
                # [FUNCTION DOC] HELPER: __getclass of C. Supporting routine for tests; returns a
                # computed value or None.
                return Wrapper(type(self))
            __class__ = property(__getclass)

        dir(C()) # This used to segfault

    def test_supers(self):
        # [FUNCTION DOC] Test: checks supers behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing super...

        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def meth(self, a):
                # [FUNCTION DOC] HELPER: meth of A. Arguments: a. Supporting routine for tests;
                # returns a computed value or None.
                return "A(%r)" % a

        self.assertEqual(A().meth(1), "A(1)")

        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                self.__super = super(B, self)
            def meth(self, a):
                # [FUNCTION DOC] HELPER: meth of B. Arguments: a. Supporting routine for tests;
                # returns a computed value or None.
                return "B(%r)" % a + self.__super.meth(a)

        self.assertEqual(B().meth(2), "B(2)A(2)")

        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def meth(self, a):
                # [FUNCTION DOC] HELPER: meth of C. Arguments: a. Supporting routine for tests;
                # returns a computed value or None.
                return "C(%r)" % a + self.__super.meth(a)
        C._C__super = super(C)

        self.assertEqual(C().meth(3), "C(3)A(3)")

        class D(C, B):
            # [CLASS DOC] HELPER: local class D defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def meth(self, a):
                # [FUNCTION DOC] HELPER: meth of D. Arguments: a. Supporting routine for tests;
                # returns a computed value or None.
                return "D(%r)" % a + super(D, self).meth(a)

        self.assertEqual(D().meth(4), "D(4)C(4)B(4)A(4)")

        # Test for subclassing super

        class mysuper(super):
            # [CLASS DOC] HELPER: local class mysuper defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: constructor. Arguments: *args. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                return super(mysuper, self).__init__(*args)

        class E(D):
            # [CLASS DOC] HELPER: local class E defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def meth(self, a):
                # [FUNCTION DOC] HELPER: meth of E. Arguments: a. Supporting routine for tests;
                # returns a computed value or None.
                return "E(%r)" % a + mysuper(E, self).meth(a)

        self.assertEqual(E().meth(5), "E(5)D(5)C(5)B(5)A(5)")

        class F(E):
            # [CLASS DOC] HELPER: local class F defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def meth(self, a):
                # [FUNCTION DOC] HELPER: meth of F. Arguments: a. Supporting routine for tests;
                # returns a computed value or None.
                s = self.__super # == mysuper(F, self)
                return "F(%r)[%s]" % (a, s.__class__.__name__) + s.meth(a)
        F._F__super = mysuper(F)

        self.assertEqual(F().meth(6), "F(6)[mysuper]E(6)D(6)C(6)B(6)A(6)")

        # Make sure certain errors are raised

        try:
            super(D, 42)
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D, 42)")

        try:
            super(D, C())
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D, C())")

        try:
            super(D).__get__(12)
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D).__get__(12)")

        try:
            super(D).__get__(C())
        except TypeError:
            pass
        else:
            self.fail("shouldn't allow super(D).__get__(C())")

        # Make sure data descriptors can be overridden and accessed via super
        # (new feature in Python 2.3)

        class DDbase(object):
            # [FUNCTION DOC] HELPER: getx of DDbase. Supporting routine for tests; returns a
            # computed value or None.
            # [CLASS DOC] HELPER: local class DDbase defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def getx(self): return 42
            x = property(getx)

        class DDsub(DDbase):
            # [FUNCTION DOC] HELPER: getx of DDsub. Supporting routine for tests; returns a
            # computed value or None.
            # [CLASS DOC] HELPER: local class DDsub defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def getx(self): return "hello"
            x = property(getx)

        dd = DDsub()
        self.assertEqual(dd.x, "hello")
        self.assertEqual(super(DDsub, dd).x, 42)

        # Ensure that super() lookup of descriptor from classmethod
        # works (SF ID# 743627)

        class Base(object):
            # [CLASS DOC] HELPER: local class Base defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            aProp = property(lambda self: "foo")

        class Sub(Base):
            @classmethod
            # [CLASS DOC] HELPER: local class Sub defined inside test_supers to exercise
            # descriptor/class-model behavior under test.
            def test(klass):
                # [FUNCTION DOC] Test: checks  behavior. Fails with AssertionError (via unittest
                # assertions) if the observed behavior differs from CPython semantics; no other
                # side effects beyond temporary objects and classes.
                return super(Sub,klass).aProp

        self.assertEqual(Sub.test(), Base.aProp)

        # Verify that super() doesn't allow keyword args
        with self.assertRaises(TypeError):
            super(Base, kw=1)

    def test_basic_inheritance(self):
        # [FUNCTION DOC] Test: checks basic inheritance behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing inheritance from basic types...

        class hexint(int):
            # [CLASS DOC] HELPER: local class hexint defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of hexint. Implements the
                # protocol hook used by tests; returns the hook result.
                return hex(self)
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: special method __add__ of hexint. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return hexint(int.__add__(self, other))
            # (Note that overriding __radd__ doesn't work,
            # because the int type gets first dibs.)
        self.assertEqual(repr(hexint(7) + 9), "0x10")
        self.assertEqual(repr(hexint(1000) + 7), "0x3ef")
        a = hexint(12345)
        self.assertEqual(a, 12345)
        self.assertEqual(int(a), 12345)
        self.assertIs(int(a).__class__, int)
        self.assertEqual(hash(a), hash(12345))
        self.assertIs((+a).__class__, int)
        self.assertIs((a >> 0).__class__, int)
        self.assertIs((a << 0).__class__, int)
        self.assertIs((hexint(0) << 12).__class__, int)
        self.assertIs((hexint(0) >> 12).__class__, int)

        class octlong(int):
            # [CLASS DOC] HELPER: local class octlong defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            __slots__ = []
            def __str__(self):
                # [FUNCTION DOC] HELPER: special method __str__ of octlong. Implements the
                # protocol hook used by tests; returns the hook result.
                return oct(self)
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: special method __add__ of octlong. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return self.__class__(super(octlong, self).__add__(other))
            __radd__ = __add__
        self.assertEqual(str(octlong(3) + 5), "0o10")
        # (Note that overriding __radd__ here only seems to work
        # because the example uses a short int left argument.)
        self.assertEqual(str(5 + octlong(3000)), "0o5675")
        a = octlong(12345)
        self.assertEqual(a, 12345)
        self.assertEqual(int(a), 12345)
        self.assertEqual(hash(a), hash(12345))
        self.assertIs(int(a).__class__, int)
        self.assertIs((+a).__class__, int)
        self.assertIs((-a).__class__, int)
        self.assertIs((-octlong(0)).__class__, int)
        self.assertIs((a >> 0).__class__, int)
        self.assertIs((a << 0).__class__, int)
        self.assertIs((a - 0).__class__, int)
        self.assertIs((a * 1).__class__, int)
        self.assertIs((a ** 1).__class__, int)
        self.assertIs((a // 1).__class__, int)
        self.assertIs((1 * a).__class__, int)
        self.assertIs((a | 0).__class__, int)
        self.assertIs((a ^ 0).__class__, int)
        self.assertIs((a & -1).__class__, int)
        self.assertIs((octlong(0) << 12).__class__, int)
        self.assertIs((octlong(0) >> 12).__class__, int)
        self.assertIs(abs(octlong(0)).__class__, int)

        # Because octlong overrides __add__, we can't check the absence of +0
        # optimizations using octlong.
        class longclone(int):
            # [CLASS DOC] HELPER: local class longclone defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        a = longclone(1)
        self.assertIs((a + 0).__class__, int)
        self.assertIs((0 + a).__class__, int)

        # Check that negative clones don't segfault
        a = longclone(-1)
        self.assertEqual(a.__dict__, {})
        self.assertEqual(int(a), -1)  # self.assertTrue PyNumber_Long() copies the sign bit

        class precfloat(float):
            # [CLASS DOC] HELPER: local class precfloat defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            __slots__ = ['prec']
            def __init__(self, value=0.0, prec=12):
                # [FUNCTION DOC] HELPER: constructor. Arguments: value, prec. Stores the given
                # values on the instance, and for test cases initializes the base CPythonTestCase.
                self.prec = int(prec)
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of precfloat. Implements the
                # protocol hook used by tests; returns the hook result.
                return "%.*g" % (self.prec, self)
        self.assertEqual(repr(precfloat(1.1)), "1.1")
        a = precfloat(12345)
        self.assertEqual(a, 12345.0)
        self.assertEqual(float(a), 12345.0)
        self.assertIs(float(a).__class__, float)
        self.assertEqual(hash(a), hash(12345.0))
        self.assertIs((+a).__class__, float)

        class madcomplex(complex):
            # [CLASS DOC] HELPER: local class madcomplex defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of madcomplex. Implements the
                # protocol hook used by tests; returns the hook result.
                return "%.17gj%+.17g" % (self.imag, self.real)
        a = madcomplex(-3, 4)
        self.assertEqual(repr(a), "4j-3")
        base = complex(-3, 4)
        self.assertEqual(base.__class__, complex)
        self.assertEqual(a, base)
        self.assertEqual(complex(a), base)
        self.assertEqual(complex(a).__class__, complex)
        a = madcomplex(a)  # just trying another form of the constructor
        self.assertEqual(repr(a), "4j-3")
        self.assertEqual(a, base)
        self.assertEqual(complex(a), base)
        self.assertEqual(complex(a).__class__, complex)
        self.assertEqual(hash(a), hash(base))
        self.assertEqual((+a).__class__, complex)
        self.assertEqual((a + 0).__class__, complex)
        self.assertEqual(a + 0, base)
        self.assertEqual((a - 0).__class__, complex)
        self.assertEqual(a - 0, base)
        self.assertEqual((a * 1).__class__, complex)
        self.assertEqual(a * 1, base)
        self.assertEqual((a / 1).__class__, complex)
        self.assertEqual(a / 1, base)

        class madtuple(tuple):
            # [CLASS DOC] HELPER: local class madtuple defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            _rev = None
            def rev(self):
                # [FUNCTION DOC] HELPER: rev of madtuple. Supporting routine for tests; returns a
                # computed value or None.
                if self._rev is not None:
                    return self._rev
                L = list(self)
                L.reverse()
                self._rev = self.__class__(L)
                return self._rev
        a = madtuple((1,2,3,4,5,6,7,8,9,0))
        self.assertEqual(a, (1,2,3,4,5,6,7,8,9,0))
        self.assertEqual(a.rev(), madtuple((0,9,8,7,6,5,4,3,2,1)))
        self.assertEqual(a.rev().rev(), madtuple((1,2,3,4,5,6,7,8,9,0)))
        for i in range(512):
            t = madtuple(range(i))
            u = t.rev()
            v = u.rev()
            self.assertEqual(v, t)
        a = madtuple((1,2,3,4,5))
        self.assertEqual(tuple(a), (1,2,3,4,5))
        self.assertIs(tuple(a).__class__, tuple)
        self.assertEqual(hash(a), hash((1,2,3,4,5)))
        self.assertIs(a[:].__class__, tuple)
        self.assertIs((a * 1).__class__, tuple)
        self.assertIs((a * 0).__class__, tuple)
        self.assertIs((a + ()).__class__, tuple)
        a = madtuple(())
        self.assertEqual(tuple(a), ())
        self.assertIs(tuple(a).__class__, tuple)
        self.assertIs((a + a).__class__, tuple)
        self.assertIs((a * 0).__class__, tuple)
        self.assertIs((a * 1).__class__, tuple)
        self.assertIs((a * 2).__class__, tuple)
        self.assertIs(a[:].__class__, tuple)

        class madstring(str):
            # [CLASS DOC] HELPER: local class madstring defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            _rev = None
            def rev(self):
                # [FUNCTION DOC] HELPER: rev of madstring. Supporting routine for tests; returns a
                # computed value or None.
                if self._rev is not None:
                    return self._rev
                L = list(self)
                L.reverse()
                self._rev = self.__class__("".join(L))
                return self._rev
        s = madstring("abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(s, "abcdefghijklmnopqrstuvwxyz")
        self.assertEqual(s.rev(), madstring("zyxwvutsrqponmlkjihgfedcba"))
        self.assertEqual(s.rev().rev(), madstring("abcdefghijklmnopqrstuvwxyz"))
        for i in range(256):
            s = madstring("".join(map(chr, range(i))))
            t = s.rev()
            u = t.rev()
            self.assertEqual(u, s)
        s = madstring("12345")
        self.assertEqual(str(s), "12345")
        self.assertIs(str(s).__class__, str)

        base = "\x00" * 5
        s = madstring(base)
        self.assertEqual(s, base)
        self.assertEqual(str(s), base)
        self.assertIs(str(s).__class__, str)
        self.assertEqual(hash(s), hash(base))
        self.assertEqual({s: 1}[base], 1)
        self.assertEqual({base: 1}[s], 1)
        self.assertIs((s + "").__class__, str)
        self.assertEqual(s + "", base)
        self.assertIs(("" + s).__class__, str)
        self.assertEqual("" + s, base)
        self.assertIs((s * 0).__class__, str)
        self.assertEqual(s * 0, "")
        self.assertIs((s * 1).__class__, str)
        self.assertEqual(s * 1, base)
        self.assertIs((s * 2).__class__, str)
        self.assertEqual(s * 2, base + base)
        self.assertIs(s[:].__class__, str)
        self.assertEqual(s[:], base)
        self.assertIs(s[0:0].__class__, str)
        self.assertEqual(s[0:0], "")
        self.assertIs(s.strip().__class__, str)
        self.assertEqual(s.strip(), base)
        self.assertIs(s.lstrip().__class__, str)
        self.assertEqual(s.lstrip(), base)
        self.assertIs(s.rstrip().__class__, str)
        self.assertEqual(s.rstrip(), base)
        identitytab = {}
        self.assertIs(s.translate(identitytab).__class__, str)
        self.assertEqual(s.translate(identitytab), base)
        self.assertIs(s.replace("x", "x").__class__, str)
        self.assertEqual(s.replace("x", "x"), base)
        self.assertIs(s.ljust(len(s)).__class__, str)
        self.assertEqual(s.ljust(len(s)), base)
        self.assertIs(s.rjust(len(s)).__class__, str)
        self.assertEqual(s.rjust(len(s)), base)
        self.assertIs(s.center(len(s)).__class__, str)
        self.assertEqual(s.center(len(s)), base)
        self.assertIs(s.lower().__class__, str)
        self.assertEqual(s.lower(), base)

        class madunicode(str):
            # [CLASS DOC] HELPER: local class madunicode defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            _rev = None
            def rev(self):
                # [FUNCTION DOC] HELPER: rev of madunicode. Supporting routine for tests; returns
                # a computed value or None.
                if self._rev is not None:
                    return self._rev
                L = list(self)
                L.reverse()
                self._rev = self.__class__("".join(L))
                return self._rev
        u = madunicode("ABCDEF")
        self.assertEqual(u, "ABCDEF")
        self.assertEqual(u.rev(), madunicode("FEDCBA"))
        self.assertEqual(u.rev().rev(), madunicode("ABCDEF"))
        base = "12345"
        u = madunicode(base)
        self.assertEqual(str(u), base)
        self.assertIs(str(u).__class__, str)
        self.assertEqual(hash(u), hash(base))
        self.assertEqual({u: 1}[base], 1)
        self.assertEqual({base: 1}[u], 1)
        self.assertIs(u.strip().__class__, str)
        self.assertEqual(u.strip(), base)
        self.assertIs(u.lstrip().__class__, str)
        self.assertEqual(u.lstrip(), base)
        self.assertIs(u.rstrip().__class__, str)
        self.assertEqual(u.rstrip(), base)
        self.assertIs(u.replace("x", "x").__class__, str)
        self.assertEqual(u.replace("x", "x"), base)
        self.assertIs(u.replace("xy", "xy").__class__, str)
        self.assertEqual(u.replace("xy", "xy"), base)
        self.assertIs(u.center(len(u)).__class__, str)
        self.assertEqual(u.center(len(u)), base)
        self.assertIs(u.ljust(len(u)).__class__, str)
        self.assertEqual(u.ljust(len(u)), base)
        self.assertIs(u.rjust(len(u)).__class__, str)
        self.assertEqual(u.rjust(len(u)), base)
        self.assertIs(u.lower().__class__, str)
        self.assertEqual(u.lower(), base)
        self.assertIs(u.upper().__class__, str)
        self.assertEqual(u.upper(), base)
        self.assertIs(u.capitalize().__class__, str)
        self.assertEqual(u.capitalize(), base)
        self.assertIs(u.title().__class__, str)
        self.assertEqual(u.title(), base)
        self.assertIs((u + "").__class__, str)
        self.assertEqual(u + "", base)
        self.assertIs(("" + u).__class__, str)
        self.assertEqual("" + u, base)
        self.assertIs((u * 0).__class__, str)
        self.assertEqual(u * 0, "")
        self.assertIs((u * 1).__class__, str)
        self.assertEqual(u * 1, base)
        self.assertIs((u * 2).__class__, str)
        self.assertEqual(u * 2, base + base)
        self.assertIs(u[:].__class__, str)
        self.assertEqual(u[:], base)
        self.assertIs(u[0:0].__class__, str)
        self.assertEqual(u[0:0], "")

        class sublist(list):
            # [CLASS DOC] HELPER: local class sublist defined inside test_basic_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        a = sublist(range(5))
        self.assertEqual(a, list(range(5)))
        a.append("hello")
        self.assertEqual(a, list(range(5)) + ["hello"])
        a[5] = 5
        self.assertEqual(a, list(range(6)))
        a.extend(range(6, 20))
        self.assertEqual(a, list(range(20)))
        a[-5:] = []
        self.assertEqual(a, list(range(15)))
        del a[10:15]
        self.assertEqual(len(a), 10)
        self.assertEqual(a, list(range(10)))
        self.assertEqual(list(a), list(range(10)))
        self.assertEqual(a[0], 0)
        self.assertEqual(a[9], 9)
        self.assertEqual(a[-10], 0)
        self.assertEqual(a[-1], 9)
        self.assertEqual(a[:5], list(range(5)))

        ## class CountedInput(file):
        ##    """Counts lines read by self.readline().
        ##
        ##     self.lineno is the 0-based ordinal of the last line read, up to
        ##     a maximum of one greater than the number of lines in the file.
        ##
        ##     self.ateof is true if and only if the final "" line has been read,
        ##     at which point self.lineno stops incrementing, and further calls
        ##     to readline() continue to return "".
        ##     """
        ##
        ##     lineno = 0
        ##     ateof = 0
        ##     def readline(self):
        ##         if self.ateof:
        ##             return ""
        ##         s = file.readline(self)
        ##         # Next line works too.
        ##         # s = super(CountedInput, self).readline()
        ##         self.lineno += 1
        ##         if s == "":
        ##             self.ateof = 1
        ##        return s
        ##
        ## f = file(name=os_helper.TESTFN, mode='w')
        ## lines = ['a\n', 'b\n', 'c\n']
        ## try:
        ##     f.writelines(lines)
        ##     f.close()
        ##     f = CountedInput(os_helper.TESTFN)
        ##     for (i, expected) in zip(range(1, 5) + [4], lines + 2 * [""]):
        ##         got = f.readline()
        ##         self.assertEqual(expected, got)
        ##         self.assertEqual(f.lineno, i)
        ##         self.assertEqual(f.ateof, (i > len(lines)))
        ##     f.close()
        ## finally:
        ##     try:
        ##         f.close()
        ##     except:
        ##         pass
        ##     os_helper.unlink(os_helper.TESTFN)

    def test_keywords(self):
        # [FUNCTION DOC] Test: checks keywords behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing keyword args to basic type constructors ...
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            int(x=1)
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            float(x=2)
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            bool(x=2)
        self.assertEqual(complex(imag=42, real=666), complex(666, 42))
        self.assertEqual(str(object=500), '500')
        self.assertEqual(str(object=b'abc', errors='strict'), 'abc')
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            tuple(sequence=range(3))
        with self.assertRaisesRegex(TypeError, 'keyword argument'):
            list(sequence=(0, 1, 2))
        # note: as of Python 2.3, dict() no longer has an "items" keyword arg

        for constructor in (int, float, int, complex, str, str,
                            tuple, list):
            try:
                constructor(bogus_keyword_arg=1)
            except TypeError:
                pass
            else:
                self.fail("expected TypeError from bogus keyword argument to %r"
                            % constructor)

    def test_str_subclass_as_dict_key(self):
        # [FUNCTION DOC] Test: checks str subclass as dict key behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing a str subclass used as dict key ..

        class cistr(str):
            # [CLASS DOC] HELPER: local class cistr defined inside test_str_subclass_as_dict_key
            # to exercise descriptor/class-model behavior under test. Existing docstring says:
            # Subclass of str that computes __eq__ case-insensitively.
            """Subclass of str that computes __eq__ case-insensitively.

            Also computes a hash code of the string in canonical form.
            """

            def __init__(self, value):
                # [FUNCTION DOC] HELPER: constructor. Arguments: value. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.canonical = value.lower()
                self.hashcode = hash(self.canonical)

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: special method __eq__ of cistr. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                if not isinstance(other, cistr):
                    other = cistr(other)
                return self.canonical == other.canonical

            def __hash__(self):
                # [FUNCTION DOC] HELPER: special method __hash__ of cistr. Implements the protocol
                # hook used by tests; returns the hook result.
                return self.hashcode

        self.assertEqual(cistr('ABC'), 'abc')
        self.assertEqual('aBc', cistr('ABC'))
        self.assertEqual(str(cistr('ABC')), 'ABC')

        d = {cistr('one'): 1, cistr('two'): 2, cistr('tHree'): 3}
        self.assertEqual(d[cistr('one')], 1)
        self.assertEqual(d[cistr('tWo')], 2)
        self.assertEqual(d[cistr('THrEE')], 3)
        self.assertIn(cistr('ONe'), d)
        self.assertEqual(d.get(cistr('thrEE')), 3)

    def test_classic_comparisons(self):
        # [FUNCTION DOC] Test: checks classic comparisons behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing classic comparisons...
        class classic:
            # [CLASS DOC] HELPER: local class classic defined inside test_classic_comparisons to
            # exercise descriptor/class-model behavior under test.
            pass

        for base in (classic, int, object):
            class C(base):
                # [CLASS DOC] HELPER: local class C defined inside test_classic_comparisons to
                # exercise descriptor/class-model behavior under test.
                def __init__(self, value):
                    # [FUNCTION DOC] HELPER: constructor. Arguments: value. Stores the given
                    # values on the instance, and for test cases initializes the base
                    # CPythonTestCase.
                    self.value = int(value)
                def __eq__(self, other):
                    # [FUNCTION DOC] HELPER: special method __eq__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value == other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value == other
                    return NotImplemented
                def __ne__(self, other):
                    # [FUNCTION DOC] HELPER: special method __ne__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value != other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value != other
                    return NotImplemented
                def __lt__(self, other):
                    # [FUNCTION DOC] HELPER: special method __lt__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value < other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value < other
                    return NotImplemented
                def __le__(self, other):
                    # [FUNCTION DOC] HELPER: special method __le__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value <= other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value <= other
                    return NotImplemented
                def __gt__(self, other):
                    # [FUNCTION DOC] HELPER: special method __gt__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value > other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value > other
                    return NotImplemented
                def __ge__(self, other):
                    # [FUNCTION DOC] HELPER: special method __ge__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value >= other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value >= other
                    return NotImplemented

            c1 = C(1)
            c2 = C(2)
            c3 = C(3)
            self.assertEqual(c1, 1)
            c = {1: c1, 2: c2, 3: c3}
            for x in 1, 2, 3:
                for y in 1, 2, 3:
                    for op in "<", "<=", "==", "!=", ">", ">=":
                        self.assertEqual(eval("c[x] %s c[y]" % op),
                                     eval("x %s y" % op),
                                     "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("c[x] %s y" % op),
                                     eval("x %s y" % op),
                                     "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("x %s c[y]" % op),
                                     eval("x %s y" % op),
                                     "x=%d, y=%d" % (x, y))

    def test_rich_comparisons(self):
        # [FUNCTION DOC] Test: checks rich comparisons behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing rich comparisons...
        class Z(complex):
            # [CLASS DOC] HELPER: local class Z defined inside test_rich_comparisons to exercise
            # descriptor/class-model behavior under test.
            pass
        z = Z(1)
        self.assertEqual(z, 1+0j)
        self.assertEqual(1+0j, z)
        class ZZ(complex):
            # [CLASS DOC] HELPER: local class ZZ defined inside test_rich_comparisons to exercise
            # descriptor/class-model behavior under test.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: special method __eq__ of ZZ. Arguments: other. Implements
                # the protocol hook used by tests; returns the hook result.
                try:
                    return abs(self - other) <= 1e-6
                except:
                    return NotImplemented
        zz = ZZ(1.0000003)
        self.assertEqual(zz, 1+0j)
        self.assertEqual(1+0j, zz)

        class classic:
            # [CLASS DOC] HELPER: local class classic defined inside test_rich_comparisons to
            # exercise descriptor/class-model behavior under test.
            pass
        for base in (classic, int, object, list):
            class C(base):
                # [CLASS DOC] HELPER: local class C defined inside test_rich_comparisons to
                # exercise descriptor/class-model behavior under test.
                def __init__(self, value):
                    # [FUNCTION DOC] HELPER: constructor. Arguments: value. Stores the given
                    # values on the instance, and for test cases initializes the base
                    # CPythonTestCase.
                    self.value = int(value)
                def __cmp__(self_, other):
                    # [FUNCTION DOC] HELPER: special method __cmp__ of C. Arguments: self_, other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    self.fail("shouldn't call __cmp__")
                def __eq__(self, other):
                    # [FUNCTION DOC] HELPER: special method __eq__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value == other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value == other
                    return NotImplemented
                def __ne__(self, other):
                    # [FUNCTION DOC] HELPER: special method __ne__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value != other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value != other
                    return NotImplemented
                def __lt__(self, other):
                    # [FUNCTION DOC] HELPER: special method __lt__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value < other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value < other
                    return NotImplemented
                def __le__(self, other):
                    # [FUNCTION DOC] HELPER: special method __le__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value <= other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value <= other
                    return NotImplemented
                def __gt__(self, other):
                    # [FUNCTION DOC] HELPER: special method __gt__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value > other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value > other
                    return NotImplemented
                def __ge__(self, other):
                    # [FUNCTION DOC] HELPER: special method __ge__ of C. Arguments: other.
                    # Implements the protocol hook used by tests; returns the hook result.
                    if isinstance(other, C):
                        return self.value >= other.value
                    if isinstance(other, int) or isinstance(other, int):
                        return self.value >= other
                    return NotImplemented
            c1 = C(1)
            c2 = C(2)
            c3 = C(3)
            self.assertEqual(c1, 1)
            c = {1: c1, 2: c2, 3: c3}
            for x in 1, 2, 3:
                for y in 1, 2, 3:
                    for op in "<", "<=", "==", "!=", ">", ">=":
                        self.assertEqual(eval("c[x] %s c[y]" % op),
                                         eval("x %s y" % op),
                                         "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("c[x] %s y" % op),
                                         eval("x %s y" % op),
                                         "x=%d, y=%d" % (x, y))
                        self.assertEqual(eval("x %s c[y]" % op),
                                         eval("x %s y" % op),
                                         "x=%d, y=%d" % (x, y))

    def test_descrdoc(self):
        # [FUNCTION DOC] Test: checks descrdoc behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing descriptor doc strings...
        from _io import FileIO
        def check(descr, what):
            # [FUNCTION DOC] HELPER: local function check inside test_descrdoc. Arguments: descr,
            # what. Returns its computed value or None; used by the enclosing test.
            self.assertEqual(descr.__doc__, what)
        check(FileIO.closed, "True if the file is closed") # getset descriptor
        check(complex.real, "the real part of a complex number") # member descriptor

    def test_doc_descriptor(self):
        # [FUNCTION DOC] Test: checks doc descriptor behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing __doc__ descriptor...
        # SF bug 542984
        class DocDescr(object):
            # [CLASS DOC] HELPER: local class DocDescr defined inside test_doc_descriptor to
            # exercise descriptor/class-model behavior under test.
            def __get__(self, object, otype):
                # [FUNCTION DOC] HELPER: special method __get__ of DocDescr. Arguments: object,
                # otype. Implements the protocol hook used by tests; returns the hook result.
                if object:
                    object = object.__class__.__name__ + ' instance'
                if otype:
                    otype = otype.__name__
                return 'object=%s; type=%s' % (object, otype)
        class NewClass:
            # [CLASS DOC] HELPER: local class NewClass defined inside test_doc_descriptor to
            # exercise descriptor/class-model behavior under test.
            __doc__ = DocDescr()
        self.assertEqual(NewClass.__doc__, 'object=None; type=NewClass')
        self.assertEqual(NewClass().__doc__, 'object=NewClass instance; type=NewClass')

    def test_set_class(self):
        # [FUNCTION DOC] Test: checks set class behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing __class__ assignment...
        # [CLASS DOC] HELPER: local class C defined inside test_set_class to exercise
        # descriptor/class-model behavior under test.
        class C(object): pass
        # [CLASS DOC] HELPER: local class D defined inside test_set_class to exercise
        # descriptor/class-model behavior under test.
        class D(object): pass
        # [CLASS DOC] HELPER: local class E defined inside test_set_class to exercise
        # descriptor/class-model behavior under test.
        class E(object): pass
        # [CLASS DOC] HELPER: local class F defined inside test_set_class to exercise
        # descriptor/class-model behavior under test.
        class F(D, E): pass
        for cls in C, D, E, F:
            for cls2 in C, D, E, F:
                x = cls()
                x.__class__ = cls2
                self.assertIs(x.__class__, cls2)
                x.__class__ = cls
                self.assertIs(x.__class__, cls)
        def cant(x, C):
            # [FUNCTION DOC] HELPER: local function cant inside test_set_class. Arguments: x, C.
            # Returns its computed value or None; used by the enclosing test.
            try:
                x.__class__ = C
            except TypeError:
                pass
            else:
                self.fail("shouldn't allow %r.__class__ = %r" % (x, C))
            try:
                delattr(x, "__class__")
            except (TypeError, AttributeError):
                pass
            else:
                self.fail("shouldn't allow del %r.__class__" % x)
        cant(C(), list)
        cant(list(), C)
        cant(C(), 1)
        cant(C(), object)
        cant(object(), list)
        cant(list(), object)
        # [CLASS DOC] HELPER: local class Int defined inside test_set_class to exercise
        # descriptor/class-model behavior under test.
        class Int(int): __slots__ = []
        cant(True, int)
        cant(2, bool)
        o = object()
        cant(o, int)
        cant(o, type(None))
        del o
        class G(object):
            # [CLASS DOC] HELPER: local class G defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["a", "b"]
        class H(object):
            # [CLASS DOC] HELPER: local class H defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["b", "a"]
        class I(object):
            # [CLASS DOC] HELPER: local class I defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["a", "b"]
        class J(object):
            # [CLASS DOC] HELPER: local class J defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["c", "b"]
        class K(object):
            # [CLASS DOC] HELPER: local class K defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["a", "b", "d"]
        class L(H):
            # [CLASS DOC] HELPER: local class L defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["e"]
        class M(I):
            # [CLASS DOC] HELPER: local class M defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["e"]
        class N(J):
            # [CLASS DOC] HELPER: local class N defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__weakref__"]
        class P(J):
            # [CLASS DOC] HELPER: local class P defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__dict__"]
        class Q(J):
            # [CLASS DOC] HELPER: local class Q defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            pass
        class R(J):
            # [CLASS DOC] HELPER: local class R defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ["__dict__", "__weakref__"]

        for cls, cls2 in ((G, H), (G, I), (I, H), (Q, R), (R, Q)):
            x = cls()
            x.a = 1
            x.__class__ = cls2
            self.assertIs(x.__class__, cls2,
                   "assigning %r as __class__ for %r silently failed" % (cls2, x))
            self.assertEqual(x.a, 1)
            x.__class__ = cls
            self.assertIs(x.__class__, cls,
                   "assigning %r as __class__ for %r silently failed" % (cls, x))
            self.assertEqual(x.a, 1)
        for cls in G, J, K, L, M, N, P, R, list, Int:
            for cls2 in G, J, K, L, M, N, P, R, list, Int:
                if cls is cls2:
                    continue
                cant(cls(), cls2)

        # Issue5283: when __class__ changes in __del__, the wrong
        # type gets DECREF'd.
        class O(object):
            # [CLASS DOC] HELPER: local class O defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            pass
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_set_class to exercise
            # descriptor/class-model behavior under test.
            def __del__(self):
                # [FUNCTION DOC] HELPER: special method __del__ of A. Implements the protocol hook
                # used by tests; returns the hook result.
                self.__class__ = O
        l = [A() for x in range(100)]
        del l

    def test_set_dict(self):
        # [FUNCTION DOC] Test: checks set dict behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing __dict__ assignment...
        # [CLASS DOC] HELPER: local class C defined inside test_set_dict to exercise
        # descriptor/class-model behavior under test.
        class C(object): pass
        a = C()
        a.__dict__ = {'b': 1}
        self.assertEqual(a.b, 1)
        def cant(x, dict):
            # [FUNCTION DOC] HELPER: local function cant inside test_set_dict. Arguments: x, dict.
            # Returns its computed value or None; used by the enclosing test.
            try:
                x.__dict__ = dict
            except (AttributeError, TypeError):
                pass
            else:
                self.fail("shouldn't allow %r.__dict__ = %r" % (x, dict))
        cant(a, None)
        cant(a, [])
        cant(a, 1)
        del a.__dict__ # Deleting __dict__ is allowed

        class Base(object):
            # [CLASS DOC] HELPER: local class Base defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        def verify_dict_readonly(x):
            # [FUNCTION DOC] HELPER: local function verify_dict_readonly inside test_set_dict.
            # Arguments: x. Returns its computed value or None; used by the enclosing test.
            """
            x has to be an instance of a class inheriting from Base.
            """
            cant(x, {})
            try:
                del x.__dict__
            except (AttributeError, TypeError):
                pass
            else:
                self.fail("shouldn't allow del %r.__dict__" % x)
            dict_descr = Base.__dict__["__dict__"]
            try:
                dict_descr.__set__(x, {})
            except (AttributeError, TypeError):
                pass
            else:
                self.fail("dict_descr allowed access to %r's dict" % x)

        # Classes don't allow __dict__ assignment and have readonly dicts
        class Meta1(type, Base):
            # [CLASS DOC] HELPER: local class Meta1 defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        class Meta2(Base, type):
            # [CLASS DOC] HELPER: local class Meta2 defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        class D(object, metaclass=Meta1):
            # [CLASS DOC] HELPER: local class D defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        class E(object, metaclass=Meta2):
            # [CLASS DOC] HELPER: local class E defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        for cls in C, D, E:
            verify_dict_readonly(cls)
            class_dict = cls.__dict__
            try:
                class_dict["spam"] = "eggs"
            except TypeError:
                pass
            else:
                self.fail("%r's __dict__ can be modified" % cls)

        # Modules also disallow __dict__ assignment
        class Module1(types.ModuleType, Base):
            # [CLASS DOC] HELPER: local class Module1 defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        class Module2(Base, types.ModuleType):
            # [CLASS DOC] HELPER: local class Module2 defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        for ModuleType in Module1, Module2:
            mod = ModuleType("spam")
            verify_dict_readonly(mod)
            mod.__dict__["spam"] = "eggs"

        # Exception's __dict__ can be replaced, but not deleted
        # (at least not any more than regular exception's __dict__ can
        # be deleted; on CPython it is not the case, whereas on PyPy they
        # can, just like any other new-style instance's __dict__.)
        def can_delete_dict(e):
            # [FUNCTION DOC] HELPER: local function can_delete_dict inside test_set_dict.
            # Arguments: e. Returns its computed value or None; used by the enclosing test.
            try:
                del e.__dict__
            except (TypeError, AttributeError):
                return False
            else:
                return True
        class Exception1(Exception, Base):
            # [CLASS DOC] HELPER: local class Exception1 defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        class Exception2(Base, Exception):
            # [CLASS DOC] HELPER: local class Exception2 defined inside test_set_dict to exercise
            # descriptor/class-model behavior under test.
            pass
        for ExceptionType in Exception, Exception1, Exception2:
            e = ExceptionType()
            e.__dict__ = {"a": 1}
            self.assertEqual(e.a, 1)
            self.assertEqual(can_delete_dict(e), can_delete_dict(ValueError()))

    def test_binary_operator_override(self):
        # [FUNCTION DOC] Test: checks binary operator override behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing overrides of binary operations...
        class I(int):
            # [CLASS DOC] HELPER: local class I defined inside test_binary_operator_override to
            # exercise descriptor/class-model behavior under test.
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of I. Implements the protocol
                # hook used by tests; returns the hook result.
                return "I(%r)" % int(self)
            def __add__(self, other):
                # [FUNCTION DOC] HELPER: special method __add__ of I. Arguments: other. Implements
                # the protocol hook used by tests; returns the hook result.
                return I(int(self) + int(other))
            __radd__ = __add__
            def __pow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: special method __pow__ of I. Arguments: other, mod.
                # Implements the protocol hook used by tests; returns the hook result.
                if mod is None:
                    return I(pow(int(self), int(other)))
                else:
                    return I(pow(int(self), int(other), int(mod)))
            def __rpow__(self, other, mod=None):
                # [FUNCTION DOC] HELPER: special method __rpow__ of I. Arguments: other, mod.
                # Implements the protocol hook used by tests; returns the hook result.
                if mod is None:
                    return I(pow(int(other), int(self), mod))
                else:
                    return I(pow(int(other), int(self), int(mod)))

        self.assertEqual(repr(I(1) + I(2)), "I(3)")
        self.assertEqual(repr(I(1) + 2), "I(3)")
        self.assertEqual(repr(1 + I(2)), "I(3)")
        self.assertEqual(repr(I(2) ** I(3)), "I(8)")
        self.assertEqual(repr(2 ** I(3)), "I(8)")
        self.assertEqual(repr(I(2) ** 3), "I(8)")
        self.assertEqual(repr(pow(I(2), I(3), I(5))), "I(3)")
        class S(str):
            # [CLASS DOC] HELPER: local class S defined inside test_binary_operator_override to
            # exercise descriptor/class-model behavior under test.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: special method __eq__ of S. Arguments: other. Implements
                # the protocol hook used by tests; returns the hook result.
                return self.lower() == other.lower()

    def test_subclass_propagation(self):
        # [FUNCTION DOC] Test: checks subclass propagation behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing propagation of slot functions to subclasses...
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_subclass_propagation to
            # exercise descriptor/class-model behavior under test.
            pass
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_subclass_propagation to
            # exercise descriptor/class-model behavior under test.
            pass
        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside test_subclass_propagation to
            # exercise descriptor/class-model behavior under test.
            pass
        class D(B, C):
            # [CLASS DOC] HELPER: local class D defined inside test_subclass_propagation to
            # exercise descriptor/class-model behavior under test.
            pass
        d = D()
        orig_hash = hash(d) # related to id(d) in platform-dependent ways
        A.__hash__ = lambda self: 42
        self.assertEqual(hash(d), 42)
        C.__hash__ = lambda self: 314
        self.assertEqual(hash(d), 314)
        B.__hash__ = lambda self: 144
        self.assertEqual(hash(d), 144)
        D.__hash__ = lambda self: 100
        self.assertEqual(hash(d), 100)
        D.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del D.__hash__
        self.assertEqual(hash(d), 144)
        B.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del B.__hash__
        self.assertEqual(hash(d), 314)
        C.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del C.__hash__
        self.assertEqual(hash(d), 42)
        A.__hash__ = None
        self.assertRaises(TypeError, hash, d)
        del A.__hash__
        self.assertEqual(hash(d), orig_hash)
        d.foo = 42
        d.bar = 42
        self.assertEqual(d.foo, 42)
        self.assertEqual(d.bar, 42)
        def __getattribute__(self, name):
            # [FUNCTION DOC] HELPER: local function __getattribute__ inside
            # test_subclass_propagation. Arguments: name. Returns its computed value or None; used
            # by the enclosing test.
            if name == "foo":
                return 24
            return object.__getattribute__(self, name)
        A.__getattribute__ = __getattribute__
        self.assertEqual(d.foo, 24)
        self.assertEqual(d.bar, 42)
        def __getattr__(self, name):
            # [FUNCTION DOC] HELPER: local function __getattr__ inside test_subclass_propagation.
            # Arguments: name. Returns its computed value or None; used by the enclosing test.
            if name in ("spam", "foo", "bar"):
                return "hello"
            raise AttributeError(name)
        B.__getattr__ = __getattr__
        self.assertEqual(d.spam, "hello")
        self.assertEqual(d.foo, 24)
        self.assertEqual(d.bar, 42)
        del A.__getattribute__
        self.assertEqual(d.foo, 42)
        del d.foo
        self.assertEqual(d.foo, "hello")
        self.assertEqual(d.bar, 42)
        del B.__getattr__
        try:
            d.foo
        except AttributeError:
            pass
        else:
            self.fail("d.foo should be undefined now")

        # Test a nasty bug in recurse_down_subclasses()
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_subclass_propagation to
            # exercise descriptor/class-model behavior under test.
            pass
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_subclass_propagation to
            # exercise descriptor/class-model behavior under test.
            pass
        del B
        support.gc_collect()
        A.__setitem__ = lambda *a: None # crash

    def test_buffer_inheritance(self):
        # [FUNCTION DOC] Test: checks buffer inheritance behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing that buffer interface is inherited ...

        import binascii
        # SF bug [#470040] ParseTuple t# vs subclasses.

        class MyBytes(bytes):
            # [CLASS DOC] HELPER: local class MyBytes defined inside test_buffer_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        base = b'abc'
        m = MyBytes(base)
        # b2a_hex uses the buffer interface to get its argument's value, via
        # PyArg_ParseTuple 't#' code.
        self.assertEqual(binascii.b2a_hex(m), binascii.b2a_hex(base))

        class MyInt(int):
            # [CLASS DOC] HELPER: local class MyInt defined inside test_buffer_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        m = MyInt(42)
        try:
            binascii.b2a_hex(m)
            self.fail('subclass of int should not have a buffer interface')
        except TypeError:
            pass

    def test_str_of_str_subclass(self):
        # [FUNCTION DOC] Test: checks str of str subclass behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing __str__ defined in subclass of str ...
        import binascii

        class octetstring(str):
            # [CLASS DOC] HELPER: local class octetstring defined inside test_str_of_str_subclass
            # to exercise descriptor/class-model behavior under test.
            def __str__(self):
                # [FUNCTION DOC] HELPER: special method __str__ of octetstring. Implements the
                # protocol hook used by tests; returns the hook result.
                return binascii.b2a_hex(self.encode('ascii')).decode("ascii")
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of octetstring. Implements the
                # protocol hook used by tests; returns the hook result.
                return self + " repr"

        o = octetstring('A')
        self.assertEqual(type(o), octetstring)
        self.assertEqual(type(str(o)), str)
        self.assertEqual(type(repr(o)), str)
        self.assertEqual(ord(o), 0x41)
        self.assertEqual(str(o), '41')
        self.assertEqual(repr(o), 'A repr')
        self.assertEqual(o.__str__(), '41')
        self.assertEqual(o.__repr__(), 'A repr')

    def test_repr_with_module_str_subclass(self):
        # [FUNCTION DOC] Test: checks repr with module str subclass behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # gh-98783
        class StrSub(str):
            # [CLASS DOC] HELPER: local class StrSub defined inside
            # test_repr_with_module_str_subclass to exercise descriptor/class-model behavior under
            # test.
            pass
        class Some:
            # [CLASS DOC] HELPER: local class Some defined inside
            # test_repr_with_module_str_subclass to exercise descriptor/class-model behavior under
            # test.
            pass
        Some.__module__ = StrSub('example')
        self.assertIsInstance(repr(Some), str)  # should not crash
        self.assertIsInstance(repr(Some()), str)  # should not crash

    def test_keyword_arguments(self):
        # [FUNCTION DOC] Test: checks keyword arguments behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing keyword arguments to __init__, __call__...
        # [FUNCTION DOC] HELPER: local function f inside test_keyword_arguments. Arguments: a.
        # Returns its computed value or None; used by the enclosing test.
        def f(a): return a
        self.assertEqual(f.__call__(a=42), 42)
        ba = bytearray()
        bytearray.__init__(ba, 'abc\xbd\u20ac',
                           encoding='latin1', errors='replace')
        self.assertEqual(ba, b'abc\xbd?')

    def test_recursive_call(self):
        # [FUNCTION DOC] Test: checks recursive call behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing recursive __call__() by setting to instance of class...
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_recursive_call to exercise
            # descriptor/class-model behavior under test.
            pass

        A.__call__ = A()
        with self.assertRaises(RecursionError):
            A()()

    def test_delete_hook(self):
        # [FUNCTION DOC] Test: checks delete hook behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing __del__ hook...
        log = []
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_delete_hook to exercise
            # descriptor/class-model behavior under test.
            def __del__(self):
                # [FUNCTION DOC] HELPER: special method __del__ of C. Implements the protocol hook
                # used by tests; returns the hook result.
                log.append(1)
        c = C()
        self.assertEqual(log, [])
        del c
        support.gc_collect()
        self.assertEqual(log, [1])

        # [CLASS DOC] HELPER: local class D defined inside test_delete_hook to exercise
        # descriptor/class-model behavior under test.
        class D(object): pass
        d = D()
        try: del d[0]
        except TypeError: pass
        else: self.fail("invalid del() didn't raise TypeError")

    def test_hash_inheritance(self):
        # [FUNCTION DOC] Test: checks hash inheritance behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing hash of mutable subclasses...

        class mydict(dict):
            # [CLASS DOC] HELPER: local class mydict defined inside test_hash_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        d = mydict()
        try:
            hash(d)
        except TypeError:
            pass
        else:
            self.fail("hash() of dict subclass should fail")

        class mylist(list):
            # [CLASS DOC] HELPER: local class mylist defined inside test_hash_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        d = mylist()
        try:
            hash(d)
        except TypeError:
            pass
        else:
            self.fail("hash() of list subclass should fail")

    def test_str_operations(self):
        # [FUNCTION DOC] Test: checks str operations behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        try: 'a' + 5
        except TypeError: pass
        else: self.fail("'' + 5 doesn't raise TypeError")

        try: ''.split('')
        except ValueError: pass
        else: self.fail("''.split('') doesn't raise ValueError")

        try: ''.join([0])
        except TypeError: pass
        else: self.fail("''.join([0]) doesn't raise TypeError")

        try: ''.rindex('5')
        except ValueError: pass
        else: self.fail("''.rindex('5') doesn't raise ValueError")

        try: '%(n)s' % None
        except TypeError: pass
        else: self.fail("'%(n)s' % None doesn't raise TypeError")

        try: '%(n' % {}
        except ValueError: pass
        else: self.fail("'%(n' % {} '' doesn't raise ValueError")

        try: '%*s' % ('abc')
        except TypeError: pass
        else: self.fail("'%*s' % ('abc') doesn't raise TypeError")

        try: '%*.*s' % ('abc', 5)
        except TypeError: pass
        else: self.fail("'%*.*s' % ('abc', 5) doesn't raise TypeError")

        try: '%s' % (1, 2)
        except TypeError: pass
        else: self.fail("'%s' % (1, 2) doesn't raise TypeError")

        try: '%' % None
        except ValueError: pass
        else: self.fail("'%' % None doesn't raise ValueError")

        self.assertEqual('534253'.isdigit(), 1)
        self.assertEqual('534253x'.isdigit(), 0)
        self.assertEqual('%c' % 5, '\x05')
        self.assertEqual('%c' % '5', '5')

    def test_deepcopy_recursive(self):
        # [FUNCTION DOC] Test: checks deepcopy recursive behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing deepcopy of recursive objects...
        class Node:
            # [CLASS DOC] HELPER: local class Node defined inside test_deepcopy_recursive to
            # exercise descriptor/class-model behavior under test.
            pass
        a = Node()
        b = Node()
        a.b = b
        b.a = a
        z = deepcopy(a) # This blew up before

    def test_uninitialized_modules(self):
        # [FUNCTION DOC] Test: checks uninitialized modules behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing uninitialized module objects...
        from types import ModuleType as M
        m = M.__new__(M)
        str(m)
        self.assertNotHasAttr(m, "__name__")
        self.assertNotHasAttr(m, "__file__")
        self.assertNotHasAttr(m, "foo")
        self.assertFalse(m.__dict__)   # None or {} are both reasonable answers
        m.foo = 1
        self.assertEqual(m.__dict__, {"foo": 1})

    def test_funny_new(self):
        # [FUNCTION DOC] Test: checks funny new behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing __new__ returning something unexpected...
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_funny_new to exercise
            # descriptor/class-model behavior under test.
            def __new__(cls, arg):
                # [FUNCTION DOC] HELPER: special method __new__ of C. Arguments: arg. Implements
                # the protocol hook used by tests; returns the hook result.
                if isinstance(arg, str): return [1, 2, 3]
                elif isinstance(arg, int): return object.__new__(D)
                else: return object.__new__(cls)
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_funny_new to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, arg):
                # [FUNCTION DOC] HELPER: constructor. Arguments: arg. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.foo = arg
        self.assertEqual(C("1"), [1, 2, 3])
        self.assertEqual(D("1"), [1, 2, 3])
        d = D(None)
        self.assertEqual(d.foo, None)
        d = C(1)
        self.assertIsInstance(d, D)
        self.assertEqual(d.foo, 1)
        d = D(1)
        self.assertIsInstance(d, D)
        self.assertEqual(d.foo, 1)

        class C(object):
            @staticmethod
            # [CLASS DOC] HELPER: local class C defined inside test_funny_new to exercise
            # descriptor/class-model behavior under test.
            def __new__(*args):
                # [FUNCTION DOC] HELPER: special method __new__ of C. Arguments: *args. Implements
                # the protocol hook used by tests; returns the hook result.
                return args
        self.assertEqual(C(1, 2), (C, 1, 2))
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_funny_new to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(D(1, 2), (D, 1, 2))

        class C(object):
            @classmethod
            # [CLASS DOC] HELPER: local class C defined inside test_funny_new to exercise
            # descriptor/class-model behavior under test.
            def __new__(*args):
                # [FUNCTION DOC] HELPER: special method __new__ of C. Arguments: *args. Implements
                # the protocol hook used by tests; returns the hook result.
                return args
        self.assertEqual(C(1, 2), (C, C, 1, 2))
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_funny_new to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertEqual(D(1, 2), (D, D, 1, 2))

    def test_imul_bug(self):
        # [FUNCTION DOC] Test: checks imul bug behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing for __imul__ problems...
        # SF bug 544647
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_imul_bug to exercise
            # descriptor/class-model behavior under test.
            def __imul__(self, other):
                # [FUNCTION DOC] HELPER: special method __imul__ of C. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return (self, other)
        x = C()
        y = x
        y *= 1.0
        self.assertEqual(y, (x, 1.0))
        y = x
        y *= 2
        self.assertEqual(y, (x, 2))
        y = x
        y *= 3
        self.assertEqual(y, (x, 3))
        y = x
        y *= 1<<100
        self.assertEqual(y, (x, 1<<100))
        y = x
        y *= None
        self.assertEqual(y, (x, None))
        y = x
        y *= "foo"
        self.assertEqual(y, (x, "foo"))

    def test_copy_setstate(self):
        # [FUNCTION DOC] Test: checks copy setstate behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing that copy.*copy() correctly uses __setstate__...
        import copy
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_copy_setstate to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, foo=None):
                # [FUNCTION DOC] HELPER: constructor. Arguments: foo. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.foo = foo
                self.__foo = foo
            def setfoo(self, foo=None):
                # [FUNCTION DOC] HELPER: setfoo of C. Arguments: foo. Supporting routine for
                # tests; returns a computed value or None.
                self.foo = foo
            def getfoo(self):
                # [FUNCTION DOC] HELPER: getfoo of C. Supporting routine for tests; returns a
                # computed value or None.
                return self.__foo
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: special method __getstate__ of C. Implements the protocol
                # hook used by tests; returns the hook result.
                return [self.foo]
            def __setstate__(self_, lst):
                # [FUNCTION DOC] HELPER: special method __setstate__ of C. Arguments: self_, lst.
                # Implements the protocol hook used by tests; returns the hook result.
                self.assertEqual(len(lst), 1)
                self_.__foo = self_.foo = lst[0]
        a = C(42)
        a.setfoo(24)
        self.assertEqual(a.foo, 24)
        self.assertEqual(a.getfoo(), 42)
        b = copy.copy(a)
        self.assertEqual(b.foo, 24)
        self.assertEqual(b.getfoo(), 24)
        b = copy.deepcopy(a)
        self.assertEqual(b.foo, 24)
        self.assertEqual(b.getfoo(), 24)

    def test_slices(self):
        # [FUNCTION DOC] Test: checks slices behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing cases with slices and overridden __getitem__ ...

        # Strings
        self.assertEqual("hello"[:4], "hell")
        self.assertEqual("hello"[slice(4)], "hell")
        self.assertEqual(str.__getitem__("hello", slice(4)), "hell")
        class S(str):
            # [CLASS DOC] HELPER: local class S defined inside test_slices to exercise
            # descriptor/class-model behavior under test.
            def __getitem__(self, x):
                # [FUNCTION DOC] HELPER: special method __getitem__ of S. Arguments: x. Implements
                # the protocol hook used by tests; returns the hook result.
                return str.__getitem__(self, x)
        self.assertEqual(S("hello")[:4], "hell")
        self.assertEqual(S("hello")[slice(4)], "hell")
        self.assertEqual(S("hello").__getitem__(slice(4)), "hell")
        # Tuples
        self.assertEqual((1,2,3)[:2], (1,2))
        self.assertEqual((1,2,3)[slice(2)], (1,2))
        self.assertEqual(tuple.__getitem__((1,2,3), slice(2)), (1,2))
        class T(tuple):
            # [CLASS DOC] HELPER: local class T defined inside test_slices to exercise
            # descriptor/class-model behavior under test.
            def __getitem__(self, x):
                # [FUNCTION DOC] HELPER: special method __getitem__ of T. Arguments: x. Implements
                # the protocol hook used by tests; returns the hook result.
                return tuple.__getitem__(self, x)
        self.assertEqual(T((1,2,3))[:2], (1,2))
        self.assertEqual(T((1,2,3))[slice(2)], (1,2))
        self.assertEqual(T((1,2,3)).__getitem__(slice(2)), (1,2))
        # Lists
        self.assertEqual([1,2,3][:2], [1,2])
        self.assertEqual([1,2,3][slice(2)], [1,2])
        self.assertEqual(list.__getitem__([1,2,3], slice(2)), [1,2])
        class L(list):
            # [CLASS DOC] HELPER: local class L defined inside test_slices to exercise
            # descriptor/class-model behavior under test.
            def __getitem__(self, x):
                # [FUNCTION DOC] HELPER: special method __getitem__ of L. Arguments: x. Implements
                # the protocol hook used by tests; returns the hook result.
                return list.__getitem__(self, x)
        self.assertEqual(L([1,2,3])[:2], [1,2])
        self.assertEqual(L([1,2,3])[slice(2)], [1,2])
        self.assertEqual(L([1,2,3]).__getitem__(slice(2)), [1,2])
        # Now do lists and __setitem__
        a = L([1,2,3])
        a[slice(1, 3)] = [3,2]
        self.assertEqual(a, [1,3,2])
        a[slice(0, 2, 1)] = [3,1]
        self.assertEqual(a, [3,1,2])
        a.__setitem__(slice(1, 3), [2,1])
        self.assertEqual(a, [3,2,1])
        a.__setitem__(slice(0, 2, 1), [2,3])
        self.assertEqual(a, [2,3,1])

    def test_subtype_resurrection(self):
        # [FUNCTION DOC] Test: checks subtype resurrection behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing resurrection of new-style instance...

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_subtype_resurrection to
            # exercise descriptor/class-model behavior under test.
            container = []

            def __del__(self):
                # [FUNCTION DOC] HELPER: special method __del__ of C. Implements the protocol hook
                # used by tests; returns the hook result.
                # resurrect the instance
                C.container.append(self)

        c = C()
        c.attr = 42

        # The most interesting thing here is whether this blows up, due to
        # flawed GC tracking logic in typeobject.c's call_finalizer() (a 2.2.1
        # bug).
        del c

        support.gc_collect()
        self.assertEqual(len(C.container), 1)

        # Make c mortal again, so that the test framework with -l doesn't report
        # it as a leak.
        del C.__del__

    def test_slots_trash(self):
        # [FUNCTION DOC] Test: checks slots trash behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing slot trash...
        # Deallocating deeply nested slotted trash caused stack overflows
        class trash(object):
            # [CLASS DOC] HELPER: local class trash defined inside test_slots_trash to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['x']
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: constructor. Arguments: x. Stores the given values on the
                # instance, and for test cases initializes the base CPythonTestCase.
                self.x = x
        o = None
        for i in range(50000):
            o = trash(o)
        del o

    def test_slots_multiple_inheritance(self):
        # [FUNCTION DOC] Test: checks slots multiple inheritance behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # SF bug 575229, multiple inheritance w/ slots dumps core
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_slots_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            __slots__=()
        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside test_slots_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            pass
        class C(A,B) :
            # [CLASS DOC] HELPER: local class C defined inside test_slots_multiple_inheritance to
            # exercise descriptor/class-model behavior under test.
            __slots__=()
        if support.check_impl_detail():
            self.assertEqual(C.__basicsize__, B.__basicsize__)
        self.assertHasAttr(C, '__dict__')
        self.assertHasAttr(C, '__weakref__')
        C().x = 2

    def test_rmul(self):
        # [FUNCTION DOC] Test: checks rmul behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing correct invocation of __rmul__...
        # SF patch 592646
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_rmul to exercise
            # descriptor/class-model behavior under test.
            def __mul__(self, other):
                # [FUNCTION DOC] HELPER: special method __mul__ of C. Arguments: other. Implements
                # the protocol hook used by tests; returns the hook result.
                return "mul"
            def __rmul__(self, other):
                # [FUNCTION DOC] HELPER: special method __rmul__ of C. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "rmul"
        a = C()
        self.assertEqual(a*2, "mul")
        self.assertEqual(a*2.2, "mul")
        self.assertEqual(2*a, "rmul")
        self.assertEqual(2.2*a, "rmul")

    def test_ipow(self):
        # [FUNCTION DOC] Test: checks ipow behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing correct invocation of __ipow__...
        # [SF bug 620179]
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_ipow to exercise
            # descriptor/class-model behavior under test.
            def __ipow__(self, other):
                # [FUNCTION DOC] HELPER: special method __ipow__ of C. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                pass
        a = C()
        a **= 2

    def test_ipow_returns_not_implemented(self):
        # [FUNCTION DOC] Test: checks ipow returns not implemented behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        class A:
            # [CLASS DOC] HELPER: local class A defined inside test_ipow_returns_not_implemented
            # to exercise descriptor/class-model behavior under test.
            def __ipow__(self, other):
                # [FUNCTION DOC] HELPER: special method __ipow__ of A. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return NotImplemented

        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_ipow_returns_not_implemented
            # to exercise descriptor/class-model behavior under test.
            def __rpow__(self, other):
                # [FUNCTION DOC] HELPER: special method __rpow__ of B. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return 1

        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside test_ipow_returns_not_implemented
            # to exercise descriptor/class-model behavior under test.
            def __pow__(self, other):
                # [FUNCTION DOC] HELPER: special method __pow__ of C. Arguments: other. Implements
                # the protocol hook used by tests; returns the hook result.
                return 2
        a = A()
        b = B()
        c = C()

        a **= b
        self.assertEqual(a, 1)

        c **= b
        self.assertEqual(c, 2)

    def test_no_ipow(self):
        # [FUNCTION DOC] Test: checks no ipow behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        class B:
            # [CLASS DOC] HELPER: local class B defined inside test_no_ipow to exercise
            # descriptor/class-model behavior under test.
            def __rpow__(self, other):
                # [FUNCTION DOC] HELPER: special method __rpow__ of B. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return 1

        a = object()
        b = B()
        a **= b
        self.assertEqual(a, 1)

    def test_ipow_exception_text(self):
        # [FUNCTION DOC] Test: checks ipow exception text behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        x = None
        with self.assertRaises(TypeError) as cm:
            x **= 2
        self.assertIn('unsupported operand type(s) for **=', str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            y = x ** 2
        self.assertIn('unsupported operand type(s) for **', str(cm.exception))

    def test_mutable_bases(self):
        # [FUNCTION DOC] Test: checks mutable bases behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing mutable bases...

        # stuff that should work:
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_mutable_bases to exercise
            # descriptor/class-model behavior under test.
            pass
        class C2(object):
            # [CLASS DOC] HELPER: local class C2 defined inside test_mutable_bases to exercise
            # descriptor/class-model behavior under test.
            def __getattribute__(self, attr):
                # [FUNCTION DOC] HELPER: special method __getattribute__ of C2. Arguments: attr.
                # Implements the protocol hook used by tests; returns the hook result.
                if attr == 'a':
                    return 2
                else:
                    return super(C2, self).__getattribute__(attr)
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of C2. Supporting routine for tests; returns a
                # computed value or None.
                return 1
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_mutable_bases to exercise
            # descriptor/class-model behavior under test.
            pass
        class E(D):
            # [CLASS DOC] HELPER: local class E defined inside test_mutable_bases to exercise
            # descriptor/class-model behavior under test.
            pass
        d = D()
        e = E()
        D.__bases__ = (C,)
        D.__bases__ = (C2,)
        self.assertEqual(d.meth(), 1)
        self.assertEqual(e.meth(), 1)
        self.assertEqual(d.a, 2)
        self.assertEqual(e.a, 2)
        self.assertEqual(C2.__subclasses__(), [D])

        try:
            del D.__bases__
        except (TypeError, AttributeError):
            pass
        else:
            self.fail("shouldn't be able to delete .__bases__")

        try:
            D.__bases__ = ()
        except TypeError as msg:
            if str(msg) == "a new-style class can't have only classic bases":
                self.fail("wrong error message for .__bases__ = ()")
        else:
            self.fail("shouldn't be able to set .__bases__ to ()")

        try:
            D.__bases__ = (D,)
        except TypeError:
            pass
        else:
            # actually, we'll have crashed by here...
            self.fail("shouldn't be able to create inheritance cycles")

        try:
            D.__bases__ = (C, C)
        except TypeError:
            pass
        else:
            self.fail("didn't detect repeated base classes")

        try:
            D.__bases__ = (E,)
        except TypeError:
            pass
        else:
            self.fail("shouldn't be able to create inheritance cycles")

    def test_builtin_bases(self):
        # [FUNCTION DOC] Test: checks builtin bases behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Make sure all the builtin types can have their base queried without
        # segfaulting. See issue #5787.
        builtin_types = [tp for tp in builtins.__dict__.values()
                         if isinstance(tp, type)]
        for tp in builtin_types:
            object.__getattribute__(tp, "__bases__")
            if tp is not object:
                if tp is ExceptionGroup:
                    num_bases = 2
                else:
                    num_bases = 1
                self.assertEqual(len(tp.__bases__), num_bases, tp)

        class L(list):
            # [CLASS DOC] HELPER: local class L defined inside test_builtin_bases to exercise
            # descriptor/class-model behavior under test.
            pass

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_builtin_bases to exercise
            # descriptor/class-model behavior under test.
            pass

        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_builtin_bases to exercise
            # descriptor/class-model behavior under test.
            pass

        try:
            L.__bases__ = (dict,)
        except TypeError:
            pass
        else:
            self.fail("shouldn't turn list subclass into dict subclass")

        try:
            list.__bases__ = (dict,)
        except TypeError:
            pass
        else:
            self.fail("shouldn't be able to assign to list.__bases__")

        try:
            D.__bases__ = (C, list)
        except TypeError:
            pass
        else:
            self.fail("best_base calculation found wanting")

    def test_unsubclassable_types(self):
        # [FUNCTION DOC] Test: checks unsubclassable types behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        with self.assertRaises(TypeError):
            class X(type(None)):
                # [CLASS DOC] HELPER: local class X defined inside test_unsubclassable_types to
                # exercise descriptor/class-model behavior under test.
                pass
        with self.assertRaises(TypeError):
            class X(object, type(None)):
                # [CLASS DOC] HELPER: local class X defined inside test_unsubclassable_types to
                # exercise descriptor/class-model behavior under test.
                pass
        with self.assertRaises(TypeError):
            class X(type(None), object):
                # [CLASS DOC] HELPER: local class X defined inside test_unsubclassable_types to
                # exercise descriptor/class-model behavior under test.
                pass
        class O(object):
            # [CLASS DOC] HELPER: local class O defined inside test_unsubclassable_types to
            # exercise descriptor/class-model behavior under test.
            pass
        with self.assertRaises(TypeError):
            class X(O, type(None)):
                # [CLASS DOC] HELPER: local class X defined inside test_unsubclassable_types to
                # exercise descriptor/class-model behavior under test.
                pass
        with self.assertRaises(TypeError):
            class X(type(None), O):
                # [CLASS DOC] HELPER: local class X defined inside test_unsubclassable_types to
                # exercise descriptor/class-model behavior under test.
                pass

        class X(object):
            # [CLASS DOC] HELPER: local class X defined inside test_unsubclassable_types to
            # exercise descriptor/class-model behavior under test.
            pass
        with self.assertRaises(TypeError):
            X.__bases__ = type(None),
        with self.assertRaises(TypeError):
            X.__bases__ = object, type(None)
        with self.assertRaises(TypeError):
            X.__bases__ = type(None), object
        with self.assertRaises(TypeError):
            X.__bases__ = O, type(None)
        with self.assertRaises(TypeError):
            X.__bases__ = type(None), O

    def test_mutable_bases_with_failing_mro(self):
        # [FUNCTION DOC] Test: checks mutable bases with failing mro behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # Testing mutable bases with failing mro...
        class WorkOnce(type):
            # [CLASS DOC] HELPER: local class WorkOnce defined inside
            # test_mutable_bases_with_failing_mro to exercise descriptor/class-model behavior
            # under test.
            def __new__(self, name, bases, ns):
                # [FUNCTION DOC] HELPER: special method __new__ of WorkOnce. Arguments: name,
                # bases, ns. Implements the protocol hook used by tests; returns the hook result.
                self.flag = 0
                return super(WorkOnce, self).__new__(WorkOnce, name, bases, ns)
            def mro(self):
                # [FUNCTION DOC] HELPER: mro of WorkOnce. Supporting routine for tests; returns a
                # computed value or None.
                if self.flag > 0:
                    raise RuntimeError("bozo")
                else:
                    self.flag += 1
                    return type.mro(self)

        class WorkAlways(type):
            # [CLASS DOC] HELPER: local class WorkAlways defined inside
            # test_mutable_bases_with_failing_mro to exercise descriptor/class-model behavior
            # under test.
            def mro(self):
                # [FUNCTION DOC] HELPER: mro of WorkAlways. Supporting routine for tests; returns
                # a computed value or None.
                # this is here to make sure that .mro()s aren't called
                # with an exception set (which was possible at one point).
                # An error message will be printed in a debug build.
                # What's a good way to test for this?
                return type.mro(self)

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_mutable_bases_with_failing_mro
            # to exercise descriptor/class-model behavior under test.
            pass

        class C2(object):
            # [CLASS DOC] HELPER: local class C2 defined inside
            # test_mutable_bases_with_failing_mro to exercise descriptor/class-model behavior
            # under test.
            pass

        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_mutable_bases_with_failing_mro
            # to exercise descriptor/class-model behavior under test.
            pass

        class E(D):
            # [CLASS DOC] HELPER: local class E defined inside test_mutable_bases_with_failing_mro
            # to exercise descriptor/class-model behavior under test.
            pass

        class F(D, metaclass=WorkOnce):
            # [CLASS DOC] HELPER: local class F defined inside test_mutable_bases_with_failing_mro
            # to exercise descriptor/class-model behavior under test.
            pass

        class G(D, metaclass=WorkAlways):
            # [CLASS DOC] HELPER: local class G defined inside test_mutable_bases_with_failing_mro
            # to exercise descriptor/class-model behavior under test.
            pass

        # Immediate subclasses have their mro's adjusted in alphabetical
        # order, so E's will get adjusted before adjusting F's fails.  We
        # check here that E's gets restored.

        E_mro_before = E.__mro__
        D_mro_before = D.__mro__

        try:
            D.__bases__ = (C2,)
        except RuntimeError:
            self.assertEqual(E.__mro__, E_mro_before)
            self.assertEqual(D.__mro__, D_mro_before)
        else:
            self.fail("exception not propagated")

    def test_mutable_bases_catch_mro_conflict(self):
        # [FUNCTION DOC] Test: checks mutable bases catch mro conflict behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # Testing mutable bases catch mro conflict...
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside
            # test_mutable_bases_catch_mro_conflict to exercise descriptor/class-model behavior
            # under test.
            pass

        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside
            # test_mutable_bases_catch_mro_conflict to exercise descriptor/class-model behavior
            # under test.
            pass

        class C(A, B):
            # [CLASS DOC] HELPER: local class C defined inside
            # test_mutable_bases_catch_mro_conflict to exercise descriptor/class-model behavior
            # under test.
            pass

        class D(A, B):
            # [CLASS DOC] HELPER: local class D defined inside
            # test_mutable_bases_catch_mro_conflict to exercise descriptor/class-model behavior
            # under test.
            pass

        class E(C, D):
            # [CLASS DOC] HELPER: local class E defined inside
            # test_mutable_bases_catch_mro_conflict to exercise descriptor/class-model behavior
            # under test.
            pass

        try:
            C.__bases__ = (B, A)
        except TypeError:
            pass
        else:
            self.fail("didn't catch MRO conflict")

    def test_mutable_names(self):
        # [FUNCTION DOC] Test: checks mutable names behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing mutable names...
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_mutable_names to exercise
            # descriptor/class-model behavior under test.
            pass

        # C.__module__ could be 'test_descr' or '__main__'
        mod = C.__module__

        C.__name__ = 'D'
        self.assertEqual((C.__module__, C.__name__), (mod, 'D'))

        C.__name__ = 'D.E'
        self.assertEqual((C.__module__, C.__name__), (mod, 'D.E'))

    def test_evil_type_name(self):
        # [FUNCTION DOC] Test: checks evil type name behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # A badly placed Py_DECREF in type_set_name led to arbitrary code
        # execution while the type structure was not in a sane state, and a
        # possible segmentation fault as a result.  See bug #16447.
        class Nasty(str):
            # [CLASS DOC] HELPER: local class Nasty defined inside test_evil_type_name to exercise
            # descriptor/class-model behavior under test.
            def __del__(self):
                # [FUNCTION DOC] HELPER: special method __del__ of Nasty. Implements the protocol
                # hook used by tests; returns the hook result.
                C.__name__ = "other"

        class C:
            # [CLASS DOC] HELPER: local class C defined inside test_evil_type_name to exercise
            # descriptor/class-model behavior under test.
            pass

        C.__name__ = Nasty("abc")
        C.__name__ = "normal"

    def test_subclass_right_op(self):
        # [FUNCTION DOC] Test: checks subclass right op behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing correct dispatch of subclass overloading __r<op>__...

        # This code tests various cases where right-dispatch of a subclass
        # should be preferred over left-dispatch of a base class.

        # Case 1: subclass of int; this tests code in abstract.c::binary_op1()

        class B(int):
            # [CLASS DOC] HELPER: local class B defined inside test_subclass_right_op to exercise
            # descriptor/class-model behavior under test.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: special method __floordiv__ of B. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "B.__floordiv__"
            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: special method __rfloordiv__ of B. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "B.__rfloordiv__"

        self.assertEqual(B(1) // 1, "B.__floordiv__")
        self.assertEqual(1 // B(1), "B.__rfloordiv__")

        # Case 2: subclass of object; this is just the baseline for case 3

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_subclass_right_op to exercise
            # descriptor/class-model behavior under test.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: special method __floordiv__ of C. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "C.__floordiv__"
            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: special method __rfloordiv__ of C. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "C.__rfloordiv__"

        self.assertEqual(C() // 1, "C.__floordiv__")
        self.assertEqual(1 // C(), "C.__rfloordiv__")

        # Case 3: subclass of new-style class; here it gets interesting

        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_subclass_right_op to exercise
            # descriptor/class-model behavior under test.
            def __floordiv__(self, other):
                # [FUNCTION DOC] HELPER: special method __floordiv__ of D. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "D.__floordiv__"
            def __rfloordiv__(self, other):
                # [FUNCTION DOC] HELPER: special method __rfloordiv__ of D. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                return "D.__rfloordiv__"

        self.assertEqual(D() // C(), "D.__floordiv__")
        self.assertEqual(C() // D(), "D.__rfloordiv__")

        # Case 4: this didn't work right in 2.2.2 and 2.3a1

        class E(C):
            # [CLASS DOC] HELPER: local class E defined inside test_subclass_right_op to exercise
            # descriptor/class-model behavior under test.
            pass

        self.assertEqual(E.__rfloordiv__, C.__rfloordiv__)

        self.assertEqual(E() // 1, "C.__floordiv__")
        self.assertEqual(1 // E(), "C.__rfloordiv__")
        self.assertEqual(E() // C(), "C.__floordiv__")
        self.assertEqual(C() // E(), "C.__floordiv__") # This one would fail

    @support.impl_detail("testing an internal kind of method object")
    def test_meth_class_get(self):
        # [FUNCTION DOC] Test: checks meth class get behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing __get__ method of METH_CLASS C methods...
        # Full coverage of descrobject.c::classmethod_get()

        # Baseline
        arg = [1, 2, 3]
        res = {1: None, 2: None, 3: None}
        self.assertEqual(dict.fromkeys(arg), res)
        self.assertEqual({}.fromkeys(arg), res)

        # Now get the descriptor
        descr = dict.__dict__["fromkeys"]

        # More baseline using the descriptor directly
        self.assertEqual(descr.__get__(None, dict)(arg), res)
        self.assertEqual(descr.__get__({})(arg), res)

        # Now check various error cases
        try:
            descr.__get__(None, None)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(None, None)")
        try:
            descr.__get__(42)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(42)")
        try:
            descr.__get__(None, 42)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(None, 42)")
        try:
            descr.__get__(None, int)
        except TypeError:
            pass
        else:
            self.fail("shouldn't have allowed descr.__get__(None, int)")

    def test_isinst_isclass(self):
        # [FUNCTION DOC] Test: checks isinst isclass behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing proxy isinstance() and isclass()...
        class Proxy(object):
            # [CLASS DOC] HELPER: local class Proxy defined inside test_isinst_isclass to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: constructor. Arguments: obj. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.__obj = obj
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattribute__ of Proxy. Arguments:
                # name. Implements the protocol hook used by tests; returns the hook result.
                if name.startswith("_Proxy__"):
                    return object.__getattribute__(self, name)
                else:
                    return getattr(self.__obj, name)
        # Test with a classic class
        class C:
            # [CLASS DOC] HELPER: local class C defined inside test_isinst_isclass to exercise
            # descriptor/class-model behavior under test.
            pass
        a = C()
        pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test
        # Test with a classic subclass
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_isinst_isclass to exercise
            # descriptor/class-model behavior under test.
            pass
        a = D()
        pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test
        # Test with a new-style class
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_isinst_isclass to exercise
            # descriptor/class-model behavior under test.
            pass
        a = C()
        pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test
        # Test with a new-style subclass
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_isinst_isclass to exercise
            # descriptor/class-model behavior under test.
            pass
        a = D()
        pa = Proxy(a)
        self.assertIsInstance(a, C)  # Baseline
        self.assertIsInstance(pa, C) # Test

    def test_proxy_super(self):
        # [FUNCTION DOC] Test: checks proxy super behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing super() for a proxy object...
        class Proxy(object):
            # [CLASS DOC] HELPER: local class Proxy defined inside test_proxy_super to exercise
            # descriptor/class-model behavior under test.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: constructor. Arguments: obj. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.__obj = obj
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattribute__ of Proxy. Arguments:
                # name. Implements the protocol hook used by tests; returns the hook result.
                if name.startswith("_Proxy__"):
                    return object.__getattribute__(self, name)
                else:
                    return getattr(self.__obj, name)

        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside test_proxy_super to exercise
            # descriptor/class-model behavior under test.
            def f(self):
                # [FUNCTION DOC] HELPER: f of B. Supporting routine for tests; returns a computed
                # value or None.
                return "B.f"

        class C(B):
            # [CLASS DOC] HELPER: local class C defined inside test_proxy_super to exercise
            # descriptor/class-model behavior under test.
            def f(self):
                # [FUNCTION DOC] HELPER: f of C. Supporting routine for tests; returns a computed
                # value or None.
                return super(C, self).f() + "->C.f"

        obj = C()
        p = Proxy(obj)
        self.assertEqual(C.__dict__["f"](p), "B.f->C.f")

    def test_carloverre(self):
        # [FUNCTION DOC] Test: checks carloverre behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing prohibition of Carlo Verre's hack...
        try:
            object.__setattr__(str, "foo", 42)
        except TypeError:
            pass
        else:
            self.fail("Carlo Verre __setattr__ succeeded!")
        try:
            object.__delattr__(str, "lower")
        except TypeError:
            pass
        else:
            self.fail("Carlo Verre __delattr__ succeeded!")

    def test_carloverre_multi_inherit_valid(self):
        # [FUNCTION DOC] Test: checks carloverre multi inherit valid behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        class A(type):
            # [CLASS DOC] HELPER: local class A defined inside test_carloverre_multi_inherit_valid
            # to exercise descriptor/class-model behavior under test.
            def __setattr__(cls, key, value):
                # [FUNCTION DOC] HELPER: special method __setattr__ of A. Arguments: key, value.
                # Implements the protocol hook used by tests; returns the hook result.
                type.__setattr__(cls, key, value)

        class B:
            # [CLASS DOC] HELPER: local class B defined inside test_carloverre_multi_inherit_valid
            # to exercise descriptor/class-model behavior under test.
            pass

        class C(B, A):
            # [CLASS DOC] HELPER: local class C defined inside test_carloverre_multi_inherit_valid
            # to exercise descriptor/class-model behavior under test.
            pass

        obj = C('D', (object,), {})
        try:
            obj.test = True
        except TypeError:
            self.fail("setattr through direct base types should be legal")

    def test_carloverre_multi_inherit_invalid(self):
        # [FUNCTION DOC] Test: checks carloverre multi inherit invalid behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        class A(type):
            # [CLASS DOC] HELPER: local class A defined inside
            # test_carloverre_multi_inherit_invalid to exercise descriptor/class-model behavior
            # under test.
            def __setattr__(cls, key, value):
                # [FUNCTION DOC] HELPER: special method __setattr__ of A. Arguments: key, value.
                # Implements the protocol hook used by tests; returns the hook result.
                object.__setattr__(cls, key, value)  # this should fail!

        class B:
            # [CLASS DOC] HELPER: local class B defined inside
            # test_carloverre_multi_inherit_invalid to exercise descriptor/class-model behavior
            # under test.
            pass

        class C(B, A):
            # [CLASS DOC] HELPER: local class C defined inside
            # test_carloverre_multi_inherit_invalid to exercise descriptor/class-model behavior
            # under test.
            pass

        obj = C('D', (object,), {})
        try:
            obj.test = True
        except TypeError:
            pass
        else:
            self.fail("setattr through indirect base types should be rejected")

    def test_weakref_segfault(self):
        # [FUNCTION DOC] Test: checks weakref segfault behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing weakref segfault...
        # SF 742911
        import weakref

        class Provoker:
            # [CLASS DOC] HELPER: local class Provoker defined inside test_weakref_segfault to
            # exercise descriptor/class-model behavior under test.
            def __init__(self, referrent):
                # [FUNCTION DOC] HELPER: constructor. Arguments: referrent. Stores the given
                # values on the instance, and for test cases initializes the base CPythonTestCase.
                self.ref = weakref.ref(referrent)

            def __del__(self):
                # [FUNCTION DOC] HELPER: special method __del__ of Provoker. Implements the
                # protocol hook used by tests; returns the hook result.
                x = self.ref()

        class Oops(object):
            # [CLASS DOC] HELPER: local class Oops defined inside test_weakref_segfault to
            # exercise descriptor/class-model behavior under test.
            pass

        o = Oops()
        o.whatever = Provoker(o)
        del o

    @support.requires_resource('cpu')
    def test_wrapper_segfault(self):
        # [FUNCTION DOC] Test: checks wrapper segfault behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # SF 927248: deeply nested wrappers could cause stack overflow
        f = lambda:None
        for i in range(1000000):
            f = f.__call__
        f = None

    def test_file_fault(self):
        # [FUNCTION DOC] Test: checks file fault behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing sys.stdout is changed in getattr...
        class StdoutGuard:
            # [CLASS DOC] HELPER: local class StdoutGuard defined inside test_file_fault to
            # exercise descriptor/class-model behavior under test.
            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: special method __getattr__ of StdoutGuard. Arguments:
                # attr. Implements the protocol hook used by tests; returns the hook result.
                sys.stdout = sys.__stdout__
                raise RuntimeError(f"Premature access to sys.stdout.{attr}")

        with redirect_stdout(StdoutGuard()):
            with self.assertRaises(RuntimeError):
                print("Oops!")

    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    def test_vicious_descriptor_nonsense(self):
        # [FUNCTION DOC] Test: checks vicious descriptor nonsense behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # Testing vicious_descriptor_nonsense...

        # A potential segfault spotted by Thomas Wouters in mail to
        # python-dev 2003-04-17, turned into an example & fixed by Michael
        # Hudson just less than four months later...

        class Evil(object):
            # [CLASS DOC] HELPER: local class Evil defined inside test_vicious_descriptor_nonsense
            # to exercise descriptor/class-model behavior under test.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: special method __hash__ of Evil. Implements the protocol
                # hook used by tests; returns the hook result.
                return hash('attr')
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: special method __eq__ of Evil. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                try:
                    del C.attr
                except AttributeError:
                    # possible race condition
                    pass
                return 0

        class Descr(object):
            # [CLASS DOC] HELPER: local class Descr defined inside
            # test_vicious_descriptor_nonsense to exercise descriptor/class-model behavior under
            # test.
            def __get__(self, ob, type=None):
                # [FUNCTION DOC] HELPER: special method __get__ of Descr. Arguments: ob, type.
                # Implements the protocol hook used by tests; returns the hook result.
                return 1

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_vicious_descriptor_nonsense to
            # exercise descriptor/class-model behavior under test.
            attr = Descr()

        c = C()
        c.__dict__[Evil()] = 0

        self.assertEqual(c.attr, 1)
        # this makes a crash more likely:
        support.gc_collect()
        self.assertNotHasAttr(c, 'attr')

    def test_init(self):
        # [FUNCTION DOC] Test: checks init behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # SF 1155938
        class Foo(object):
            # [CLASS DOC] HELPER: local class Foo defined inside test_init to exercise
            # descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                return 10
        try:
            Foo()
        except TypeError:
            pass
        else:
            self.fail("did not test __init__() for None return")

    def assertNotOrderable(self, a, b):
        # [FUNCTION DOC] HELPER: assertNotOrderable of ClassPropertiesAndMethods. Arguments: a, b.
        # Supporting routine for tests; returns a computed value or None.
        with self.assertRaises(TypeError):
            a < b
        with self.assertRaises(TypeError):
            a > b
        with self.assertRaises(TypeError):
            a <= b
        with self.assertRaises(TypeError):
            a >= b

    def test_method_wrapper(self):
        # [FUNCTION DOC] Test: checks method wrapper behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing method-wrapper objects...
        # <type 'method-wrapper'> did not support any reflection before 2.5
        l = []
        self.assertTrue(l.__add__ == l.__add__)
        self.assertFalse(l.__add__ != l.__add__)
        self.assertFalse(l.__add__ == [].__add__)
        self.assertTrue(l.__add__ != [].__add__)
        self.assertFalse(l.__add__ == l.__mul__)
        self.assertTrue(l.__add__ != l.__mul__)
        self.assertNotOrderable(l.__add__, l.__add__)
        self.assertEqual(l.__add__.__name__, '__add__')
        self.assertIs(l.__add__.__self__, l)
        self.assertIs(l.__add__.__objclass__, list)
        self.assertEqual(l.__add__.__doc__, list.__add__.__doc__)
        # hash([].__add__) should not be based on hash([])
        hash(l.__add__)

    def test_builtin_function_or_method(self):
        # [FUNCTION DOC] Test: checks builtin function or method behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # Not really belonging to test_descr, but introspection and
        # comparison on <type 'builtin_function_or_method'> seems not
        # to be tested elsewhere
        l = []
        self.assertTrue(l.append == l.append)
        self.assertFalse(l.append != l.append)
        self.assertFalse(l.append == [].append)
        self.assertTrue(l.append != [].append)
        self.assertFalse(l.append == l.pop)
        self.assertTrue(l.append != l.pop)
        self.assertNotOrderable(l.append, l.append)
        self.assertEqual(l.append.__name__, 'append')
        self.assertIs(l.append.__self__, l)
        # self.assertIs(l.append.__objclass__, list) --- could be added?
        self.assertEqual(l.append.__doc__, list.append.__doc__)
        # hash([].append) should not be based on hash([])
        hash(l.append)

    def test_special_unbound_method_types(self):
        # [FUNCTION DOC] Test: checks special unbound method types behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # Testing objects of <type 'wrapper_descriptor'>...
        self.assertTrue(list.__add__ == list.__add__)
        self.assertFalse(list.__add__ != list.__add__)
        self.assertFalse(list.__add__ == list.__mul__)
        self.assertTrue(list.__add__ != list.__mul__)
        self.assertNotOrderable(list.__add__, list.__add__)
        self.assertEqual(list.__add__.__name__, '__add__')
        self.assertIs(list.__add__.__objclass__, list)

        # Testing objects of <type 'method_descriptor'>...
        self.assertTrue(list.append == list.append)
        self.assertFalse(list.append != list.append)
        self.assertFalse(list.append == list.pop)
        self.assertTrue(list.append != list.pop)
        self.assertNotOrderable(list.append, list.append)
        self.assertEqual(list.append.__name__, 'append')
        self.assertIs(list.append.__objclass__, list)

    def test_not_implemented(self):
        # [FUNCTION DOC] Test: checks not implemented behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing NotImplemented...
        # all binary methods should be able to return a NotImplemented

        def specialmethod(self, other):
            # [FUNCTION DOC] HELPER: local function specialmethod inside test_not_implemented.
            # Arguments: other. Returns its computed value or None; used by the enclosing test.
            return NotImplemented

        def check(expr, x, y):
            # [FUNCTION DOC] HELPER: local function check inside test_not_implemented. Arguments:
            # expr, x, y. Returns its computed value or None; used by the enclosing test.
            with (
                self.subTest(expr=expr, x=x, y=y),
                self.assertRaises(TypeError),
            ):
                with torch._dynamo.error_on_graph_break(False):
                    exec(expr, {'x': x, 'y': y})

        N1 = sys.maxsize + 1    # might trigger OverflowErrors instead of
                                # TypeErrors
        N2 = sys.maxsize         # if sizeof(int) < sizeof(long), might trigger
                                #   ValueErrors instead of TypeErrors
        for name, expr, iexpr in [
                ('__add__',      'x + y',                   'x += y'),
                ('__sub__',      'x - y',                   'x -= y'),
                ('__mul__',      'x * y',                   'x *= y'),
                ('__matmul__',   'x @ y',                   'x @= y'),
                ('__truediv__',  'x / y',                   'x /= y'),
                ('__floordiv__', 'x // y',                  'x //= y'),
                ('__mod__',      'x % y',                   'x %= y'),
                ('__divmod__',   'divmod(x, y)',            None),
                ('__pow__',      'x ** y',                  'x **= y'),
                ('__lshift__',   'x << y',                  'x <<= y'),
                ('__rshift__',   'x >> y',                  'x >>= y'),
                ('__and__',      'x & y',                   'x &= y'),
                ('__or__',       'x | y',                   'x |= y'),
                ('__xor__',      'x ^ y',                   'x ^= y')]:
            # Defines 'left' magic method:
            A = type('A', (), {name: specialmethod})
            a = A()
            check(expr, a, a)
            check(expr, a, N1)
            check(expr, a, N2)
            # Defines 'right' magic method:
            rname = '__r' + name[2:]
            B = type('B', (), {rname: specialmethod})
            b = B()
            check(expr, b, b)
            check(expr, a, b)
            check(expr, b, a)
            check(expr, b, N1)
            check(expr, b, N2)
            check(expr, N1, b)
            check(expr, N2, b)
            if iexpr:
                check(iexpr, a, a)
                check(iexpr, a, N1)
                check(iexpr, a, N2)
                iname = '__i' + name[2:]
                C = type('C', (), {iname: specialmethod})
                c = C()
                check(iexpr, c, a)
                check(iexpr, c, N1)
                check(iexpr, c, N2)

    def test_assign_slice(self):
        # [FUNCTION DOC] Test: checks assign slice behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # ceval.c's assign_slice used to check for
        # tp->tp_as_sequence->sq_slice instead of
        # tp->tp_as_sequence->sq_ass_slice

        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_assign_slice to exercise
            # descriptor/class-model behavior under test.
            def __setitem__(self, idx, value):
                # [FUNCTION DOC] HELPER: special method __setitem__ of C. Arguments: idx, value.
                # Implements the protocol hook used by tests; returns the hook result.
                self.value = value

        c = C()
        c[1:2] = 3
        self.assertEqual(c.value, 3)

    def test_set_and_no_get(self):
        # [FUNCTION DOC] Test: checks set and no get behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # See
        # http://mail.python.org/pipermail/python-dev/2010-January/095637.html
        class Descr(object):
            # [CLASS DOC] HELPER: local class Descr defined inside test_set_and_no_get to exercise
            # descriptor/class-model behavior under test.

            def __init__(self, name):
                # [FUNCTION DOC] HELPER: constructor. Arguments: name. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.name = name

            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: special method __set__ of Descr. Arguments: obj, value.
                # Implements the protocol hook used by tests; returns the hook result.
                obj.__dict__[self.name] = value
        descr = Descr("a")

        class X(object):
            # [CLASS DOC] HELPER: local class X defined inside test_set_and_no_get to exercise
            # descriptor/class-model behavior under test.
            a = descr

        x = X()
        self.assertIs(x.a, descr)
        x.a = 42
        self.assertEqual(x.a, 42)

        # Also check type_getattro for correctness.
        class Meta(type):
            # [CLASS DOC] HELPER: local class Meta defined inside test_set_and_no_get to exercise
            # descriptor/class-model behavior under test.
            pass
        class X(metaclass=Meta):
            # [CLASS DOC] HELPER: local class X defined inside test_set_and_no_get to exercise
            # descriptor/class-model behavior under test.
            pass
        X.a = 42
        Meta.a = Descr("a")
        self.assertEqual(X.a, 42)

    def test_getattr_hooks(self):
        # [FUNCTION DOC] Test: checks getattr hooks behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # issue 4230

        class Descriptor(object):
            # [CLASS DOC] HELPER: local class Descriptor defined inside test_getattr_hooks to
            # exercise descriptor/class-model behavior under test.
            counter = 0
            def __get__(self, obj, objtype=None):
                # [FUNCTION DOC] HELPER: special method __get__ of Descriptor. Arguments: obj,
                # objtype. Implements the protocol hook used by tests; returns the hook result.
                def getter(name):
                    # [FUNCTION DOC] HELPER: local function getter inside __get__. Arguments:
                    # name. Returns its computed value or None; used by the enclosing test.
                    self.counter += 1
                    raise AttributeError(name)
                return getter

        descr = Descriptor()
        class A(object):
            # [CLASS DOC] HELPER: local class A defined inside test_getattr_hooks to exercise
            # descriptor/class-model behavior under test.
            __getattribute__ = descr
        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside test_getattr_hooks to exercise
            # descriptor/class-model behavior under test.
            __getattr__ = descr
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside test_getattr_hooks to exercise
            # descriptor/class-model behavior under test.
            __getattribute__ = descr
            __getattr__ = descr

        self.assertRaises(AttributeError, getattr, A(), "attr")
        self.assertEqual(descr.counter, 1)
        self.assertRaises(AttributeError, getattr, B(), "attr")
        self.assertEqual(descr.counter, 2)
        self.assertRaises(AttributeError, getattr, C(), "attr")
        self.assertEqual(descr.counter, 4)

        class EvilGetattribute(object):
            # [CLASS DOC] HELPER: local class EvilGetattribute defined inside test_getattr_hooks
            # to exercise descriptor/class-model behavior under test.
            # This used to segfault
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattr__ of EvilGetattribute.
                # Arguments: name. Implements the protocol hook used by tests; returns the hook
                # result.
                raise AttributeError(name)
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattribute__ of EvilGetattribute.
                # Arguments: name. Implements the protocol hook used by tests; returns the hook
                # result.
                del EvilGetattribute.__getattr__
                for i in range(5):
                    gc.collect()
                raise AttributeError(name)

        self.assertRaises(AttributeError, getattr, EvilGetattribute(), "attr")

    def test_type___getattribute__(self):
        # [FUNCTION DOC] Test: checks type   getattribute behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        self.assertRaises(TypeError, type.__getattribute__, list, type)

    def test_abstractmethods(self):
        # [FUNCTION DOC] Test: checks abstractmethods behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # type pretends not to have __abstractmethods__.
        self.assertRaises(AttributeError, getattr, type, "__abstractmethods__")
        class meta(type):
            # [CLASS DOC] HELPER: local class meta defined inside test_abstractmethods to exercise
            # descriptor/class-model behavior under test.
            pass
        self.assertRaises(AttributeError, getattr, meta, "__abstractmethods__")
        class X(object):
            # [CLASS DOC] HELPER: local class X defined inside test_abstractmethods to exercise
            # descriptor/class-model behavior under test.
            pass
        with self.assertRaises(AttributeError):
            del X.__abstractmethods__

    def test_gh55664(self):
        # [FUNCTION DOC] Test: checks gh55664 behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # gh-55664: issue a warning when the
        # __dict__ of a class contains non-string keys
        with self.assertWarnsRegex(RuntimeWarning, 'MyClass'):
            MyClass = type('MyClass', (), {1: 2})

        class meta(type):
            # [CLASS DOC] HELPER: local class meta defined inside test_gh55664 to exercise
            # descriptor/class-model behavior under test.
            def __new__(mcls, name, bases, ns):
                # [FUNCTION DOC] HELPER: special method __new__ of meta. Arguments: mcls, name,
                # bases, ns. Implements the protocol hook used by tests; returns the hook result.
                ns[1] = 2
                return super().__new__(mcls, name, bases, ns)

        with self.assertWarnsRegex(RuntimeWarning, 'MyClass'):
            MyClass = meta('MyClass', (), {})

    def test_proxy_call(self):
        # [FUNCTION DOC] Test: checks proxy call behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        class FakeStr:
            # [CLASS DOC] HELPER: local class FakeStr defined inside test_proxy_call to exercise
            # descriptor/class-model behavior under test.
            __class__ = str

        fake_str = FakeStr()
        # isinstance() reads __class__
        self.assertIsInstance(fake_str, str)

        # call a method descriptor
        with self.assertRaises(TypeError):
            str.split(fake_str)

        # call a slot wrapper descriptor
        with self.assertRaises(TypeError):
            str.__add__(fake_str, "abc")

    def test_specialized_method_calls_check_types(self):
        # [FUNCTION DOC] Test: checks specialized method calls check types behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # https://github.com/python/cpython/issues/92063
        class Thing:
            # [CLASS DOC] HELPER: local class Thing defined inside
            # test_specialized_method_calls_check_types to exercise descriptor/class-model
            # behavior under test.
            pass
        thing = Thing()
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
                list.sort(thing)
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_FAST_WITH_KEYWORDS
                str.split(thing)
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_NOARGS
                str.upper(thing)
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_FAST
                str.strip(thing)
        from collections import deque
        for i in range(20):
            with self.assertRaises(TypeError):
                # CALL_METHOD_DESCRIPTOR_O
                deque.append(thing, thing)

    def test_repr_as_str(self):
        # [FUNCTION DOC] Test: checks repr as str behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Issue #11603: crash or infinite loop when rebinding __str__ as
        # __repr__.
        class Foo:
            # [CLASS DOC] HELPER: local class Foo defined inside test_repr_as_str to exercise
            # descriptor/class-model behavior under test.
            pass
        Foo.__repr__ = Foo.__str__
        foo = Foo()
        self.assertRaises(RecursionError, str, foo)
        self.assertRaises(RecursionError, repr, foo)

    def test_mixing_slot_wrappers(self):
        # [FUNCTION DOC] Test: checks mixing slot wrappers behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        class X(dict):
            # [CLASS DOC] HELPER: local class X defined inside test_mixing_slot_wrappers to
            # exercise descriptor/class-model behavior under test.
            __setattr__ = dict.__setitem__
            __neg__ = dict.copy
        x = X()
        x.y = 42
        self.assertEqual(x["y"], 42)
        self.assertEqual(x, -x)

    def test_wrong_class_slot_wrapper(self):
        # [FUNCTION DOC] Test: checks wrong class slot wrapper behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Check bpo-37619: a wrapper descriptor taken from the wrong class
        # should raise an exception instead of silently being ignored
        class A(int):
            # [CLASS DOC] HELPER: local class A defined inside test_wrong_class_slot_wrapper to
            # exercise descriptor/class-model behavior under test.
            __eq__ = str.__eq__
            __add__ = str.__add__
        a = A()
        with self.assertRaises(TypeError):
            a == a
        with self.assertRaises(TypeError):
            a + a

    def test_slot_shadows_class_variable(self):
        # [FUNCTION DOC] Test: checks slot shadows class variable behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        with self.assertRaises(ValueError) as cm:
            class X:
                # [CLASS DOC] HELPER: local class X defined inside
                # test_slot_shadows_class_variable to exercise descriptor/class-model behavior
                # under test.
                __slots__ = ["foo"]
                foo = None
        m = str(cm.exception)
        self.assertEqual("'foo' in __slots__ conflicts with class variable", m)

    def test_set_doc(self):
        # [FUNCTION DOC] Test: checks set doc behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        class X:
            # [CLASS DOC] HELPER: local class X defined inside test_set_doc to exercise
            # descriptor/class-model behavior under test. Existing docstring says: elephant
            "elephant"
        X.__doc__ = "banana"
        self.assertEqual(X.__doc__, "banana")

        with self.assertRaises(TypeError) as cm:
            type(list).__dict__["__doc__"].__set__(list, "blah")
        self.assertIn("cannot set '__doc__' attribute of immutable type 'list'", str(cm.exception))

        with self.assertRaises(TypeError) as cm:
            type(X).__dict__["__doc__"].__delete__(X)
        self.assertIn("cannot delete '__doc__' attribute of immutable type 'X'", str(cm.exception))
        self.assertEqual(X.__doc__, "banana")

    def test_qualname(self):
        # [FUNCTION DOC] Test: checks qualname behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        descriptors = [str.lower, complex.real, float.real, int.__add__]
        types = ['method', 'member', 'getset', 'wrapper']

        # make sure we have an example of each type of descriptor
        for d, n in zip(descriptors, types):
            self.assertEqual(type(d).__name__, n + '_descriptor')

        for d in descriptors:
            qualname = d.__objclass__.__qualname__ + '.' + d.__name__
            self.assertEqual(d.__qualname__, qualname)

        self.assertEqual(str.lower.__qualname__, 'str.lower')
        self.assertEqual(complex.real.__qualname__, 'complex.real')
        self.assertEqual(float.real.__qualname__, 'float.real')
        self.assertEqual(int.__add__.__qualname__, 'int.__add__')

        class X:
            # [CLASS DOC] HELPER: local class X defined inside test_qualname to exercise
            # descriptor/class-model behavior under test.
            pass
        with self.assertRaises(TypeError):
            del X.__qualname__

        self.assertRaises(TypeError, type.__dict__['__qualname__'].__set__,
                          str, 'Oink')

        global Y

        class Y:
            # [CLASS DOC] HELPER: local class Y defined inside test_qualname to exercise
            # descriptor/class-model behavior under test.
            class Inside:
                # [CLASS DOC] HELPER: nested helper class Inside inside class Y used by tests.
                pass
        self.assertEqual(Y.__qualname__, 'Y')
        self.assertEqual(Y.Inside.__qualname__, 'Y.Inside')

    def test_qualname_dict(self):
        # [FUNCTION DOC] Test: checks qualname dict behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        ns = {'__qualname__': 'some.name'}
        tp = type('Foo', (), ns)
        self.assertEqual(tp.__qualname__, 'some.name')
        self.assertNotIn('__qualname__', tp.__dict__)
        self.assertEqual(ns, {'__qualname__': 'some.name'})

        ns = {'__qualname__': 1}
        self.assertRaises(TypeError, type, 'Foo', (), ns)

    def test_cycle_through_dict(self):
        # [FUNCTION DOC] Test: checks cycle through dict behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # See bug #1469629
        class X(dict):
            # [CLASS DOC] HELPER: local class X defined inside test_cycle_through_dict to exercise
            # descriptor/class-model behavior under test.
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                dict.__init__(self)
                self.__dict__ = self
        x = X()
        x.attr = 42
        wr = weakref.ref(x)
        del x
        support.gc_collect()
        self.assertIsNone(wr())
        for o in gc.get_objects():
            self.assertIsNot(type(o), X)

    def test_object_new_and_init_with_parameters(self):
        # [FUNCTION DOC] Test: checks object new and init with parameters behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # See issue #1683368
        class OverrideNeither:
            # [CLASS DOC] HELPER: local class OverrideNeither defined inside
            # test_object_new_and_init_with_parameters to exercise descriptor/class-model behavior
            # under test.
            pass

        self.assertRaises(TypeError, OverrideNeither, 1)
        self.assertRaises(TypeError, OverrideNeither, kw=1)

        class OverrideNew:
            # [CLASS DOC] HELPER: local class OverrideNew defined inside
            # test_object_new_and_init_with_parameters to exercise descriptor/class-model behavior
            # under test.
            def __new__(cls, foo, kw=0, *args, **kwds):
                # [FUNCTION DOC] HELPER: special method __new__ of OverrideNew. Arguments: foo,
                # kw, *args, **kwds. Implements the protocol hook used by tests; returns the hook
                # result.
                return object.__new__(cls, *args, **kwds)
        class OverrideInit:
            # [CLASS DOC] HELPER: local class OverrideInit defined inside
            # test_object_new_and_init_with_parameters to exercise descriptor/class-model behavior
            # under test.
            def __init__(self, foo, kw=0, *args, **kwargs):
                # [FUNCTION DOC] HELPER: constructor. Arguments: foo, kw, *args, **kwargs. Stores
                # the given values on the instance, and for test cases initializes the base
                # CPythonTestCase.
                return object.__init__(self, *args, **kwargs)
        class OverrideBoth(OverrideNew, OverrideInit):
            # [CLASS DOC] HELPER: local class OverrideBoth defined inside
            # test_object_new_and_init_with_parameters to exercise descriptor/class-model behavior
            # under test.
            pass

        for case in OverrideNew, OverrideInit, OverrideBoth:
            case(1)
            case(1, kw=2)
            self.assertRaises(TypeError, case, 1, 2, 3)
            self.assertRaises(TypeError, case, 1, 2, foo=3)

    def test_subclassing_does_not_duplicate_dict_descriptors(self):
        # [FUNCTION DOC] Test: checks subclassing does not duplicate dict descriptors behavior.
        # Fails with AssertionError (via unittest assertions) if the observed behavior differs
        # from CPython semantics; no other side effects beyond temporary objects and classes.
        class Base:
            # [CLASS DOC] HELPER: local class Base defined inside
            # test_subclassing_does_not_duplicate_dict_descriptors to exercise descriptor/class-
            # model behavior under test.
            pass
        class Sub(Base):
            # [CLASS DOC] HELPER: local class Sub defined inside
            # test_subclassing_does_not_duplicate_dict_descriptors to exercise descriptor/class-
            # model behavior under test.
            pass
        self.assertIn("__dict__", Base.__dict__)
        self.assertNotIn("__dict__", Sub.__dict__)

    def test_bound_method_repr(self):
        # [FUNCTION DOC] Test: checks bound method repr behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        class Foo:
            # [CLASS DOC] HELPER: local class Foo defined inside test_bound_method_repr to
            # exercise descriptor/class-model behavior under test.
            def method(self):
                # [FUNCTION DOC] HELPER: method of Foo. Supporting routine for tests; returns a
                # computed value or None.
                pass
        self.assertRegex(repr(Foo().method),
            r"<bound method .*Foo\.method of <.*Foo object at .*>>")


        class Base:
            # [CLASS DOC] HELPER: local class Base defined inside test_bound_method_repr to
            # exercise descriptor/class-model behavior under test.
            def method(self):
                # [FUNCTION DOC] HELPER: method of Base. Supporting routine for tests; returns a
                # computed value or None.
                pass
        class Derived1(Base):
            # [CLASS DOC] HELPER: local class Derived1 defined inside test_bound_method_repr to
            # exercise descriptor/class-model behavior under test.
            pass
        class Derived2(Base):
            # [CLASS DOC] HELPER: local class Derived2 defined inside test_bound_method_repr to
            # exercise descriptor/class-model behavior under test.
            def method(self):
                # [FUNCTION DOC] HELPER: method of Derived2. Supporting routine for tests; returns
                # a computed value or None.
                pass
        base = Base()
        derived1 = Derived1()
        derived2 = Derived2()
        super_d2 = super(Derived2, derived2)
        self.assertRegex(repr(base.method),
            r"<bound method .*Base\.method of <.*Base object at .*>>")
        self.assertRegex(repr(derived1.method),
            r"<bound method .*Base\.method of <.*Derived1 object at .*>>")
        self.assertRegex(repr(derived2.method),
            r"<bound method .*Derived2\.method of <.*Derived2 object at .*>>")
        self.assertRegex(repr(super_d2.method),
            r"<bound method .*Base\.method of <.*Derived2 object at .*>>")

        class Foo:
            @classmethod
            # [CLASS DOC] HELPER: local class Foo defined inside test_bound_method_repr to
            # exercise descriptor/class-model behavior under test.
            def method(cls):
                # [FUNCTION DOC] HELPER: method of Foo. Supporting routine for tests; returns a
                # computed value or None.
                pass
        foo = Foo()
        self.assertRegex(repr(foo.method), # access via instance
            r"<bound method .*Foo\.method of <class '.*Foo'>>")
        self.assertRegex(repr(Foo.method), # access via the class
            r"<bound method .*Foo\.method of <class '.*Foo'>>")


        class MyCallable:
            # [CLASS DOC] HELPER: local class MyCallable defined inside test_bound_method_repr to
            # exercise descriptor/class-model behavior under test.
            def __call__(self, arg):
                # [FUNCTION DOC] HELPER: special method __call__ of MyCallable. Arguments: arg.
                # Implements the protocol hook used by tests; returns the hook result.
                pass
        func = MyCallable() # func has no __name__ or __qualname__ attributes
        instance = object()
        method = types.MethodType(func, instance)
        self.assertRegex(repr(method),
            r"<bound method \? of <object object at .*>>")
        func.__name__ = "name"
        self.assertRegex(repr(method),
            r"<bound method name of <object object at .*>>")
        func.__qualname__ = "qualname"
        self.assertRegex(repr(method),
            r"<bound method qualname of <object object at .*>>")

    @unittest.skipIf(_testcapi is None, 'need the _testcapi module')
    def test_bpo25750(self):
        # [FUNCTION DOC] Test: checks bpo25750 behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # bpo-25750: calling a descriptor (implemented as built-in
        # function with METH_FASTCALL) should not crash CPython if the
        # descriptor deletes itself from the class.
        class Descr:
            # [CLASS DOC] HELPER: local class Descr defined inside test_bpo25750 to exercise
            # descriptor/class-model behavior under test.
            __get__ = _testcapi.bad_get

        class X:
            # [CLASS DOC] HELPER: local class X defined inside test_bpo25750 to exercise
            # descriptor/class-model behavior under test.
            descr = Descr()
            def __new__(cls):
                # [FUNCTION DOC] HELPER: special method __new__ of X. Implements the protocol hook
                # used by tests; returns the hook result.
                cls.descr = None
                # Create this large list to corrupt some unused memory
                cls.lst = [2**i for i in range(10000)]
        X.descr

    @support.suppress_immortalization()
    def test_remove_subclass(self):
        # [FUNCTION DOC] Test: checks remove subclass behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # bpo-46417: when the last subclass of a type is deleted,
        # remove_subclass() clears the internal dictionary of subclasses:
        # set PyTypeObject.tp_subclasses to NULL. remove_subclass() is called
        # when a type is deallocated.
        class Parent:
            # [CLASS DOC] HELPER: local class Parent defined inside test_remove_subclass to
            # exercise descriptor/class-model behavior under test.
            pass
        self.assertEqual(Parent.__subclasses__(), [])

        class Child(Parent):
            # [CLASS DOC] HELPER: local class Child defined inside test_remove_subclass to
            # exercise descriptor/class-model behavior under test.
            pass
        self.assertEqual(Parent.__subclasses__(), [Child])

        del Child
        gc.collect()
        self.assertEqual(Parent.__subclasses__(), [])

    def test_instance_method_get_behavior(self):
        # [FUNCTION DOC] Test: checks instance method get behavior behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # test case for gh-113157

        class A:
            # [CLASS DOC] HELPER: local class A defined inside test_instance_method_get_behavior
            # to exercise descriptor/class-model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of A. Supporting routine for tests; returns a
                # computed value or None.
                return self

        class B:
            # [CLASS DOC] HELPER: local class B defined inside test_instance_method_get_behavior
            # to exercise descriptor/class-model behavior under test.
            pass

        a = A()
        b = B()
        b.meth = a.meth.__get__(b, B)
        self.assertEqual(b.meth(), a)

    def test_attr_raise_through_property(self):
        # [FUNCTION DOC] Test: checks attr raise through property behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # test case for gh-103272
        class A:
            # [CLASS DOC] HELPER: local class A defined inside test_attr_raise_through_property to
            # exercise descriptor/class-model behavior under test.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattr__ of A. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                raise ValueError("FOO")

            @property
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of A. Supporting routine for tests; returns a
                # computed value or None.
                return self.__getattr__("asdf")

        with self.assertRaisesRegex(ValueError, "FOO"):
            A().foo

        # test case for gh-103551
        class B:
            @property
            # [CLASS DOC] HELPER: local class B defined inside test_attr_raise_through_property to
            # exercise descriptor/class-model behavior under test.
            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: special method __getattr__ of B. Arguments: name.
                # Implements the protocol hook used by tests; returns the hook result.
                raise ValueError("FOO")

            @property
            def foo(self):
                # [FUNCTION DOC] HELPER: foo of B. Supporting routine for tests; returns a
                # computed value or None.
                raise NotImplementedError("BAR")

        with self.assertRaisesRegex(NotImplementedError, "BAR"):
            B().foo


class DictProxyTests(CPythonTestCase):
    # [CLASS DOC] Test case for mappingproxy behavior of class dictionaries, using a small fixture
    # class created in setUp.
    def setUp(self):
        # [FUNCTION DOC] HELPER: unittest fixture hook setUp; prepares or cleans state for each
        # test of the class.
        super().setUp()
        class C(object):
            # [CLASS DOC] HELPER: local class C defined inside setUp to exercise descriptor/class-
            # model behavior under test.
            def meth(self):
                # [FUNCTION DOC] HELPER: meth of C. Supporting routine for tests; returns a
                # computed value or None.
                pass
        self.C = C

    @unittest.skipIf(hasattr(sys, 'gettrace') and sys.gettrace(),
                        'trace function introduces __local__')
    def test_iter_keys(self):
        # [FUNCTION DOC] Test: checks iter keys behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing dict-proxy keys...
        it = self.C.__dict__.keys()
        self.assertNotIsInstance(it, list)
        keys = list(it)
        keys.sort()
        self.assertEqual(keys, ['__dict__', '__doc__', '__firstlineno__',
                                '__module__',
                                '__static_attributes__', '__weakref__',
                                'meth'])

    @unittest.skipIf(hasattr(sys, 'gettrace') and sys.gettrace(),
                        'trace function introduces __local__')
    def test_iter_values(self):
        # [FUNCTION DOC] Test: checks iter values behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing dict-proxy values...
        it = self.C.__dict__.values()
        self.assertNotIsInstance(it, list)
        values = list(it)
        self.assertEqual(len(values), 7)

    @unittest.skipIf(hasattr(sys, 'gettrace') and sys.gettrace(),
                        'trace function introduces __local__')
    def test_iter_items(self):
        # [FUNCTION DOC] Test: checks iter items behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing dict-proxy iteritems...
        it = self.C.__dict__.items()
        self.assertNotIsInstance(it, list)
        keys = [item[0] for item in it]
        keys.sort()
        self.assertEqual(keys, ['__dict__', '__doc__', '__firstlineno__',
                                '__module__',
                                '__static_attributes__', '__weakref__',
                                'meth'])

    def test_dict_type_with_metaclass(self):
        # [FUNCTION DOC] Test: checks dict type with metaclass behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        # Testing type of __dict__ when metaclass set...
        class B(object):
            # [CLASS DOC] HELPER: local class B defined inside test_dict_type_with_metaclass to
            # exercise descriptor/class-model behavior under test.
            pass
        class M(type):
            # [CLASS DOC] HELPER: local class M defined inside test_dict_type_with_metaclass to
            # exercise descriptor/class-model behavior under test.
            pass
        class C(metaclass=M):
            # [CLASS DOC] HELPER: local class C defined inside test_dict_type_with_metaclass to
            # exercise descriptor/class-model behavior under test.
            # In 2.3a1, C.__dict__ was a real dict rather than a dict proxy
            pass
        self.assertEqual(type(C.__dict__), type(B.__dict__))

    def test_repr(self):
        # [FUNCTION DOC] Test: checks repr behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Testing mappingproxy.__repr__.
        # We can't blindly compare with the repr of another dict as ordering
        # of keys and values is arbitrary and may differ.
        r = repr(self.C.__dict__)
        self.assertTrue(r.startswith('mappingproxy('), r)
        self.assertTrue(r.endswith(')'), r)
        for k, v in self.C.__dict__.items():
            self.assertIn('{!r}: {!r}'.format(k, v), r)


class AAAPTypesLongInitTest(CPythonTestCase):
    # [CLASS DOC] Test case checking that type initialization of a type with a long init chain
    # works (run first in module order).
    # This is in its own TestCase so that it can be run before any other tests.
    # (Hence the 'AAA' in the test class name: to make it the first
    # item in a list sorted by name, like
    # unittest.TestLoader.getTestCaseNames() does.)
    def test_pytype_long_ready(self):
        # [FUNCTION DOC] Test: checks pytype long ready behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Testing SF bug 551412 ...

        # This dumps core when SF bug 551412 isn't fixed --
        # but only when test_descr.py is run separately.
        # (That can't be helped -- as soon as PyType_Ready()
        # is called for PyLong_Type, the bug is gone.)
        class UserLong(object):
            # [CLASS DOC] HELPER: local class UserLong defined inside test_pytype_long_ready to
            # exercise descriptor/class-model behavior under test.
            def __pow__(self, *args):
                # [FUNCTION DOC] HELPER: special method __pow__ of UserLong. Arguments: *args.
                # Implements the protocol hook used by tests; returns the hook result.
                pass
        try:
            pow(0, UserLong(), 0)
        except:
            pass

        # Another segfault only when run early
        # (before PyType_Ready(tuple) is called)
        type.mro(tuple)


class MiscTests(CPythonTestCase):
    @unittest.skipIf(
        TEST_WITH_TORCHDYNAMO,
        "__build_class__ with closed over objects not supported",
    )
    # [CLASS DOC] Test case with miscellaneous checks on type names and type objects.
    def test_type_lookup_mro_reference(self):
        # [FUNCTION DOC] Test: checks type lookup mro reference behavior. Fails with
        # AssertionError (via unittest assertions) if the observed behavior differs from CPython
        # semantics; no other side effects beyond temporary objects and classes.
        # Issue #14199: _PyType_Lookup() has to keep a strong reference to
        # the type MRO because it may be modified during the lookup, if
        # __bases__ is set during the lookup for example.
        class MyKey(object):
            # [CLASS DOC] HELPER: local class MyKey defined inside test_type_lookup_mro_reference
            # to exercise descriptor/class-model behavior under test.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: special method __hash__ of MyKey. Implements the protocol
                # hook used by tests; returns the hook result.
                return hash('mykey')

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: special method __eq__ of MyKey. Arguments: other.
                # Implements the protocol hook used by tests; returns the hook result.
                X.__bases__ = (Base2,)

        class Base(object):
            # [CLASS DOC] HELPER: local class Base defined inside test_type_lookup_mro_reference
            # to exercise descriptor/class-model behavior under test.
            mykey = 'from Base'
            mykey2 = 'from Base'

        class Base2(object):
            # [CLASS DOC] HELPER: local class Base2 defined inside test_type_lookup_mro_reference
            # to exercise descriptor/class-model behavior under test.
            mykey = 'from Base2'
            mykey2 = 'from Base2'

        with self.assertWarnsRegex(RuntimeWarning, 'X'):
            X = type('X', (Base,), {MyKey(): 5})

        # Note that the access below uses getattr() rather than normally
        # accessing the attribute.  That is done to avoid the bytecode
        # specializer activating on repeated runs of the test.

        # mykey is read from Base
        self.assertEqual(getattr(X, 'mykey'), 'from Base')
        # mykey2 is read from Base2 because MyKey.__eq__ has set __bases__
        self.assertEqual(getattr(X, 'mykey2'), 'from Base2')


class PicklingTests(CPythonTestCase):
    # [CLASS DOC] Test case for pickling and copying of objects across protocols:
    # reduce/reduce_ex/getstate/newobj hooks, slots and copyreg interaction.

    def _check_reduce(self, proto, obj, args=(), kwargs={}, state=None,
                      listitems=None, dictitems=None):
        # [FUNCTION DOC] HELPER: _check_reduce of PicklingTests. Arguments: proto, obj, args,
        # kwargs, state, listitems, dictitems. Supporting routine for tests; returns a computed
        # value or None.
        if proto >= 2:
            reduce_value = obj.__reduce_ex__(proto)
            if kwargs:
                self.assertEqual(reduce_value[0], copyreg.__newobj_ex__)
                self.assertEqual(reduce_value[1], (type(obj), args, kwargs))
            else:
                self.assertEqual(reduce_value[0], copyreg.__newobj__)
                self.assertEqual(reduce_value[1], (type(obj),) + args)
            self.assertEqual(reduce_value[2], state)
            if listitems is not None:
                self.assertListEqual(list(reduce_value[3]), listitems)
            else:
                self.assertIsNone(reduce_value[3])
            if dictitems is not None:
                self.assertDictEqual(dict(reduce_value[4]), dictitems)
            else:
                self.assertIsNone(reduce_value[4])
        else:
            base_type = type(obj).__base__
            reduce_value = (copyreg._reconstructor,
                            (type(obj),
                             base_type,
                             None if base_type is object else base_type(obj)))
            if state is not None:
                reduce_value += (state,)
            self.assertEqual(obj.__reduce_ex__(proto), reduce_value)
            self.assertEqual(obj.__reduce__(), reduce_value)

    def test_reduce(self):
        # [FUNCTION DOC] Test: checks reduce behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        protocols = range(pickle.HIGHEST_PROTOCOL + 1)
        args = (-101, "spam")
        kwargs = {'bacon': -201, 'fish': -301}
        state = {'cheese': -401}

        class C1:
            # [CLASS DOC] HELPER: local class C1 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs__ of C1. Implements the
                # protocol hook used by tests; returns the hook result.
                return args
        obj = C1()
        for proto in protocols:
            self._check_reduce(proto, obj, args)

        for name, value in state.items():
            setattr(obj, name, value)
        for proto in protocols:
            self._check_reduce(proto, obj, args, state=state)

        class C2:
            # [CLASS DOC] HELPER: local class C2 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs__ of C2. Implements the
                # protocol hook used by tests; returns the hook result.
                return "bad args"
        obj = C2()
        for proto in protocols:
            if proto >= 2:
                with self.assertRaises(TypeError):
                    obj.__reduce_ex__(proto)

        class C3:
            # [CLASS DOC] HELPER: local class C3 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C3. Implements the
                # protocol hook used by tests; returns the hook result.
                return (args, kwargs)
        obj = C3()
        for proto in protocols:
            if proto >= 2:
                self._check_reduce(proto, obj, args, kwargs)

        class C4:
            # [CLASS DOC] HELPER: local class C4 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C4. Implements the
                # protocol hook used by tests; returns the hook result.
                return (args, "bad dict")
        class C5:
            # [CLASS DOC] HELPER: local class C5 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C5. Implements the
                # protocol hook used by tests; returns the hook result.
                return ("bad tuple", kwargs)
        class C6:
            # [CLASS DOC] HELPER: local class C6 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C6. Implements the
                # protocol hook used by tests; returns the hook result.
                return ()
        class C7:
            # [CLASS DOC] HELPER: local class C7 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C7. Implements the
                # protocol hook used by tests; returns the hook result.
                return "bad args"
        for proto in protocols:
            for cls in C4, C5, C6, C7:
                obj = cls()
                if proto >= 2:
                    with self.assertRaises((TypeError, ValueError)):
                        obj.__reduce_ex__(proto)

        class C9:
            # [CLASS DOC] HELPER: local class C9 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C9. Implements the
                # protocol hook used by tests; returns the hook result.
                return (args, {})
        obj = C9()
        for proto in protocols:
            self._check_reduce(proto, obj, args)

        class C10:
            # [CLASS DOC] HELPER: local class C10 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getnewargs_ex__(self):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C10. Implements the
                # protocol hook used by tests; returns the hook result.
                raise IndexError
        obj = C10()
        for proto in protocols:
            if proto >= 2:
                with self.assertRaises(IndexError):
                    obj.__reduce_ex__(proto)

        class C11:
            # [CLASS DOC] HELPER: local class C11 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: special method __getstate__ of C11. Implements the
                # protocol hook used by tests; returns the hook result.
                return state
        obj = C11()
        for proto in protocols:
            self._check_reduce(proto, obj, state=state)

        class C12:
            # [CLASS DOC] HELPER: local class C12 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: special method __getstate__ of C12. Implements the
                # protocol hook used by tests; returns the hook result.
                return "not dict"
        obj = C12()
        for proto in protocols:
            self._check_reduce(proto, obj, state="not dict")

        class C13:
            # [CLASS DOC] HELPER: local class C13 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: special method __getstate__ of C13. Implements the
                # protocol hook used by tests; returns the hook result.
                raise IndexError
        obj = C13()
        for proto in protocols:
            with self.assertRaises(IndexError):
                obj.__reduce_ex__(proto)
            if proto < 2:
                with self.assertRaises(IndexError):
                    obj.__reduce__()

        class C14:
            # [CLASS DOC] HELPER: local class C14 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            __slots__ = tuple(state)
            def __init__(self):
                # [FUNCTION DOC] HELPER: constructor. Stores the given values on the instance, and
                # for test cases initializes the base CPythonTestCase.
                for name, value in state.items():
                    setattr(self, name, value)

        obj = C14()
        for proto in protocols:
            if proto >= 2:
                self._check_reduce(proto, obj, state=(None, state))
            else:
                with self.assertRaises(TypeError):
                    obj.__reduce_ex__(proto)
                with self.assertRaises(TypeError):
                    obj.__reduce__()

        class C15(dict):
            # [CLASS DOC] HELPER: local class C15 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            pass
        obj = C15({"quebec": -601})
        for proto in protocols:
            self._check_reduce(proto, obj, dictitems=dict(obj))

        class C16(list):
            # [CLASS DOC] HELPER: local class C16 defined inside test_reduce to exercise
            # descriptor/class-model behavior under test.
            pass
        obj = C16(["yukon"])
        for proto in protocols:
            self._check_reduce(proto, obj, listitems=list(obj))

    def test_special_method_lookup(self):
        # [FUNCTION DOC] Test: checks special method lookup behavior. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        protocols = range(pickle.HIGHEST_PROTOCOL + 1)
        class Picky:
            # [CLASS DOC] HELPER: local class Picky defined inside test_special_method_lookup to
            # exercise descriptor/class-model behavior under test.
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: special method __getstate__ of Picky. Implements the
                # protocol hook used by tests; returns the hook result.
                return {}

            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: special method __getattr__ of Picky. Arguments: attr.
                # Implements the protocol hook used by tests; returns the hook result.
                if attr in ("__getnewargs__", "__getnewargs_ex__"):
                    raise AssertionError(attr)
                return None
        for protocol in protocols:
            state = {} if protocol >= 2 else None
            self._check_reduce(protocol, Picky(), state=state)

    def _assert_is_copy(self, obj, objcopy, msg=None):
        # [FUNCTION DOC] HELPER: _assert_is_copy of PicklingTests. Arguments: obj, objcopy, msg.
        # Supporting routine for tests; returns a computed value or None. Docstring: Utility
        # method to verify if two objects are copies of each others.
        """Utility method to verify if two objects are copies of each others.
        """
        if msg is None:
            msg = "{!r} is not a copy of {!r}".format(obj, objcopy)
        if type(obj).__repr__ is object.__repr__:
            # We have this limitation for now because we use the object's repr
            # to help us verify that the two objects are copies. This allows
            # us to delegate the non-generic verification logic to the objects
            # themselves.
            raise ValueError("object passed to _assert_is_copy must " +
                             "override the __repr__ method.")
        self.assertIsNot(obj, objcopy, msg=msg)
        self.assertIs(type(obj), type(objcopy), msg=msg)
        if hasattr(obj, '__dict__'):
            self.assertDictEqual(obj.__dict__, objcopy.__dict__, msg=msg)
            self.assertIsNot(obj.__dict__, objcopy.__dict__, msg=msg)
        if hasattr(obj, '__slots__'):
            self.assertListEqual(obj.__slots__, objcopy.__slots__, msg=msg)
            for slot in obj.__slots__:
                self.assertEqual(
                    hasattr(obj, slot), hasattr(objcopy, slot), msg=msg)
                self.assertEqual(getattr(obj, slot, None),
                                 getattr(objcopy, slot, None), msg=msg)
        self.assertEqual(repr(obj), repr(objcopy), msg=msg)

    @staticmethod
    def _generate_pickle_copiers():
        # [FUNCTION DOC] HELPER: _generate_pickle_copiers of PicklingTests. Supporting routine for
        # tests; returns a computed value or None. Docstring: Utility method to generate the many
        # possible pickle configurations.
        """Utility method to generate the many possible pickle configurations.
        """
        class PickleCopier:
            # [CLASS DOC] HELPER: local class PickleCopier defined inside _generate_pickle_copiers
            # to exercise descriptor/class-model behavior under test. Existing docstring says:
            # This class copies object using pickle.
            "This class copies object using pickle."
            def __init__(self, proto, dumps, loads):
                # [FUNCTION DOC] HELPER: constructor. Arguments: proto, dumps, loads. Stores the
                # given values on the instance, and for test cases initializes the base
                # CPythonTestCase.
                self.proto = proto
                self.dumps = dumps
                self.loads = loads
            def copy(self, obj):
                # [FUNCTION DOC] HELPER: copy of PickleCopier. Arguments: obj. Supporting routine
                # for tests; returns a computed value or None.
                return self.loads(self.dumps(obj, self.proto))
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of PickleCopier. Implements the
                # protocol hook used by tests; returns the hook result.
                # We try to be as descriptive as possible here since this is
                # the string which we will allow us to tell the pickle
                # configuration we are using during debugging.
                return ("PickleCopier(proto={}, dumps={}.{}, loads={}.{})"
                        .format(self.proto,
                                self.dumps.__module__, self.dumps.__qualname__,
                                self.loads.__module__, self.loads.__qualname__))
        return (PickleCopier(*args) for args in
                   itertools.product(range(pickle.HIGHEST_PROTOCOL + 1),
                                     {pickle.dumps, pickle._dumps},
                                     {pickle.loads, pickle._loads}))

    def test_pickle_slots(self):
        # [FUNCTION DOC] Test: checks pickle slots behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Tests pickling of classes with __slots__.

        # Pickling of classes with __slots__ but without __getstate__ should
        # fail (if using protocol 0 or 1)
        global C
        class C:
            # [CLASS DOC] HELPER: local class C defined inside test_pickle_slots to exercise
            # descriptor/class-model behavior under test.
            __slots__ = ['a']
        with self.assertRaises(TypeError):
            pickle.dumps(C(), 0)

        global D
        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_pickle_slots to exercise
            # descriptor/class-model behavior under test.
            pass
        with self.assertRaises(TypeError):
            pickle.dumps(D(), 0)

        class C:
            # [CLASS DOC] HELPER: local class C defined inside test_pickle_slots to exercise
            # descriptor/class-model behavior under test. Existing docstring says: A class with
            # __getstate__ and __setstate__ implemented.
            "A class with __getstate__ and __setstate__ implemented."
            __slots__ = ['a']
            def __getstate__(self):
                # [FUNCTION DOC] HELPER: special method __getstate__ of C. Implements the protocol
                # hook used by tests; returns the hook result.
                state = getattr(self, '__dict__', {}).copy()
                for cls in type(self).__mro__:
                    for slot in cls.__dict__.get('__slots__', ()):
                        try:
                            state[slot] = getattr(self, slot)
                        except AttributeError:
                            pass
                return state
            def __setstate__(self, state):
                # [FUNCTION DOC] HELPER: special method __setstate__ of C. Arguments: state.
                # Implements the protocol hook used by tests; returns the hook result.
                for k, v in state.items():
                    setattr(self, k, v)
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of C. Implements the protocol
                # hook used by tests; returns the hook result.
                return "%s()<%r>" % (type(self).__name__, self.__getstate__())

        class D(C):
            # [CLASS DOC] HELPER: local class D defined inside test_pickle_slots to exercise
            # descriptor/class-model behavior under test. Existing docstring says: A subclass of a
            # class with slots.
            "A subclass of a class with slots."
            pass

        global E
        class E(C):
            # [CLASS DOC] HELPER: local class E defined inside test_pickle_slots to exercise
            # descriptor/class-model behavior under test. Existing docstring says: A subclass with
            # an extra slot.
            "A subclass with an extra slot."
            __slots__ = ['b']

        # Now it should work
        for pickle_copier in self._generate_pickle_copiers():
            with self.subTest(pickle_copier=pickle_copier):
                x = C()
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

                x.a = 42
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

                x = D()
                x.a = 42
                x.b = 100
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

                x = E()
                x.a = 42
                x.b = "foo"
                y = pickle_copier.copy(x)
                self._assert_is_copy(x, y)

    def test_reduce_copying(self):
        # [FUNCTION DOC] Test: checks reduce copying behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Tests pickling and copying new-style classes and objects.
        global C1
        class C1:
            # [CLASS DOC] HELPER: local class C1 defined inside test_reduce_copying to exercise
            # descriptor/class-model behavior under test. Existing docstring says: The state of
            # this class is copyable via its instance dict.
            "The state of this class is copyable via its instance dict."
            ARGS = (1, 2)
            NEED_DICT_COPYING = True
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: constructor. Arguments: a, b. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                super().__init__()
                self.a = a
                self.b = b
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of C1. Implements the protocol
                # hook used by tests; returns the hook result.
                return "C1(%r, %r)" % (self.a, self.b)

        global C2
        class C2(list):
            # [CLASS DOC] HELPER: local class C2 defined inside test_reduce_copying to exercise
            # descriptor/class-model behavior under test. Existing docstring says: A list subclass
            # copyable via __getnewargs__.
            "A list subclass copyable via __getnewargs__."
            ARGS = (1, 2)
            NEED_DICT_COPYING = False
            def __new__(cls, a, b):
                # [FUNCTION DOC] HELPER: special method __new__ of C2. Arguments: a, b. Implements
                # the protocol hook used by tests; returns the hook result.
                self = super().__new__(cls)
                self.a = a
                self.b = b
                return self
            def __init__(self, *args):
                # [FUNCTION DOC] HELPER: constructor. Arguments: *args. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                super().__init__()
                # This helps testing that __init__ is not called during the
                # unpickling process, which would cause extra appends.
                self.append("cheese")
            @classmethod
            def __getnewargs__(cls):
                # [FUNCTION DOC] HELPER: special method __getnewargs__ of C2. Implements the
                # protocol hook used by tests; returns the hook result.
                return cls.ARGS
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of C2. Implements the protocol
                # hook used by tests; returns the hook result.
                return "C2(%r, %r)<%r>" % (self.a, self.b, list(self))

        global C3
        class C3(list):
            # [CLASS DOC] HELPER: local class C3 defined inside test_reduce_copying to exercise
            # descriptor/class-model behavior under test. Existing docstring says: A list subclass
            # copyable via __getstate__.
            "A list subclass copyable via __getstate__."
            ARGS = (1, 2)
            NEED_DICT_COPYING = False
            def __init__(self, a, b):
                # [FUNCTION DOC] HELPER: constructor. Arguments: a, b. Stores the given values on
                # the instance, and for test cases initializes the base CPythonTestCase.
                self.a = a
                self.b = b
                # This helps testing that __init__ is not called during the
                # unpickling process, which would cause extra appends.
                self.append("cheese")
            @classmethod
            def __getstate__(cls):
                # [FUNCTION DOC] HELPER: special method __getstate__ of C3. Implements the
                # protocol hook used by tests; returns the hook result.
                return cls.ARGS
            def __setstate__(self, state):
                # [FUNCTION DOC] HELPER: special method __setstate__ of C3. Arguments: state.
                # Implements the protocol hook used by tests; returns the hook result.
                a, b = state
                self.a = a
                self.b = b
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of C3. Implements the protocol
                # hook used by tests; returns the hook result.
                return "C3(%r, %r)<%r>" % (self.a, self.b, list(self))

        global C4
        class C4(int):
            # [CLASS DOC] HELPER: local class C4 defined inside test_reduce_copying to exercise
            # descriptor/class-model behavior under test. Existing docstring says: An int subclass
            # copyable via __getnewargs__.
            "An int subclass copyable via __getnewargs__."
            ARGS = ("hello", "world", 1)
            NEED_DICT_COPYING = False
            def __new__(cls, a, b, value):
                # [FUNCTION DOC] HELPER: special method __new__ of C4. Arguments: a, b, value.
                # Implements the protocol hook used by tests; returns the hook result.
                self = super().__new__(cls, value)
                self.a = a
                self.b = b
                return self
            @classmethod
            def __getnewargs__(cls):
                # [FUNCTION DOC] HELPER: special method __getnewargs__ of C4. Implements the
                # protocol hook used by tests; returns the hook result.
                return cls.ARGS
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of C4. Implements the protocol
                # hook used by tests; returns the hook result.
                return "C4(%r, %r)<%r>" % (self.a, self.b, int(self))

        global C5
        class C5(int):
            # [CLASS DOC] HELPER: local class C5 defined inside test_reduce_copying to exercise
            # descriptor/class-model behavior under test. Existing docstring says: An int subclass
            # copyable via __getnewargs_ex__.
            "An int subclass copyable via __getnewargs_ex__."
            ARGS = (1, 2)
            KWARGS = {'value': 3}
            NEED_DICT_COPYING = False
            def __new__(cls, a, b, *, value=0):
                # [FUNCTION DOC] HELPER: special method __new__ of C5. Arguments: a, b, value.
                # Implements the protocol hook used by tests; returns the hook result.
                self = super().__new__(cls, value)
                self.a = a
                self.b = b
                return self
            @classmethod
            def __getnewargs_ex__(cls):
                # [FUNCTION DOC] HELPER: special method __getnewargs_ex__ of C5. Implements the
                # protocol hook used by tests; returns the hook result.
                return (cls.ARGS, cls.KWARGS)
            def __repr__(self):
                # [FUNCTION DOC] HELPER: special method __repr__ of C5. Implements the protocol
                # hook used by tests; returns the hook result.
                return "C5(%r, %r)<%r>" % (self.a, self.b, int(self))

        test_classes = (C1, C2, C3, C4, C5)
        # Testing copying through pickle
        pickle_copiers = self._generate_pickle_copiers()
        for cls, pickle_copier in itertools.product(test_classes, pickle_copiers):
            with self.subTest(cls=cls, pickle_copier=pickle_copier):
                kwargs = getattr(cls, 'KWARGS', {})
                obj = cls(*cls.ARGS, **kwargs)
                proto = pickle_copier.proto
                objcopy = pickle_copier.copy(obj)
                self._assert_is_copy(obj, objcopy)
                # For test classes that supports this, make sure we didn't go
                # around the reduce protocol by simply copying the attribute
                # dictionary. We clear attributes using the previous copy to
                # not mutate the original argument.
                if proto >= 2 and not cls.NEED_DICT_COPYING:
                    objcopy.__dict__.clear()
                    objcopy2 = pickle_copier.copy(objcopy)
                    self._assert_is_copy(obj, objcopy2)

        # Testing copying through copy.deepcopy()
        for cls in test_classes:
            with self.subTest(cls=cls):
                kwargs = getattr(cls, 'KWARGS', {})
                obj = cls(*cls.ARGS, **kwargs)
                objcopy = deepcopy(obj)
                self._assert_is_copy(obj, objcopy)
                # For test classes that supports this, make sure we didn't go
                # around the reduce protocol by simply copying the attribute
                # dictionary. We clear attributes using the previous copy to
                # not mutate the original argument.
                if not cls.NEED_DICT_COPYING:
                    objcopy.__dict__.clear()
                    objcopy2 = deepcopy(objcopy)
                    self._assert_is_copy(obj, objcopy2)

    def test_issue24097(self):
        # [FUNCTION DOC] Test: checks issue24097 behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Slot name is freed inside __getattr__ and is later used.
        class S(str):  # Not interned
            # [CLASS DOC] HELPER: local class S defined inside test_issue24097 to exercise
            # descriptor/class-model behavior under test.
            pass
        class A:
            # [CLASS DOC] HELPER: local class A defined inside test_issue24097 to exercise
            # descriptor/class-model behavior under test.
            __slotnames__ = [S('spam')]
            def __getattr__(self, attr):
                # [FUNCTION DOC] HELPER: special method __getattr__ of A. Arguments: attr.
                # Implements the protocol hook used by tests; returns the hook result.
                if attr == 'spam':
                    A.__slotnames__[:] = [S('spam')]
                    return 42
                else:
                    raise AttributeError

        import copyreg
        expected = (copyreg.__newobj__, (A,), (None, {'spam': 42}), None, None)
        self.assertEqual(A().__reduce_ex__(2), expected)  # Shouldn't crash

    def test_object_reduce(self):
        # [FUNCTION DOC] Test: checks object reduce behavior. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        # Issue #29914
        # __reduce__() takes no arguments
        object().__reduce__()
        with self.assertRaises(TypeError):
            object().__reduce__(0)
        # __reduce_ex__() takes one integer argument
        object().__reduce_ex__(0)
        with self.assertRaises(TypeError):
            object().__reduce_ex__()
        with self.assertRaises(TypeError):
            object().__reduce_ex__(None)


class SharedKeyTests(CPythonTestCase):

    @support.cpython_only
    # [CLASS DOC] Test case for key-sharing instance dictionaries: verifies instance dicts share
    # keys and sizes behave as expected.
    def test_subclasses(self):
        # [FUNCTION DOC] Test: checks subclasses behavior. Fails with AssertionError (via unittest
        # assertions) if the observed behavior differs from CPython semantics; no other side
        # effects beyond temporary objects and classes.
        # Verify that subclasses can share keys (per PEP 412)
        class A:
            # [CLASS DOC] HELPER: local class A defined inside test_subclasses to exercise
            # descriptor/class-model behavior under test.
            pass
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_subclasses to exercise
            # descriptor/class-model behavior under test.
            pass

        #Shrink keys by repeatedly creating instances
        [(A(), B()) for _ in range(30)]

        a, b = A(), B()
        self.assertEqual(sys.getsizeof(vars(a)), sys.getsizeof(vars(b)))
        self.assertLess(sys.getsizeof(vars(a)), sys.getsizeof({"a":1}))
        # Initial hash table can contain only one or two elements.
        # Set 6 attributes to cause internal resizing.
        a.x, a.y, a.z, a.w, a.v, a.u = range(6)
        self.assertNotEqual(sys.getsizeof(vars(a)), sys.getsizeof(vars(b)))
        a2 = A()
        self.assertGreater(sys.getsizeof(vars(a)), sys.getsizeof(vars(a2)))
        self.assertLess(sys.getsizeof(vars(a2)), sys.getsizeof({"a":1}))
        self.assertLess(sys.getsizeof(vars(b)), sys.getsizeof({"a":1}))


class DebugHelperMeta(type):
    # [CLASS DOC] HELPER: metaclass that wraps class creation, setting and deletion of attributes
    # to debug type attribute caches. Existing docstring says: Sets default __doc__ and simplifies
    # repr() output.
    """
    Sets default __doc__ and simplifies repr() output.
    """
    def __new__(mcls, name, bases, attrs):
        # [FUNCTION DOC] HELPER: special method __new__ of DebugHelperMeta. Arguments: mcls, name,
        # bases, attrs. Implements the protocol hook used by tests; returns the hook result.
        if attrs.get('__doc__') is None:
            attrs['__doc__'] = name  # helps when debugging with gdb
        return type.__new__(mcls, name, bases, attrs)
    def __repr__(cls):
        # [FUNCTION DOC] HELPER: special method __repr__ of DebugHelperMeta. Implements the
        # protocol hook used by tests; returns the hook result.
        return repr(cls.__name__)


class MroTest(CPythonTestCase):
    # [CLASS DOC] Test case for method resolution order handling: custom mro methods, invalid mro
    # results, and mutation of bases. Existing docstring says: Regressions for some bugs revealed
    # through mcsl.mro() customization (typeobject.c: mro_internal()) and cls.__bases__ assignment
    # (typeobject.c: type_set_bases()).
    """
    Regressions for some bugs revealed through
    mcsl.mro() customization (typeobject.c: mro_internal()) and
    cls.__bases__ assignment (typeobject.c: type_set_bases()).
    """

    def setUp(self):
        # [FUNCTION DOC] HELPER: unittest fixture hook setUp; prepares or cleans state for each
        # test of the class.
        super().setUp()
        self.step = 0
        self.ready = False

    def step_until(self, limit):
        # [FUNCTION DOC] HELPER: step_until of MroTest. Arguments: limit. Supporting routine for
        # tests; returns a computed value or None.
        ret = (self.step < limit)
        if ret:
            self.step += 1
        return ret

    def test_incomplete_set_bases_on_self(self):
        # [FUNCTION DOC] Test: checks incomplete set bases on self behavior. Docstring:
        # type_set_bases must be aware that type->tp_mro can be NULL. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        """
        type_set_bases must be aware that type->tp_mro can be NULL.
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_incomplete_set_bases_on_self
            # to exercise descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if self.step_until(1):
                    assert cls.__mro__ is None
                    cls.__bases__ += ()

                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside test_incomplete_set_bases_on_self
            # to exercise descriptor/class-model behavior under test.
            pass

    def test_reent_set_bases_on_base(self):
        # [FUNCTION DOC] Test: checks reent set bases on base behavior. Docstring: Deep reentrancy
        # must not over-decref old_mro. Fails with AssertionError (via unittest assertions) if the
        # observed behavior differs from CPython semantics; no other side effects beyond temporary
        # objects and classes.
        """
        Deep reentrancy must not over-decref old_mro.
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_reent_set_bases_on_base to
            # exercise descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if cls.__mro__ is not None and cls.__name__ == 'B':
                    # 4-5 steps are usually enough to make it crash somewhere
                    if self.step_until(10):
                        A.__bases__ += ()

                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside test_reent_set_bases_on_base to
            # exercise descriptor/class-model behavior under test.
            pass
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_reent_set_bases_on_base to
            # exercise descriptor/class-model behavior under test.
            pass
        B.__bases__ += ()

    def test_reent_set_bases_on_direct_base(self):
        # [FUNCTION DOC] Test: checks reent set bases on direct base behavior. Docstring: Similar
        # to test_reent_set_bases_on_base, but may crash differently. Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        """
        Similar to test_reent_set_bases_on_base, but may crash differently.
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_reent_set_bases_on_direct_base
            # to exercise descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                base = cls.__bases__[0]
                if base is not object:
                    if self.step_until(5):
                        base.__bases__ += ()

                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside test_reent_set_bases_on_direct_base
            # to exercise descriptor/class-model behavior under test.
            pass
        class B(A):
            # [CLASS DOC] HELPER: local class B defined inside test_reent_set_bases_on_direct_base
            # to exercise descriptor/class-model behavior under test.
            pass
        class C(B):
            # [CLASS DOC] HELPER: local class C defined inside test_reent_set_bases_on_direct_base
            # to exercise descriptor/class-model behavior under test.
            pass

    def test_reent_set_bases_tp_base_cycle(self):
        # [FUNCTION DOC] Test: checks reent set bases tp base cycle behavior. Docstring:
        # type_set_bases must check for an inheritance cycle not only through MRO of the type,
        # which may be not yet updated in case of reentrance, but also through tp_base chain,
        # which is assigned before diving into inner calls to mro(). Fails with AssertionError
        # (via unittest assertions) if the observed behavior differs from CPython semantics; no
        # other side effects beyond temporary objects and classes.
        """
        type_set_bases must check for an inheritance cycle not only through
        MRO of the type, which may be not yet updated in case of reentrance,
        but also through tp_base chain, which is assigned before diving into
        inner calls to mro().

        Otherwise, the following snippet can loop forever:
            do {
                // ...
                type = type->tp_base;
            } while (type != NULL);

        Functions that rely on tp_base (like solid_base and PyType_IsSubtype)
        would not be happy in that case, causing a stack overflow.
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_reent_set_bases_tp_base_cycle
            # to exercise descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if self.ready:
                    if cls.__name__ == 'B1':
                        B2.__bases__ = (B1,)
                    if cls.__name__ == 'B2':
                        B1.__bases__ = (B2,)
                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside test_reent_set_bases_tp_base_cycle
            # to exercise descriptor/class-model behavior under test.
            pass
        class B1(A):
            # [CLASS DOC] HELPER: local class B1 defined inside test_reent_set_bases_tp_base_cycle
            # to exercise descriptor/class-model behavior under test.
            pass
        class B2(A):
            # [CLASS DOC] HELPER: local class B2 defined inside test_reent_set_bases_tp_base_cycle
            # to exercise descriptor/class-model behavior under test.
            pass

        self.ready = True
        with self.assertRaises(TypeError):
            B1.__bases__ += ()

    def test_tp_subclasses_cycle_in_update_slots(self):
        # [FUNCTION DOC] Test: checks tp subclasses cycle in update slots behavior. Docstring:
        # type_set_bases must check for reentrancy upon finishing its job by updating
        # tp_subclasses of old/new bases of the type. Otherwise, an implicit inheritance cycle
        # through tp_subclasses can break functions that recurse on elements of that field (like
        # recurse_down_subclasses and mro_hierarchy) eventually leading to a stack overflow. Fails
        # with AssertionError (via unittest assertions) if the observed behavior differs from
        # CPython semantics; no other side effects beyond temporary objects and classes.
        """
        type_set_bases must check for reentrancy upon finishing its job
        by updating tp_subclasses of old/new bases of the type.
        Otherwise, an implicit inheritance cycle through tp_subclasses
        can break functions that recurse on elements of that field
        (like recurse_down_subclasses and mro_hierarchy) eventually
        leading to a stack overflow.
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside
            # test_tp_subclasses_cycle_in_update_slots to exercise descriptor/class-model behavior
            # under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if self.ready and cls.__name__ == 'C':
                    self.ready = False
                    C.__bases__ = (B2,)
                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside
            # test_tp_subclasses_cycle_in_update_slots to exercise descriptor/class-model behavior
            # under test.
            pass
        class B1(A):
            # [CLASS DOC] HELPER: local class B1 defined inside
            # test_tp_subclasses_cycle_in_update_slots to exercise descriptor/class-model behavior
            # under test.
            pass
        class B2(A):
            # [CLASS DOC] HELPER: local class B2 defined inside
            # test_tp_subclasses_cycle_in_update_slots to exercise descriptor/class-model behavior
            # under test.
            pass
        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside
            # test_tp_subclasses_cycle_in_update_slots to exercise descriptor/class-model behavior
            # under test.
            pass

        self.ready = True
        C.__bases__ = (B1,)
        B1.__bases__ = (C,)

        self.assertEqual(C.__bases__, (B2,))
        self.assertEqual(B2.__subclasses__(), [C])
        self.assertEqual(B1.__subclasses__(), [])

        self.assertEqual(B1.__bases__, (C,))
        self.assertEqual(C.__subclasses__(), [B1])

    def test_tp_subclasses_cycle_error_return_path(self):
        # [FUNCTION DOC] Test: checks tp subclasses cycle error return path behavior. Docstring:
        # The same as test_tp_subclasses_cycle_in_update_slots, but tests a code path executed on
        # error (goto bail). Fails with AssertionError (via unittest assertions) if the observed
        # behavior differs from CPython semantics; no other side effects beyond temporary objects
        # and classes.
        """
        The same as test_tp_subclasses_cycle_in_update_slots, but tests
        a code path executed on error (goto bail).
        """
        class E(Exception):
            # [CLASS DOC] HELPER: local class E defined inside
            # test_tp_subclasses_cycle_error_return_path to exercise descriptor/class-model
            # behavior under test.
            pass
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside
            # test_tp_subclasses_cycle_error_return_path to exercise descriptor/class-model
            # behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if self.ready and cls.__name__ == 'C':
                    if C.__bases__ == (B2,):
                        self.ready = False
                    else:
                        C.__bases__ = (B2,)
                        raise E
                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside
            # test_tp_subclasses_cycle_error_return_path to exercise descriptor/class-model
            # behavior under test.
            pass
        class B1(A):
            # [CLASS DOC] HELPER: local class B1 defined inside
            # test_tp_subclasses_cycle_error_return_path to exercise descriptor/class-model
            # behavior under test.
            pass
        class B2(A):
            # [CLASS DOC] HELPER: local class B2 defined inside
            # test_tp_subclasses_cycle_error_return_path to exercise descriptor/class-model
            # behavior under test.
            pass
        class C(A):
            # [CLASS DOC] HELPER: local class C defined inside
            # test_tp_subclasses_cycle_error_return_path to exercise descriptor/class-model
            # behavior under test.
            pass

        self.ready = True
        with self.assertRaises(E):
            C.__bases__ = (B1,)
        B1.__bases__ = (C,)

        self.assertEqual(C.__bases__, (B2,))
        self.assertEqual(C.__mro__, tuple(type.mro(C)))

    def test_incomplete_extend(self):
        # [FUNCTION DOC] Test: checks incomplete extend behavior. Docstring: Extending an
        # uninitialized type with type->tp_mro == NULL must throw a reasonable TypeError
        # exception, instead of failing with PyErr_BadInternalCall. Fails with AssertionError (via
        # unittest assertions) if the observed behavior differs from CPython semantics; no other
        # side effects beyond temporary objects and classes.
        """
        Extending an uninitialized type with type->tp_mro == NULL must
        throw a reasonable TypeError exception, instead of failing
        with PyErr_BadInternalCall.
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_incomplete_extend to exercise
            # descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if cls.__mro__ is None and cls.__name__ != 'X':
                    with self.assertRaises(TypeError):
                        class X(cls):
                            # [CLASS DOC] HELPER: local class X defined inside mro to exercise
                            # descriptor/class-model behavior under test.
                            pass

                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside test_incomplete_extend to exercise
            # descriptor/class-model behavior under test.
            pass

    def test_incomplete_super(self):
        # [FUNCTION DOC] Test: checks incomplete super behavior. Docstring: Attribute lookup on a
        # super object must be aware that its target type can be uninitialized (type->tp_mro ==
        # NULL). Fails with AssertionError (via unittest assertions) if the observed behavior
        # differs from CPython semantics; no other side effects beyond temporary objects and
        # classes.
        """
        Attribute lookup on a super object must be aware that
        its target type can be uninitialized (type->tp_mro == NULL).
        """
        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_incomplete_super to exercise
            # descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                if cls.__mro__ is None:
                    with self.assertRaises(AttributeError):
                        super(cls, cls).xxx

                return type.mro(cls)

        class A(metaclass=M):
            # [CLASS DOC] HELPER: local class A defined inside test_incomplete_super to exercise
            # descriptor/class-model behavior under test.
            pass

    def test_disappearing_custom_mro(self):
        # [FUNCTION DOC] Test: checks disappearing custom mro behavior. Docstring: gh-92112: A
        # custom mro() returning a result conflicting with __bases__ and deleting itself caused a
        # double free. Fails with AssertionError (via unittest assertions) if the observed
        # behavior differs from CPython semantics; no other side effects beyond temporary objects
        # and classes.
        """
        gh-92112: A custom mro() returning a result conflicting with
        __bases__ and deleting itself caused a double free.
        """
        class B:
            # [CLASS DOC] HELPER: local class B defined inside test_disappearing_custom_mro to
            # exercise descriptor/class-model behavior under test.
            pass

        class M(DebugHelperMeta):
            # [CLASS DOC] HELPER: local class M defined inside test_disappearing_custom_mro to
            # exercise descriptor/class-model behavior under test.
            def mro(cls):
                # [FUNCTION DOC] HELPER: mro of M. Supporting routine for tests; returns a
                # computed value or None.
                del M.mro
                return (B,)

        with self.assertRaises(TypeError):
            class A(metaclass=M):
                # [CLASS DOC] HELPER: local class A defined inside test_disappearing_custom_mro to
                # exercise descriptor/class-model behavior under test.
                pass


if __name__ == "__main__":
    run_tests()
