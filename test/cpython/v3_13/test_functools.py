# [MODULE DOC] CPython 3.13 Lib/test/test_functools.py adapted for the Dynamo CPython suite.
# Exercises the functools module in both its C (_functools) and pure-Python implementations:
# partial, partialmethod, update_wrapper/wraps, reduce, cmp_to_key, total_ordering, cache,
# lru_cache, singledispatch and friends, cached_property, and related helpers. Test classes derive
# from CPythonTestCase; mixin classes (names starting with an underscore) hold shared tests and
# are bound to a concrete implementation by subclasses setting class attributes. Import-time side
# effects: imports fresh copies of functools (one with the C module blocked, one with it forced)
# and of decimal via import_helper, and imports CPython's test.support helpers. When the C module
# is unavailable, C-specific test classes are skipped.
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_functools.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import run_tests, TEST_WITH_TORCHDYNAMO

# ======= END DYNAMO PATCH =======


import abc
import builtins
import collections
import collections.abc
import copy
from itertools import permutations
import pickle
from random import choice
import sys
from test import support
import threading
import time
import typing
import unittest
import unittest.mock
import weakref
import gc
from weakref import proxy
import contextlib
from inspect import Signature

from test.support import import_helper
from test.support import threading_helper

import functools

py_functools = import_helper.import_fresh_module('functools',
                                                 blocked=['_functools'])
c_functools = import_helper.import_fresh_module('functools',
                                                fresh=['_functools'])

decimal = import_helper.import_fresh_module('decimal', fresh=['_decimal'])


@contextlib.contextmanager
def replaced_module(name, replacement):
    # [FUNCTION DOC] HELPER: Context manager that temporarily installs a replacement module in
    # sys.modules under the given name and restores the original on exit (even on error). Used so
    # pickling by reference resolves to the implementation under test.
    original_module = sys.modules[name]
    sys.modules[name] = replacement
    try:
        yield
    finally:
        sys.modules[name] = original_module

def capture(*args, **kw):
    # [FUNCTION DOC] HELPER: Test callable returning a tuple of the positional arguments and the
    # keyword-argument mapping it was called with, to observe how partial combines arguments.
    """capture all positional and keyword arguments"""
    return args, kw


def signature(part):
    # [FUNCTION DOC] HELPER: Returns a tuple describing a partial object: its func, args, keywords
    # and instance __dict__, used to compare partials structurally.
    """ return the signature of a partial object """
    return (part.func, part.args, part.keywords, part.__dict__)

class MyTuple(tuple):
    # [CLASS DOC] HELPER: Plain tuple subclass used to check partial normalizes its state.
    pass

class BadTuple(tuple):
    # [CLASS DOC] HELPER: Tuple subclass whose concatenation returns a list, used to check that
    # partial does not use the subclass's addition when merging arguments.
    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Concatenation override returning a list of both operands' items.
        return list(self) + list(other)

class MyDict(dict):
    # [CLASS DOC] HELPER: Plain dict subclass used to check partial normalizes keyword state.
    pass


class _TestPartial:
    # [CLASS DOC] Mixin with tests shared by the C and Python partial implementations. Concrete
    # subclasses must set the class attributes partial (the class under test) and module (the
    # functools implementation module used for pickling). Holds no instance state.

    def test_basic_examples(self):
        # [FUNCTION DOC] Checks basic partial use: frozen positional and keyword args combine with
        # call-time args, and a partial of map works.
        p = self.partial(capture, 1, 2, a=10, b=20)
        self.assertTrue(callable(p))
        self.assertEqual(p(3, 4, b=30, c=40),
                         ((1, 2, 3, 4), dict(a=10, b=30, c=40)))
        p = self.partial(map, lambda x: x*10)
        self.assertEqual(list(p([1,2,3,4])), [10, 20, 30, 40])

    def test_attributes(self):
        # [FUNCTION DOC] Checks the func, args and keywords attributes are readable and correct.
        p = self.partial(capture, 1, 2, a=10, b=20)
        # attributes should be readable
        self.assertEqual(p.func, capture)
        self.assertEqual(p.args, (1, 2))
        self.assertEqual(p.keywords, dict(a=10, b=20))

    def test_argument_checking(self):
        # [FUNCTION DOC] Checks partial() with no arguments raises TypeError and that a
        # non-callable first argument is only rejected (TypeError) when the partial is called.
        self.assertRaises(TypeError, self.partial)     # need at least a func arg
        try:
            self.partial(2)()
        except TypeError:
            pass
        else:
            self.fail('First arg not checked for callability')

    def test_protection_of_callers_dict_argument(self):
        # [FUNCTION DOC] Checks calling a partial with a keyword dict does not mutate the caller's
        # dictionary and that call-time keywords override frozen ones.
        # a caller's dictionary should not be altered by partial
        def func(a=10, b=20):
            # [FUNCTION DOC] HELPER: Returns its argument a (default 10); b is unused.
            return a
        d = {'a':3}
        p = self.partial(func, a=5)
        self.assertEqual(p(**d), 3)
        self.assertEqual(d, {'a':3})
        p(b=7)
        self.assertEqual(d, {'a':3})

    def test_kwargs_copy(self):
        # [FUNCTION DOC] Checks the keywords dict passed at construction is copied, so later
        # changes to the caller's dict do not affect the partial (issue 29532).
        # Issue #29532: Altering a kwarg dictionary passed to a constructor
        # should not affect a partial object after creation
        d = {'a': 3}
        p = self.partial(capture, **d)
        self.assertEqual(p(), ((), {'a': 3}))
        d['a'] = 5
        self.assertEqual(p(), ((), {'a': 3}))

    def test_arg_combinations(self):
        # [FUNCTION DOC] Checks calls with zero frozen args and/or zero call args
        # combine correctly.
        # exercise special code paths for zero args in either partial
        # object or the caller
        p = self.partial(capture)
        self.assertEqual(p(), ((), {}))
        self.assertEqual(p(1,2), ((1,2), {}))
        p = self.partial(capture, 1, 2)
        self.assertEqual(p(), ((1,2), {}))
        self.assertEqual(p(3,4), ((1,2,3,4), {}))

    def test_kw_combinations(self):
        # [FUNCTION DOC] Checks empty/non-empty keyword combinations and that call-time keywords
        # override frozen ones.
        # exercise special code paths for no keyword args in
        # either the partial object or the caller
        p = self.partial(capture)
        self.assertEqual(p.keywords, {})
        self.assertEqual(p(), ((), {}))
        self.assertEqual(p(a=1), ((), {'a':1}))
        p = self.partial(capture, a=1)
        self.assertEqual(p.keywords, {'a':1})
        self.assertEqual(p(), ((), {'a':1}))
        self.assertEqual(p(b=2), ((), {'a':1, 'b':2}))
        # keyword args in the call override those in the partial object
        self.assertEqual(p(a=3, b=2), ((), {'a':3, 'b':2}))

    def test_positional(self):
        # [FUNCTION DOC] Checks positional args of 0 to 4 frozen items precede the call-time arg.
        # make sure positional arguments are captured correctly
        for args in [(), (0,), (0,1), (0,1,2), (0,1,2,3)]:
            p = self.partial(capture, *args)
            expected = args + ('x',)
            got, empty = p('x')
            self.assertTrue(expected == got and empty == {})

    def test_keyword(self):
        # [FUNCTION DOC] Checks frozen keyword values of several types are passed as keywords with
        # no extra positional args.
        # make sure keyword arguments are captured correctly
        for a in ['a', 0, None, 3.5]:
            p = self.partial(capture, a=a)
            expected = {'a':a,'x':None}
            empty, got = p(x=None)
            self.assertTrue(expected == got and empty == ())

    def test_no_side_effects(self):
        # [FUNCTION DOC] Checks calling a partial does not alter its frozen args or keywords, so a
        # later call sees only the original frozen values.
        # make sure there are no side effects that affect subsequent calls
        p = self.partial(capture, 0, a=1)
        args1, kw1 = p(1, b=2)
        self.assertTrue(args1 == (0,1) and kw1 == {'a':1,'b':2})
        args2, kw2 = p()
        self.assertTrue(args2 == (0,) and kw2 == {'a':1})

    def test_error_propagation(self):
        # [FUNCTION DOC] Checks exceptions from the wrapped function (ZeroDivisionError) propagate
        # for several splits of frozen versus call-time arguments.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Computes x divided by y (raises ZeroDivisionError when y
            # is zero).
            x / y
        self.assertRaises(ZeroDivisionError, self.partial(f, 1, 0))
        self.assertRaises(ZeroDivisionError, self.partial(f, 1), 0)
        self.assertRaises(ZeroDivisionError, self.partial(f), 1, 0)
        self.assertRaises(ZeroDivisionError, self.partial(f, y=0), 1)

    def test_weakref(self):
        # [FUNCTION DOC] Checks a partial supports weak references: a proxy works while the
        # partial lives and raises ReferenceError after it is collected.
        f = self.partial(int, base=16)
        p = proxy(f)
        self.assertEqual(f.func, p.func)
        f = None
        support.gc_collect()  # For PyPy or other GCs.
        self.assertRaises(ReferenceError, getattr, p, 'func')

    def test_with_bound_and_unbound_methods(self):
        # [FUNCTION DOC] Checks partial works with an unbound method (str.join) and a
        # bound method.
        data = list(map(str, range(10)))
        join = self.partial(str.join, '')
        self.assertEqual(join(data), '0123456789')
        join = self.partial(''.join)
        self.assertEqual(join(data), '0123456789')

    def test_nested_optimization(self):
        # [FUNCTION DOC] Checks a partial of a partial is flattened: nested and flat construction
        # give the same func, args, keywords and dict.
        partial = self.partial
        inner = partial(signature, 'asdf')
        nested = partial(inner, bar=True)
        flat = partial(signature, 'asdf', bar=True)
        self.assertEqual(signature(nested), signature(flat))

    def test_nested_optimization_bug(self):
        # [FUNCTION DOC] Regression test: a partial whose func is an object with a dynamic
        # attribute lookup (returning partials of itself) must not be wrongly flattened; calls
        # return the tag, children and attributes as expected.
        partial = self.partial
        class Builder:
            # [CLASS DOC] HELPER: Callable builder; calling returns a tuple (tag, children,
            # attributes) and any attribute access returns a partial of the builder bound to
            # that tag.
            def __call__(self, tag, *children, **attrib):
                # [FUNCTION DOC] HELPER: Returns a tuple of the tag, the positional children and
                # the keyword attributes.
                return (tag, children, attrib)

            def __getattr__(self, tag):
                # [FUNCTION DOC] HELPER: Returns a partial of this builder with the attribute name
                # as tag.
                return partial(self, tag)

        B = Builder()
        m = B.m
        assert m(1, 2, a=2) == ('m', (1, 2), dict(a=2))

    def test_nested_partial_with_attribute(self):
        # [FUNCTION DOC] Checks that an attribute set on a nested partial (issue 25137) is kept.
        # see issue 25137
        partial = self.partial

        def foo(bar):
            # [FUNCTION DOC] HELPER: Returns its argument.
            return bar

        p = partial(foo, 'first')
        p2 = partial(p, 'second')
        p2.new_attr = 'spam'
        self.assertEqual(p2.new_attr, 'spam')

    def test_repr(self):
        # [FUNCTION DOC] Checks repr of a partial: the qualified class name, the repr of func,
        # then positional args and keyword args (either keyword order accepted).
        args = (object(), object())
        args_repr = ', '.join(repr(a) for a in args)
        kwargs = {'a': object(), 'b': object()}
        kwargs_reprs = ['a={a!r}, b={b!r}'.format_map(kwargs),
                        'b={b!r}, a={a!r}'.format_map(kwargs)]
        name = f"{self.partial.__module__}.{self.partial.__qualname__}"

        f = self.partial(capture)
        self.assertEqual(f'{name}({capture!r})', repr(f))

        f = self.partial(capture, *args)
        self.assertEqual(f'{name}({capture!r}, {args_repr})', repr(f))

        f = self.partial(capture, **kwargs)
        self.assertIn(repr(f),
                      [f'{name}({capture!r}, {kwargs_repr})'
                       for kwargs_repr in kwargs_reprs])

        f = self.partial(capture, *args, **kwargs)
        self.assertIn(repr(f),
                      [f'{name}({capture!r}, {args_repr}, {kwargs_repr})'
                       for kwargs_repr in kwargs_reprs])

    def test_recursive_repr(self):
        # [FUNCTION DOC] Checks repr of self-referential partials (via __setstate__) uses an
        # ellipsis for the cycle in func, args and keywords positions; resets the state in a
        # finally block.
        name = f"{self.partial.__module__}.{self.partial.__qualname__}"

        f = self.partial(capture)
        f.__setstate__((f, (), {}, {}))
        try:
            self.assertEqual(repr(f), '%s(...)' % (name,))
        finally:
            f.__setstate__((capture, (), {}, {}))

        f = self.partial(capture)
        f.__setstate__((capture, (f,), {}, {}))
        try:
            self.assertEqual(repr(f), '%s(%r, ...)' % (name, capture,))
        finally:
            f.__setstate__((capture, (), {}, {}))

        f = self.partial(capture)
        f.__setstate__((capture, (), {'a': f}, {}))
        try:
            self.assertEqual(repr(f), '%s(%r, a=...)' % (name, capture,))
        finally:
            f.__setstate__((capture, (), {}, {}))

    def test_pickle(self):
        # [FUNCTION DOC] Checks a partial with an instance attribute round-trips through
        # pickle for every protocol, with equal signature. Temporarily swaps the functools
        # module in sys.modules.
        with replaced_module('functools', self.module):
            f = self.partial(signature, ['asdf'], bar=[True])
            f.attr = []
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                f_copy = pickle.loads(pickle.dumps(f, proto))
                self.assertEqual(signature(f_copy), signature(f))

    def test_copy(self):
        # [FUNCTION DOC] Checks shallow copy gives an equal partial that shares attr, args and
        # keywords objects with the original.
        f = self.partial(signature, ['asdf'], bar=[True])
        f.attr = []
        f_copy = copy.copy(f)
        self.assertEqual(signature(f_copy), signature(f))
        self.assertIs(f_copy.attr, f.attr)
        self.assertIs(f_copy.args, f.args)
        self.assertIs(f_copy.keywords, f.keywords)

    def test_deepcopy(self):
        # [FUNCTION DOC] Checks deep copy gives an equal partial whose attr, args, nested args and
        # keywords (and nested values) are distinct objects.
        f = self.partial(signature, ['asdf'], bar=[True])
        f.attr = []
        f_copy = copy.deepcopy(f)
        self.assertEqual(signature(f_copy), signature(f))
        self.assertIsNot(f_copy.attr, f.attr)
        self.assertIsNot(f_copy.args, f.args)
        self.assertIsNot(f_copy.args[0], f.args[0])
        self.assertIsNot(f_copy.keywords, f.keywords)
        self.assertIsNot(f_copy.keywords['bar'], f.keywords['bar'])

    def test_setstate(self):
        # [FUNCTION DOC] Checks __setstate__ replaces func, args, keywords and dict, and that None
        # for keywords or dict is accepted and behaves as empty.
        f = self.partial(signature)
        f.__setstate__((capture, (1,), dict(a=10), dict(attr=[])))

        self.assertEqual(signature(f),
                         (capture, (1,), dict(a=10), dict(attr=[])))
        self.assertEqual(f(2, b=20), ((1, 2), {'a': 10, 'b': 20}))

        f.__setstate__((capture, (1,), dict(a=10), None))

        self.assertEqual(signature(f), (capture, (1,), dict(a=10), {}))
        self.assertEqual(f(2, b=20), ((1, 2), {'a': 10, 'b': 20}))

        f.__setstate__((capture, (1,), None, None))
        #self.assertEqual(signature(f), (capture, (1,), {}, {}))
        self.assertEqual(f(2, b=20), ((1, 2), {'b': 20}))
        self.assertEqual(f(2), ((1, 2), {}))
        self.assertEqual(f(), ((1,), {}))

        f.__setstate__((capture, (), {}, None))
        self.assertEqual(signature(f), (capture, (), {}, {}))
        self.assertEqual(f(2, b=20), ((2,), {'b': 20}))
        self.assertEqual(f(2), ((2,), {}))
        self.assertEqual(f(), ((), {}))

    def test_setstate_errors(self):
        # [FUNCTION DOC] Checks __setstate__ raises TypeError for a wrong tuple length, a
        # non-tuple state, a non-callable func, and non-tuple args or non-dict keywords.
        f = self.partial(signature)
        self.assertRaises(TypeError, f.__setstate__, (capture, (), {}))
        self.assertRaises(TypeError, f.__setstate__, (capture, (), {}, {}, None))
        self.assertRaises(TypeError, f.__setstate__, [capture, (), {}, None])
        self.assertRaises(TypeError, f.__setstate__, (None, (), {}, None))
        self.assertRaises(TypeError, f.__setstate__, (capture, None, {}, None))
        self.assertRaises(TypeError, f.__setstate__, (capture, [], {}, None))
        self.assertRaises(TypeError, f.__setstate__, (capture, (), [], None))

    def test_setstate_subclasses(self):
        # [FUNCTION DOC] Checks __setstate__ with tuple/dict subclasses stores plain tuple and
        # dict types, including a tuple subclass with a custom addition, and the calls still work.
        f = self.partial(signature)
        f.__setstate__((capture, MyTuple((1,)), MyDict(a=10), None))
        s = signature(f)
        self.assertEqual(s, (capture, (1,), dict(a=10), {}))
        self.assertIs(type(s[1]), tuple)
        self.assertIs(type(s[2]), dict)
        r = f()
        self.assertEqual(r, ((1,), {'a': 10}))
        self.assertIs(type(r[0]), tuple)
        self.assertIs(type(r[1]), dict)

        f.__setstate__((capture, BadTuple((1,)), {}, None))
        s = signature(f)
        self.assertEqual(s, (capture, (1,), {}, {}))
        self.assertIs(type(s[1]), tuple)
        r = f(2)
        self.assertEqual(r, ((1, 2), {}))
        self.assertIs(type(r[0]), tuple)

    def test_recursive_pickle(self):
        # [FUNCTION DOC] Checks pickling of self-referential partials: a partial that is its
        # own func overflows the (lowered) recursion limit with RecursionError, while one
        # holding itself in args or keywords round-trips and preserves the cycle. Always resets
        # state afterwards.
        with replaced_module('functools', self.module):
            f = self.partial(capture)
            f.__setstate__((f, (), {}, {}))
            try:
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    # gh-117008: Small limit since pickle uses C stack memory
                    with support.infinite_recursion(100):
                        with self.assertRaises(RecursionError):
                            pickle.dumps(f, proto)
            finally:
                f.__setstate__((capture, (), {}, {}))

            f = self.partial(capture)
            f.__setstate__((capture, (f,), {}, {}))
            try:
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    try:
                        self.assertIs(f_copy.args[0], f_copy)
                    finally:
                        f_copy.__setstate__((capture, (), {}, {}))
            finally:
                f.__setstate__((capture, (), {}, {}))

            f = self.partial(capture)
            f.__setstate__((capture, (), {'a': f}, {}))
            try:
                for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    try:
                        self.assertIs(f_copy.keywords['a'], f_copy)
                    finally:
                        f_copy.__setstate__((capture, (), {}, {}))
            finally:
                f.__setstate__((capture, (), {}, {}))

    # Issue 6083: Reference counting bug
    def test_setstate_refcount(self):
        # [FUNCTION DOC] Regression test (issue 6083): __setstate__ given a sequence-like object
        # that yields a huge args tuple must raise TypeError without a reference-count bug.
        class BadSequence:
            # [CLASS DOC] HELPER: Fake 4-item sequence yielding (max, a huge tuple, {}, {}) and
            # then raising IndexError; used as an invalid state.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Reports a length of 4.
                return 4
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Returns max, a million-element tuple, or an empty dict
                # for indexes 0, 1 and 2-3; raises IndexError otherwise.
                if key == 0:
                    return max
                elif key == 1:
                    return tuple(range(1000000))
                elif key in (2, 3):
                    return {}
                raise IndexError

        f = self.partial(object)
        self.assertRaises(TypeError, f.__setstate__, BadSequence())

    def test_partial_as_method(self):
        # [FUNCTION DOC] Checks partial as a class attribute: it does not bind self like a
        # function (called with the given args only; FutureWarning on instance access), while
        # wrapping it in classmethod or staticmethod binds the class or nothing respectively.
        class A:
            # [CLASS DOC] HELPER: Holds a partial as a plain, class, and static method attribute.
            meth = self.partial(capture, 1, a=2)
            cmeth = classmethod(self.partial(capture, 1, a=2))
            smeth = staticmethod(self.partial(capture, 1, a=2))

        a = A()
        self.assertEqual(A.meth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))
        self.assertEqual(A.cmeth(3, b=4), ((1, A, 3), {'a': 2, 'b': 4}))
        self.assertEqual(A.smeth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))
        with self.assertWarns(FutureWarning) as w:
            self.assertEqual(a.meth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))
        self.assertEqual(w.filename, __file__)
        self.assertEqual(a.cmeth(3, b=4), ((1, A, 3), {'a': 2, 'b': 4}))
        self.assertEqual(a.smeth(3, b=4), ((1, 3), {'a': 2, 'b': 4}))

    def test_partial_genericalias(self):
        # [FUNCTION DOC] Checks indexing the partial class makes a generic alias with the right
        # origin, args and (empty) parameters.
        alias = self.partial[int]
        self.assertIs(alias.__origin__, self.partial)
        self.assertEqual(alias.__args__, (int,))
        self.assertEqual(alias.__parameters__, ())


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestPartialC(_TestPartial, CPythonTestCase):
    # [CLASS DOC] Runs the shared partial tests against the C implementation (the plain
    # functools.partial) plus tests for C-only behavior. Skipped when _functools is missing.
    if c_functools:
        module = functools
        partial = functools.partial

    def test_attributes_unwritable(self):
        # [FUNCTION DOC] Checks func, args and keywords cannot be assigned (AttributeError) and
        # the instance __dict__ cannot be deleted (TypeError).
        # attributes should not be writable
        p = self.partial(capture, 1, 2, a=10, b=20)
        self.assertRaises(AttributeError, setattr, p, 'func', map)
        self.assertRaises(AttributeError, setattr, p, 'args', (1, 2))
        self.assertRaises(AttributeError, setattr, p, 'keywords', dict(a=1, b=2))

        p = self.partial(hex)
        try:
            del p.__dict__
        except TypeError:
            pass
        else:
            self.fail('partial object allowed __dict__ to be deleted')

    def test_manually_adding_non_string_keyword(self):
        # [FUNCTION DOC] Checks that a non-string key inserted manually into keywords appears in
        # the repr but makes calling the partial raise TypeError.
        p = self.partial(capture)
        # Adding a non-string/unicode keyword to partial kwargs
        p.keywords[1234] = 'value'
        r = repr(p)
        self.assertIn('1234', r)
        self.assertIn("'value'", r)
        with self.assertRaises(TypeError):
            p()

    def test_keystr_replaces_value(self):
        # [FUNCTION DOC] Checks repr stays safe when a key's __str__ replaces the value in the
        # keywords dict during formatting: the original value is still shown.
        p = self.partial(capture)

        class MutatesYourDict(object):
            # [CLASS DOC] HELPER: Key type whose string conversion mutates the partial's keywords.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Replaces this key's value in the partial's keywords and
                # returns the fixed string astr.
                p.keywords[self] = ['sth2']
                return 'astr'

        # Replacing the value during key formatting should keep the original
        # value alive (at least long enough).
        p.keywords[MutatesYourDict()] = ['sth']
        r = repr(p)
        self.assertIn('astr', r)
        self.assertIn("['sth']", r)


class TestPartialPy(_TestPartial, CPythonTestCase):
    # [CLASS DOC] Runs the shared partial tests against the pure-Python partial implementation.
    module = py_functools
    partial = py_functools.partial


if c_functools:
    class CPartialSubclass(c_functools.partial):
        # [CLASS DOC] HELPER: Empty subclass of the C partial for subclass tests.
        pass

class PyPartialSubclass(py_functools.partial):
    # [CLASS DOC] HELPER: Empty subclass of the Python partial for subclass tests.
    pass

@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestPartialCSubclass(TestPartialC):
    # [CLASS DOC] Re-runs the C partial tests on a subclass; the nested-optimization test is
    # disabled by setting it to None because subclasses are not flattened.
    if c_functools:
        partial = CPartialSubclass

    # partial subclasses are not optimized for nested calls
    test_nested_optimization = None

class TestPartialPySubclass(TestPartialPy):
    # [CLASS DOC] Re-runs the Python partial tests on a subclass of the Python partial.
    partial = PyPartialSubclass

class TestPartialMethod(CPythonTestCase):
    # [CLASS DOC] Tests for functools.partialmethod. The nested class A defines methods built from
    # partialmethod in many variants and the shared instance a is used by tests.

    class A(object):
        # [CLASS DOC] HELPER: Fixture class whose attributes are partialmethod objects: plain,
        # positional, keyword, both, keyword names that collide with self/func, nested, over a
        # partial, over a staticmethod and over a classmethod.
        nothing = functools.partialmethod(capture)
        positional = functools.partialmethod(capture, 1)
        keywords = functools.partialmethod(capture, a=2)
        both = functools.partialmethod(capture, 3, b=4)
        spec_keywords = functools.partialmethod(capture, self=1, func=2)

        nested = functools.partialmethod(positional, 5)

        over_partial = functools.partialmethod(functools.partial(capture, c=6), 7)

        static = functools.partialmethod(staticmethod(capture), 8)
        cls = functools.partialmethod(classmethod(capture), d=9)

    a = A()

    def test_arg_combinations(self):
        # [FUNCTION DOC] Checks the frozen and call-time positional/keyword args combine correctly
        # for several partialmethod variants, via instance and class access.
        self.assertEqual(self.a.nothing(), ((self.a,), {}))
        self.assertEqual(self.a.nothing(5), ((self.a, 5), {}))
        self.assertEqual(self.a.nothing(c=6), ((self.a,), {'c': 6}))
        self.assertEqual(self.a.nothing(5, c=6), ((self.a, 5), {'c': 6}))

        self.assertEqual(self.a.positional(), ((self.a, 1), {}))
        self.assertEqual(self.a.positional(5), ((self.a, 1, 5), {}))
        self.assertEqual(self.a.positional(c=6), ((self.a, 1), {'c': 6}))
        self.assertEqual(self.a.positional(5, c=6), ((self.a, 1, 5), {'c': 6}))

        self.assertEqual(self.a.keywords(), ((self.a,), {'a': 2}))
        self.assertEqual(self.a.keywords(5), ((self.a, 5), {'a': 2}))
        self.assertEqual(self.a.keywords(c=6), ((self.a,), {'a': 2, 'c': 6}))
        self.assertEqual(self.a.keywords(5, c=6), ((self.a, 5), {'a': 2, 'c': 6}))

        self.assertEqual(self.a.both(), ((self.a, 3), {'b': 4}))
        self.assertEqual(self.a.both(5), ((self.a, 3, 5), {'b': 4}))
        self.assertEqual(self.a.both(c=6), ((self.a, 3), {'b': 4, 'c': 6}))
        self.assertEqual(self.a.both(5, c=6), ((self.a, 3, 5), {'b': 4, 'c': 6}))

        self.assertEqual(self.A.both(self.a, 5, c=6), ((self.a, 3, 5), {'b': 4, 'c': 6}))

        self.assertEqual(self.a.spec_keywords(), ((self.a,), {'self': 1, 'func': 2}))

    def test_nested(self):
        # [FUNCTION DOC] Checks a partialmethod built on another partialmethod accumulates args.
        self.assertEqual(self.a.nested(), ((self.a, 1, 5), {}))
        self.assertEqual(self.a.nested(6), ((self.a, 1, 5, 6), {}))
        self.assertEqual(self.a.nested(d=7), ((self.a, 1, 5), {'d': 7}))
        self.assertEqual(self.a.nested(6, d=7), ((self.a, 1, 5, 6), {'d': 7}))

        self.assertEqual(self.A.nested(self.a, 6, d=7), ((self.a, 1, 5, 6), {'d': 7}))

    def test_over_partial(self):
        # [FUNCTION DOC] Checks a partialmethod over a functools.partial merges both sets of args.
        self.assertEqual(self.a.over_partial(), ((self.a, 7), {'c': 6}))
        self.assertEqual(self.a.over_partial(5), ((self.a, 7, 5), {'c': 6}))
        self.assertEqual(self.a.over_partial(d=8), ((self.a, 7), {'c': 6, 'd': 8}))
        self.assertEqual(self.a.over_partial(5, d=8), ((self.a, 7, 5), {'c': 6, 'd': 8}))

        self.assertEqual(self.A.over_partial(self.a, 5, d=8), ((self.a, 7, 5), {'c': 6, 'd': 8}))

    def test_bound_method_introspection(self):
        # [FUNCTION DOC] Checks bound-method introspection: __self__ is the instance for normal
        # variants and the class for the classmethod variant.
        obj = self.a
        self.assertIs(obj.both.__self__, obj)
        self.assertIs(obj.nested.__self__, obj)
        self.assertIs(obj.over_partial.__self__, obj)
        self.assertIs(obj.cls.__self__, self.A)
        self.assertIs(self.A.cls.__self__, self.A)

    def test_unbound_method_retrieval(self):
        # [FUNCTION DOC] Checks that retrieving the variants on the class gives plain functions
        # with no __self__ (the static one on the instance as well).
        obj = self.A
        self.assertFalse(hasattr(obj.both, "__self__"))
        self.assertFalse(hasattr(obj.nested, "__self__"))
        self.assertFalse(hasattr(obj.over_partial, "__self__"))
        self.assertFalse(hasattr(obj.static, "__self__"))
        self.assertFalse(hasattr(self.a.static, "__self__"))

    def test_descriptors(self):
        # [FUNCTION DOC] Checks static and class partialmethods called from both the class and an
        # instance, in subtests, combining frozen and call-time args.
        for obj in [self.A, self.a]:
            with self.subTest(obj=obj):
                self.assertEqual(obj.static(), ((8,), {}))
                self.assertEqual(obj.static(5), ((8, 5), {}))
                self.assertEqual(obj.static(d=8), ((8,), {'d': 8}))
                self.assertEqual(obj.static(5, d=8), ((8, 5), {'d': 8}))

                self.assertEqual(obj.cls(), ((self.A,), {'d': 9}))
                self.assertEqual(obj.cls(5), ((self.A, 5), {'d': 9}))
                self.assertEqual(obj.cls(c=8), ((self.A,), {'c': 8, 'd': 9}))
                self.assertEqual(obj.cls(5, c=8), ((self.A, 5), {'c': 8, 'd': 9}))

    def test_overriding_keywords(self):
        # [FUNCTION DOC] Checks call-time keywords override frozen ones for partialmethod.
        self.assertEqual(self.a.keywords(a=3), ((self.a,), {'a': 3}))
        self.assertEqual(self.A.keywords(self.a, a=3), ((self.a,), {'a': 3}))

    def test_invalid_args(self):
        # [FUNCTION DOC] Checks partialmethod construction raises TypeError for a non-callable
        # func, for no args, and for func passed by keyword.
        with self.assertRaises(TypeError):
            class B(object):
                # [CLASS DOC] HELPER: Throwaway class whose body creates an invalid partialmethod.
                method = functools.partialmethod(None, 1)
        with self.assertRaises(TypeError):
            class B:
                # [CLASS DOC] HELPER: Throwaway class whose body creates an invalid partialmethod.
                method = functools.partialmethod()
        with self.assertRaises(TypeError):
            class B:
                # [CLASS DOC] HELPER: Throwaway class whose body creates an invalid partialmethod.
                method = functools.partialmethod(func=capture, a=1)

    def test_repr(self):
        # [FUNCTION DOC] Checks repr of partialmethod objects from the class dict: the function
        # and frozen args/keywords formatted after functools.partialmethod.
        self.assertEqual(repr(vars(self.A)['nothing']),
                         'functools.partialmethod({})'.format(capture))
        self.assertEqual(repr(vars(self.A)['positional']),
                         'functools.partialmethod({}, 1)'.format(capture))
        self.assertEqual(repr(vars(self.A)['keywords']),
                         'functools.partialmethod({}, a=2)'.format(capture))
        self.assertEqual(repr(vars(self.A)['spec_keywords']),
                         'functools.partialmethod({}, self=1, func=2)'.format(capture))
        self.assertEqual(repr(vars(self.A)['both']),
                         'functools.partialmethod({}, 3, b=4)'.format(capture))

    def test_abstract(self):
        # [FUNCTION DOC] Checks __isabstractmethod__ propagates from an abstract method to a
        # partialmethod built on it, and is absent for the concrete variants of class A.
        class Abstract(abc.ABCMeta):
            # [CLASS DOC] HELPER: Metaclass subclass declaring an abstract method and a
            # partialmethod of it.

            @abc.abstractmethod
            def add(self, x, y):
                # [FUNCTION DOC] HELPER: Abstract placeholder method that does nothing.
                pass

            add5 = functools.partialmethod(add, 5)

        self.assertTrue(Abstract.add.__isabstractmethod__)
        self.assertTrue(Abstract.add5.__isabstractmethod__)

        for func in [self.A.static, self.A.cls, self.A.over_partial, self.A.nested, self.A.both]:
            self.assertFalse(getattr(func, '__isabstractmethod__', False))

    def test_positional_only(self):
        # [FUNCTION DOC] Checks a partial can freeze the first of two positional-only parameters.
        def f(a, b, /):
            # [FUNCTION DOC] HELPER: Returns the sum of its two positional-only arguments.
            return a + b

        p = functools.partial(f, 1)
        self.assertEqual(p(2), f(1, 2))


class TestUpdateWrapper(CPythonTestCase):
    # [CLASS DOC] Tests for functools.update_wrapper; also the base class for the wraps tests,
    # which override the helper that builds the wrapper/wrapped pair.

    def check_wrapper(self, wrapper, wrapped,
                      assigned=functools.WRAPPER_ASSIGNMENTS,
                      updated=functools.WRAPPER_UPDATES):
        # [FUNCTION DOC] HELPER: Asserts that wrapper got every attribute named in assigned as the
        # identical object, that each mapping named in updated contains the wrapped entries
        # (except __wrapped__ in __dict__), and that wrapper.__wrapped__ is the wrapped function.
        # Check attributes were assigned
        for name in assigned:
            self.assertIs(getattr(wrapper, name), getattr(wrapped, name))
        # Check attributes were updated
        for name in updated:
            wrapper_attr = getattr(wrapper, name)
            wrapped_attr = getattr(wrapped, name)
            for key in wrapped_attr:
                if name == "__dict__" and key == "__wrapped__":
                    # __wrapped__ is overwritten by the update code
                    continue
                self.assertIs(wrapped_attr[key], wrapper_attr[key])
        # Check __wrapped__
        self.assertIs(wrapper.__wrapped__, wrapped)


    def _default_update(self):
        # [FUNCTION DOC] HELPER: Builds a generic function f (docstring, annotation, extra attr
        # and a bogus __wrapped__) and a wrapper with other annotations, applies update_wrapper
        # with defaults, and returns the (wrapper, f) pair.
        def f[T](a:'This is a new annotation'):
            # [FUNCTION DOC] HELPER: Wrapped function with a docstring, a type parameter and one
            # annotated argument.
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        f.__wrapped__ = "This is a bald faced lie"
        def wrapper(b:'This is the prior annotation'):
            # [FUNCTION DOC] HELPER: Wrapper with a different annotated argument; does nothing.
            pass
        functools.update_wrapper(wrapper, f)
        return wrapper, f

    def test_default_update(self):
        # [FUNCTION DOC] Checks default update_wrapper copies name, qualname, attr, annotations
        # and type params and sets __wrapped__ to the function (replacing the bogus one).
        wrapper, f = self._default_update()
        self.check_wrapper(wrapper, f)
        T, = f.__type_params__
        self.assertIs(wrapper.__wrapped__, f)
        self.assertEqual(wrapper.__name__, 'f')
        self.assertEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.attr, 'This is also a test')
        self.assertEqual(wrapper.__annotations__['a'], 'This is a new annotation')
        self.assertNotIn('b', wrapper.__annotations__)
        self.assertEqual(wrapper.__type_params__, (T,))

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_default_update_doc(self):
        # [FUNCTION DOC] Checks the docstring is copied by default (skipped under -OO).
        wrapper, f = self._default_update()
        self.assertEqual(wrapper.__doc__, 'This is a test')

    def test_no_update(self):
        # [FUNCTION DOC] Checks passing empty assigned/updated tuples leaves wrapper name,
        # qualname, doc, annotations and attributes untouched except for __wrapped__.
        def f():
            # [FUNCTION DOC] HELPER: Documented function with a no-op body.
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper.
            pass
        functools.update_wrapper(wrapper, f, (), ())
        self.check_wrapper(wrapper, f, (), ())
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertEqual(wrapper.__annotations__, {})
        self.assertFalse(hasattr(wrapper, 'attr'))

    def test_selective_update(self):
        # [FUNCTION DOC] Checks selective assigned/updated name tuples copy only those attributes
        # and merge the dict attribute into the wrapper's.
        def f():
            # [FUNCTION DOC] HELPER: No-op function.
            pass
        f.attr = 'This is a different test'
        f.dict_attr = dict(a=1, b=2, c=3)
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper.
            pass
        wrapper.dict_attr = {}
        assign = ('attr',)
        update = ('dict_attr',)
        functools.update_wrapper(wrapper, f, assign, update)
        self.check_wrapper(wrapper, f, assign, update)
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertEqual(wrapper.attr, 'This is a different test')
        self.assertEqual(wrapper.dict_attr, f.dict_attr)

    def test_missing_attributes(self):
        # [FUNCTION DOC] Checks attributes missing on the wrapped function are skipped, but a
        # wrapper lacking (or having a non-mapping) update target raises AttributeError.
        def f():
            # [FUNCTION DOC] HELPER: No-op function.
            pass
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper.
            pass
        wrapper.dict_attr = {}
        assign = ('attr',)
        update = ('dict_attr',)
        # Missing attributes on wrapped object are ignored
        functools.update_wrapper(wrapper, f, assign, update)
        self.assertNotIn('attr', wrapper.__dict__)
        self.assertEqual(wrapper.dict_attr, {})
        # Wrapper must have expected attributes for updating
        del wrapper.dict_attr
        with self.assertRaises(AttributeError):
            functools.update_wrapper(wrapper, f, assign, update)
        wrapper.dict_attr = 1
        with self.assertRaises(AttributeError):
            functools.update_wrapper(wrapper, f, assign, update)

    @support.requires_docstrings
    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_builtin_update(self):
        # [FUNCTION DOC] Checks update_wrapper from the builtin max copies its name and docstring
        # (bug 1576241) and gives empty annotations. Skipped without docstrings.
        # Test for bug #1576241
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper.
            pass
        functools.update_wrapper(wrapper, max)
        self.assertEqual(wrapper.__name__, 'max')
        self.assertTrue(wrapper.__doc__.startswith('max('))
        self.assertEqual(wrapper.__annotations__, {})

    def test_update_type_wrapper(self):
        # [FUNCTION DOC] Checks wrapping the builtin type gives name type, empty annotations and
        # empty type params. The wrapper body is a one-line pass, so it has no block of its own:
        # it accepts any positional arguments and does nothing.
        def wrapper(*args): pass

        functools.update_wrapper(wrapper, type)
        self.assertEqual(wrapper.__name__, 'type')
        self.assertEqual(wrapper.__annotations__, {})
        self.assertEqual(wrapper.__type_params__, ())


class TestWraps(TestUpdateWrapper):
    # [CLASS DOC] Re-runs the update_wrapper tests using the functools.wraps decorator form.

    def _default_update(self):
        # [FUNCTION DOC] HELPER: Same pair builder as the base class but using the wraps
        # decorator; returns the (wrapper, f) pair.
        def f():
            # [FUNCTION DOC] HELPER: Documented function with a no-op body.
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        f.__wrapped__ = "This is still a bald faced lie"
        @functools.wraps(f)
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper decorated by wraps.
            pass
        return wrapper, f

    def test_default_update(self):
        # [FUNCTION DOC] Checks default wraps copies name, qualname and attr.
        wrapper, f = self._default_update()
        self.check_wrapper(wrapper, f)
        self.assertEqual(wrapper.__name__, 'f')
        self.assertEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.attr, 'This is also a test')

    @unittest.skipIf(sys.flags.optimize >= 2,
                     "Docstrings are omitted with -O2 and above")
    def test_default_update_doc(self):
        # [FUNCTION DOC] Checks the default wraps copies the docstring (skipped under -OO).
        wrapper, _ = self._default_update()
        self.assertEqual(wrapper.__doc__, 'This is a test')

    def test_no_update(self):
        # [FUNCTION DOC] Checks wraps with empty assigned/updated tuples changes nothing
        # but __wrapped__.
        def f():
            # [FUNCTION DOC] HELPER: Documented function with a no-op body.
            """This is a test"""
            pass
        f.attr = 'This is also a test'
        @functools.wraps(f, (), ())
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper.
            pass
        self.check_wrapper(wrapper, f, (), ())
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertFalse(hasattr(wrapper, 'attr'))

    def test_selective_update(self):
        # [FUNCTION DOC] Checks wraps with selective assigned and updated names, with the wrapper
        # first given an empty dict attribute by a decorator.
        def f():
            # [FUNCTION DOC] HELPER: No-op function.
            pass
        f.attr = 'This is a different test'
        f.dict_attr = dict(a=1, b=2, c=3)
        def add_dict_attr(f):
            # [FUNCTION DOC] HELPER: Decorator that gives the function an empty dict_attr and
            # returns it.
            f.dict_attr = {}
            return f
        assign = ('attr',)
        update = ('dict_attr',)
        @functools.wraps(f, assign, update)
        @add_dict_attr
        def wrapper():
            # [FUNCTION DOC] HELPER: No-op wrapper.
            pass
        self.check_wrapper(wrapper, f, assign, update)
        self.assertEqual(wrapper.__name__, 'wrapper')
        self.assertNotEqual(wrapper.__qualname__, f.__qualname__)
        self.assertEqual(wrapper.__doc__, None)
        self.assertEqual(wrapper.attr, 'This is a different test')
        self.assertEqual(wrapper.dict_attr, f.dict_attr)


class _TestReduce:
    # [CLASS DOC] Mixin with reduce tests; subclasses set the class attribute reduce to the
    # implementation under test.
    def test_reduce(self):
        # [FUNCTION DOC] Checks reduce across many inputs: strings, lists, ranges, a custom lazy
        # sequence, empty sequences with and without an initial value, TypeError for bad arguments
        # and for empty input without initial, one-item inputs never calling the function, and
        # propagation of errors from iteration or indexing.
        class Squares:
            # [CLASS DOC] HELPER: Lazy sequence of squares up to a max, filled on demand.
            def __init__(self, max):
                # [FUNCTION DOC] HELPER: Stores the maximum size and starts an empty list of
                # computed squares.
                self.max = max
                self.sofar = []

            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns the number of squares computed so far.
                return len(self.sofar)

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns the square at index i, computing missing ones;
                # raises IndexError if i is outside 0 to max.
                if not 0 <= i < self.max: raise IndexError
                n = len(self.sofar)
                while n <= i:
                    self.sofar.append(n*n)
                    n += 1
                return self.sofar[i]
        def add(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum (or concatenation) of two arguments.
            return x + y
        self.assertEqual(self.reduce(add, ['a', 'b', 'c'], ''), 'abc')
        self.assertEqual(
            self.reduce(add, [['a', 'c'], [], ['d', 'w']], []),
            ['a','c','d','w']
        )
        self.assertEqual(self.reduce(lambda x, y: x*y, range(2,8), 1), 5040)
        self.assertEqual(
            self.reduce(lambda x, y: x*y, range(2,21), 1),
            2432902008176640000
        )
        self.assertEqual(self.reduce(add, Squares(10)), 285)
        self.assertEqual(self.reduce(add, Squares(10), 0), 285)
        self.assertEqual(self.reduce(add, Squares(0), 0), 0)
        self.assertRaises(TypeError, self.reduce)
        self.assertRaises(TypeError, self.reduce, 42, 42)
        self.assertRaises(TypeError, self.reduce, 42, 42, 42)
        self.assertEqual(self.reduce(42, "1"), "1") # func is never called with one item
        self.assertEqual(self.reduce(42, "", "1"), "1") # func is never called with one item
        self.assertRaises(TypeError, self.reduce, 42, (42, 42))
        self.assertRaises(TypeError, self.reduce, add, []) # arg 2 must not be empty sequence with no initial value
        self.assertRaises(TypeError, self.reduce, add, "")
        self.assertRaises(TypeError, self.reduce, add, ())
        self.assertRaises(TypeError, self.reduce, add, object())

        class TestFailingIter:
            # [CLASS DOC] HELPER: Iterable whose iteration raises RuntimeError.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Always raises RuntimeError.
                raise RuntimeError
        self.assertRaises(RuntimeError, self.reduce, add, TestFailingIter())

        self.assertEqual(self.reduce(add, [], None), None)
        self.assertEqual(self.reduce(add, [], 42), 42)

        class BadSeq:
            # [CLASS DOC] HELPER: Sequence whose indexing raises ValueError.
            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, self.reduce, 42, BadSeq())

    # Test reduce()'s use of iterators.
    def test_iterator_usage(self):
        # [FUNCTION DOC] Checks reduce uses iterator semantics: a legacy getitem sequence, with
        # and without initial value, empty input, a single item, and a mapping (reduces its keys).
        class SequenceClass:
            # [CLASS DOC] HELPER: Old-style sequence of the integers 0 to n-1 via indexing only.
            def __init__(self, n):
                # [FUNCTION DOC] HELPER: Stores the length n.
                self.n = n
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Returns i when in range; raises IndexError otherwise.
                if 0 <= i < self.n:
                    return i
                else:
                    raise IndexError

        from operator import add
        self.assertEqual(self.reduce(add, SequenceClass(5)), 10)
        self.assertEqual(self.reduce(add, SequenceClass(5), 42), 52)
        self.assertRaises(TypeError, self.reduce, add, SequenceClass(0))
        self.assertEqual(self.reduce(add, SequenceClass(0), 42), 42)
        self.assertEqual(self.reduce(add, SequenceClass(1)), 0)
        self.assertEqual(self.reduce(add, SequenceClass(1), 42), 42)

        d = {"one": 1, "two": 2, "three": 3}
        self.assertEqual(self.reduce(add, d), "".join(d.keys()))


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestReduceC(_TestReduce, CPythonTestCase):
    # [CLASS DOC] Runs the reduce tests with the C reduce (the regular functools.reduce under
    # Dynamo); skipped when _functools is missing.
    if c_functools:
        if TEST_WITH_TORCHDYNAMO:
            reduce = functools.reduce
        else:
            reduce = c_functools.reduce


class TestReducePy(_TestReduce, CPythonTestCase):
    # [CLASS DOC] Runs the reduce tests with the pure-Python reduce.
    reduce = staticmethod(py_functools.reduce)


class _TestCmpToKey:
    # [CLASS DOC] Mixin with cmp_to_key tests; subclasses set cmp_to_key to the implementation.

    def test_cmp_to_key(self):
        # [FUNCTION DOC] Checks keys made from a three-way comparison function compare equal,
        # greater and less correctly, including mixed int, float and string inputs.
        def cmp1(x, y):
            # [FUNCTION DOC] HELPER: Three-way comparison returning -1, 0 or 1.
            return (x > y) - (x < y)
        key = self.cmp_to_key(cmp1)
        self.assertEqual(key(3), key(3))
        self.assertGreater(key(3), key(1))
        self.assertGreaterEqual(key(3), key(3))

        def cmp2(x, y):
            # [FUNCTION DOC] HELPER: Compares by the integer value of each argument, returning
            # the difference.
            return int(x) - int(y)
        key = self.cmp_to_key(cmp2)
        self.assertEqual(key(4.0), key('4'))
        self.assertLess(key(2), key('35'))
        self.assertLessEqual(key(2), key('35'))
        self.assertNotEqual(key(2), key('35'))

    def test_cmp_to_key_arguments(self):
        # [FUNCTION DOC] Checks cmp_to_key and key objects accept keyword arguments (mycmp,
        # obj) and raise TypeError or AttributeError for wrong argument counts or comparing
        # with a non-key.
        def cmp1(x, y):
            # [FUNCTION DOC] HELPER: Three-way comparison returning -1, 0 or 1.
            return (x > y) - (x < y)
        key = self.cmp_to_key(mycmp=cmp1)
        self.assertEqual(key(obj=3), key(obj=3))
        self.assertGreater(key(obj=3), key(obj=1))
        with self.assertRaises((TypeError, AttributeError)):
            key(3) > 1    # rhs is not a K object
        with self.assertRaises((TypeError, AttributeError)):
            1 < key(3)    # lhs is not a K object
        with self.assertRaises(TypeError):
            key = self.cmp_to_key()             # too few args
        with self.assertRaises(TypeError):
            key = self.cmp_to_key(cmp1, None)   # too many args
        key = self.cmp_to_key(cmp1)
        with self.assertRaises(TypeError):
            key()                                    # too few args
        with self.assertRaises(TypeError):
            key(None, None)                          # too many args

    def test_bad_cmp(self):
        # [FUNCTION DOC] Checks exceptions from the comparison function, or from the object it
        # returns, propagate out of key comparisons.
        def cmp1(x, y):
            # [FUNCTION DOC] HELPER: Always raises ZeroDivisionError.
            raise ZeroDivisionError
        key = self.cmp_to_key(cmp1)
        with self.assertRaises(ZeroDivisionError):
            key(3) > key(1)

        class BadCmp:
            # [CLASS DOC] HELPER: Result type whose less-than raises ZeroDivisionError.
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Always raises ZeroDivisionError.
                raise ZeroDivisionError
        def cmp1(x, y):
            # [FUNCTION DOC] HELPER: Returns a BadCmp instance.
            # KNOWN ISSUE: this cmp1 is defined after key was created from the first cmp1 and
            # is never used, so the second assertion repeats the first check rather than
            # testing BadCmp.
            return BadCmp()
        with self.assertRaises(ZeroDivisionError):
            key(3) > key(1)

    def test_obj_field(self):
        # [FUNCTION DOC] Checks the key object exposes the wrapped value as the obj attribute.
        def cmp1(x, y):
            # [FUNCTION DOC] HELPER: Three-way comparison returning -1, 0 or 1.
            return (x > y) - (x < y)
        key = self.cmp_to_key(mycmp=cmp1)
        self.assertEqual(key(50).obj, 50)

    def test_sort_int(self):
        # [FUNCTION DOC] Checks sorting integers with a reversing comparison function gives
        # descending order.
        def mycmp(x, y):
            # [FUNCTION DOC] HELPER: Reverse comparison returning y minus x.
            return y - x
        self.assertEqual(sorted(range(5), key=self.cmp_to_key(mycmp)),
                         [4, 3, 2, 1, 0])

    def test_sort_int_str(self):
        # [FUNCTION DOC] Checks sorting a mix of ints and numeric strings by integer value.
        def mycmp(x, y):
            # [FUNCTION DOC] HELPER: Three-way comparison of the integer values of both arguments.
            x, y = int(x), int(y)
            return (x > y) - (x < y)
        values = [5, '3', 7, 2, '0', '1', 4, '10', 1]
        values = sorted(values, key=self.cmp_to_key(mycmp))
        self.assertEqual([int(value) for value in values],
                         [0, 1, 1, 2, 3, 4, 5, 7, 10])

    def test_hash(self):
        # [FUNCTION DOC] Checks key objects are unhashable (hash raises TypeError, not Hashable).
        def mycmp(x, y):
            # [FUNCTION DOC] HELPER: Reverse comparison returning y minus x.
            return y - x
        key = self.cmp_to_key(mycmp)
        k = key(10)
        self.assertRaises(TypeError, hash, k)
        self.assertNotIsInstance(k, collections.abc.Hashable)

    @unittest.skipIf(support.MISSING_C_DOCSTRINGS,
                     "Signature information for builtins requires docstrings")
    def test_cmp_to_signature(self):
        # [FUNCTION DOC] Checks the introspected signature of cmp_to_key is "(mycmp)" and that of
        # a key class is "(obj)". Skipped without C docstrings.
        sig = Signature.from_callable(self.cmp_to_key)
        self.assertEqual(str(sig), '(mycmp)')
        def mycmp(x, y):
            # [FUNCTION DOC] HELPER: Reverse comparison returning y minus x.
            return y - x
        sig = Signature.from_callable(self.cmp_to_key(mycmp))
        self.assertEqual(str(sig), '(obj)')



@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestCmpToKeyC(_TestCmpToKey, CPythonTestCase):
    # [CLASS DOC] Runs cmp_to_key tests with the C implementation (functools version under
    # Dynamo); skipped when _functools is missing.
    if c_functools:
        if TEST_WITH_TORCHDYNAMO:
            cmp_to_key = functools.cmp_to_key
        else:
            cmp_to_key = c_functools.cmp_to_key

    @support.cpython_only
    def test_disallow_instantiation(self):
        # [FUNCTION DOC] Checks the C key type refuses direct instantiation (bpo-43916).
        # CPython only.
        # Ensure that the type disallows instantiation (bpo-43916)
        support.check_disallow_instantiation(
            self, type(c_functools.cmp_to_key(None))
        )


class TestCmpToKeyPy(_TestCmpToKey, CPythonTestCase):
    # [CLASS DOC] Runs cmp_to_key tests with the pure-Python implementation.
    cmp_to_key = staticmethod(py_functools.cmp_to_key)


class TestTotalOrdering(CPythonTestCase):
    # [CLASS DOC] Tests for functools.total_ordering with classes defining one comparison.

    def test_total_ordering_lt(self):
        # [FUNCTION DOC] Checks a class with only less-than and equality gets all orderings.
        @functools.total_ordering
        class A:
            # [CLASS DOC] HELPER: Value wrapper defining less-than and equality, decorated
            # by total_ordering.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Compares values with less-than.
                return self.value < other.value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for equality.
                return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(1) > A(2))

    def test_total_ordering_le(self):
        # [FUNCTION DOC] Checks a class with only less-or-equal and equality gets all orderings.
        @functools.total_ordering
        class A:
            # [CLASS DOC] HELPER: Value wrapper defining less-or-equal and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __le__(self, other):
                # [FUNCTION DOC] HELPER: Compares values with less-or-equal.
                return self.value <= other.value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for equality.
                return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(1) >= A(2))

    def test_total_ordering_gt(self):
        # [FUNCTION DOC] Checks a class with only greater-than and equality gets all orderings.
        @functools.total_ordering
        class A:
            # [CLASS DOC] HELPER: Value wrapper defining greater-than and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __gt__(self, other):
                # [FUNCTION DOC] HELPER: Compares values with greater-than.
                return self.value > other.value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for equality.
                return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(2) < A(1))

    def test_total_ordering_ge(self):
        # [FUNCTION DOC] Checks a class with only greater-or-equal and equality gets
        # all orderings.
        @functools.total_ordering
        class A:
            # [CLASS DOC] HELPER: Value wrapper defining greater-or-equal and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __ge__(self, other):
                # [FUNCTION DOC] HELPER: Compares values with greater-or-equal.
                return self.value >= other.value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for equality.
                return self.value == other.value
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))
        self.assertFalse(A(2) <= A(1))

    def test_total_ordering_no_overwrite(self):
        # [FUNCTION DOC] Checks total_ordering keeps the ordering methods an int
        # subclass inherits.
        # new methods should not overwrite existing
        @functools.total_ordering
        class A(int):
            # [CLASS DOC] HELPER: Empty int subclass decorated by total_ordering.
            pass
        self.assertTrue(A(1) < A(2))
        self.assertTrue(A(2) > A(1))
        self.assertTrue(A(1) <= A(2))
        self.assertTrue(A(2) >= A(1))
        self.assertTrue(A(2) <= A(2))
        self.assertTrue(A(2) >= A(2))

    def test_no_operations_defined(self):
        # [FUNCTION DOC] Checks total_ordering raises ValueError for a class with no
        # ordering method.
        with self.assertRaises(ValueError):
            @functools.total_ordering
            class A:
                # [CLASS DOC] HELPER: Class without any ordering methods.
                pass

    def test_notimplemented(self):
        # [FUNCTION DOC] Checks the generated ordering methods return NotImplemented when the
        # single defined comparison does for other operand types.
        # Verify NotImplemented results are correctly handled
        @functools.total_ordering
        class ImplementsLessThan:
            # [CLASS DOC] HELPER: Defines less-than (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsLessThan):
                    return self.value == other.value
                return False
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsLessThan):
                    return self.value < other.value
                return NotImplemented

        @functools.total_ordering
        class ImplementsLessThanEqualTo:
            # [CLASS DOC] HELPER: Defines less-or-equal (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsLessThanEqualTo):
                    return self.value == other.value
                return False
            def __le__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsLessThanEqualTo):
                    return self.value <= other.value
                return NotImplemented

        @functools.total_ordering
        class ImplementsGreaterThan:
            # [CLASS DOC] HELPER: Defines greater-than (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsGreaterThan):
                    return self.value == other.value
                return False
            def __gt__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsGreaterThan):
                    return self.value > other.value
                return NotImplemented

        @functools.total_ordering
        class ImplementsGreaterThanEqualTo:
            # [CLASS DOC] HELPER: Defines greater-or-equal (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsGreaterThanEqualTo):
                    return self.value == other.value
                return False
            def __ge__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsGreaterThanEqualTo):
                    return self.value >= other.value
                return NotImplemented

        self.assertIs(ImplementsLessThan(1).__le__(1), NotImplemented)
        self.assertIs(ImplementsLessThan(1).__gt__(1), NotImplemented)
        self.assertIs(ImplementsLessThan(1).__ge__(1), NotImplemented)
        self.assertIs(ImplementsLessThanEqualTo(1).__lt__(1), NotImplemented)
        self.assertIs(ImplementsLessThanEqualTo(1).__gt__(1), NotImplemented)
        self.assertIs(ImplementsLessThanEqualTo(1).__ge__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThan(1).__lt__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThan(1).__gt__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThan(1).__ge__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThanEqualTo(1).__lt__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThanEqualTo(1).__le__(1), NotImplemented)
        self.assertIs(ImplementsGreaterThanEqualTo(1).__gt__(1), NotImplemented)

    def test_type_error_when_not_implemented(self):
        # [FUNCTION DOC] Checks bug 10042: mixed comparisons between total_ordering classes that
        # return NotImplemented raise TypeError instead of recursing, including the equal-values
        # case where the only defined comparison returns NotImplemented.
        # bug 10042; ensure stack overflow does not occur
        # when decorated types return NotImplemented
        @functools.total_ordering
        class ImplementsLessThan:
            # [CLASS DOC] HELPER: Defines less-than (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsLessThan):
                    return self.value == other.value
                return False
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsLessThan):
                    return self.value < other.value
                return NotImplemented

        @functools.total_ordering
        class ImplementsGreaterThan:
            # [CLASS DOC] HELPER: Defines greater-than (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsGreaterThan):
                    return self.value == other.value
                return False
            def __gt__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsGreaterThan):
                    return self.value > other.value
                return NotImplemented

        @functools.total_ordering
        class ImplementsLessThanEqualTo:
            # [CLASS DOC] HELPER: Defines less-or-equal (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsLessThanEqualTo):
                    return self.value == other.value
                return False
            def __le__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsLessThanEqualTo):
                    return self.value <= other.value
                return NotImplemented

        @functools.total_ordering
        class ImplementsGreaterThanEqualTo:
            # [CLASS DOC] HELPER: Defines greater-or-equal (NotImplemented for foreign types)
            # and equality.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ImplementsGreaterThanEqualTo):
                    return self.value == other.value
                return False
            def __ge__(self, other):
                # [FUNCTION DOC] HELPER: Compares values for same type; otherwise
                # returns NotImplemented.
                if isinstance(other, ImplementsGreaterThanEqualTo):
                    return self.value >= other.value
                return NotImplemented

        @functools.total_ordering
        class ComparatorNotImplemented:
            # [CLASS DOC] HELPER: Defines equality and a less-than that always
            # returns NotImplemented.
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if same type with the same value, else False.
                if isinstance(other, ComparatorNotImplemented):
                    return self.value == other.value
                return False
            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Always returns NotImplemented.
                return NotImplemented

        with self.subTest("LT < 1"), self.assertRaises(TypeError):
            ImplementsLessThan(-1) < 1

        with self.subTest("LT < LE"), self.assertRaises(TypeError):
            ImplementsLessThan(0) < ImplementsLessThanEqualTo(0)

        with self.subTest("LT < GT"), self.assertRaises(TypeError):
            ImplementsLessThan(1) < ImplementsGreaterThan(1)

        with self.subTest("LE <= LT"), self.assertRaises(TypeError):
            ImplementsLessThanEqualTo(2) <= ImplementsLessThan(2)

        with self.subTest("LE <= GE"), self.assertRaises(TypeError):
            ImplementsLessThanEqualTo(3) <= ImplementsGreaterThanEqualTo(3)

        with self.subTest("GT > GE"), self.assertRaises(TypeError):
            ImplementsGreaterThan(4) > ImplementsGreaterThanEqualTo(4)

        with self.subTest("GT > LT"), self.assertRaises(TypeError):
            ImplementsGreaterThan(5) > ImplementsLessThan(5)

        with self.subTest("GE >= GT"), self.assertRaises(TypeError):
            ImplementsGreaterThanEqualTo(6) >= ImplementsGreaterThan(6)

        with self.subTest("GE >= LE"), self.assertRaises(TypeError):
            ImplementsGreaterThanEqualTo(7) >= ImplementsLessThanEqualTo(7)

        with self.subTest("GE when equal"):
            a = ComparatorNotImplemented(8)
            b = ComparatorNotImplemented(8)
            self.assertEqual(a, b)
            with self.assertRaises(TypeError):
                a >= b

        with self.subTest("LE when equal"):
            a = ComparatorNotImplemented(9)
            b = ComparatorNotImplemented(9)
            self.assertEqual(a, b)
            with self.assertRaises(TypeError):
                a <= b

    def test_pickle(self):
        # [FUNCTION DOC] Checks the ordering methods of a module-level total_ordering class pickle
        # by reference for every protocol and unpickle to the identical function.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            for name in '__lt__', '__gt__', '__le__', '__ge__':
                with self.subTest(method=name, proto=proto):
                    method = getattr(Orderable_LT, name)
                    method_copy = pickle.loads(pickle.dumps(method, proto))
                    self.assertIs(method_copy, method)


    def test_total_ordering_for_metaclasses_issue_44605(self):
        # [FUNCTION DOC] Checks total_ordering works for a metaclass (issue 44605): classes
        # compare by name through the generated methods.
        @functools.total_ordering
        class SortableMeta(type):
            # [CLASS DOC] HELPER: Metaclass ordering classes by name, decorated by total_ordering.
            def __new__(cls, name, bases, ns):
                # [FUNCTION DOC] HELPER: Creates the class through the parent type constructor.
                return super().__new__(cls, name, bases, ns)

            def __lt__(self, other):
                # [FUNCTION DOC] HELPER: Compares class names with less-than.
                if not isinstance(other, SortableMeta):
                    pass
                return self.__name__ < other.__name__

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Compares class names for equality.
                if not isinstance(other, SortableMeta):
                    pass
                return self.__name__ == other.__name__

        class B(metaclass=SortableMeta):
            # [CLASS DOC] HELPER: Empty class using the sortable metaclass.
            pass

        class A(metaclass=SortableMeta):
            # [CLASS DOC] HELPER: Empty class using the sortable metaclass.
            pass

        self.assertTrue(A < B)
        self.assertFalse(A > B)


@functools.total_ordering
class Orderable_LT:
    # [CLASS DOC] HELPER: Module-level value wrapper with less-than and equality, decorated by
    # total_ordering; used by the pickling test.
    def __init__(self, value):
        # [FUNCTION DOC] HELPER: Stores the value.
        self.value = value
    def __lt__(self, other):
        # [FUNCTION DOC] HELPER: Compares values with less-than.
        return self.value < other.value
    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Compares values for equality.
        return self.value == other.value


class _TestCache:
    # [CLASS DOC] Mixin checking functools.cache delegates to lru_cache; subclasses set the class
    # attribute module to the implementation.
    # This tests that the pass-through is working as designed.
    # The underlying functionality is tested in TestLRU.

    def test_cache(self):
        # [FUNCTION DOC] Checks a cached recursive Fibonacci gives correct values, the expected
        # hit/miss/size cache_info with no maxsize, and cache_clear resetting the counters.
        @self.module.cache
        def fib(n):
            # [FUNCTION DOC] HELPER: Cached recursive Fibonacci.
            if n < 2:
                return n
            return fib(n-1) + fib(n-2)
        self.assertEqual([fib(n) for n in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))


class TestCachePy(_TestCache, CPythonTestCase):
    # [CLASS DOC] Runs the cache tests against the pure-Python module.
    module = py_functools


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestCacheC(_TestCache, CPythonTestCase):
    # [CLASS DOC] Runs the cache tests against the C module; skipped when _functools is missing.
    if c_functools:
        module = c_functools


class _TestLRUU:
    # [CLASS DOC] Mixin with lru_cache tests shared by the C and Python implementations. Concrete
    # subclasses set module (the functools implementation) and cached_func (a one-item tuple
    # holding a module-level cached function), and define cached_meth and cached_staticmeth.

    def test_lru(self):
        # [FUNCTION DOC] Core lru_cache test: a bounded cache over random calls reports hit, miss,
        # size and maxsize counters correctly, cache_clear resets them, __wrapped__ bypasses the
        # cache, and maxsize of 0 (never cache), 1 and 2 follow least-recently-used eviction with
        # the expected number of underlying calls.
        def orig(x, y):
            # [FUNCTION DOC] HELPER: Reference function returning three times x plus y.
            return 3 * x + y
        f = self.module.lru_cache(maxsize=20)(orig)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(maxsize, 20)
        self.assertEqual(currsize, 0)
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 0)

        domain = range(5)
        for i in range(1000):
            x, y = choice(domain), choice(domain)
            actual = f(x, y)
            expected = orig(x, y)
            self.assertEqual(actual, expected)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertTrue(hits > misses)
        self.assertEqual(hits + misses, 1000)
        self.assertEqual(currsize, 20)

        f.cache_clear()   # test clearing
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 0)
        self.assertEqual(currsize, 0)
        f(x, y)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # Test bypassing the cache
        self.assertIs(f.__wrapped__, orig)
        f.__wrapped__(x, y)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # test size zero (which means "never-cache")
        @self.module.lru_cache(0)
        def f():
            # [FUNCTION DOC] HELPER: Counts calls in the enclosing f_cnt variable and returns 20.
            nonlocal f_cnt
            f_cnt += 1
            return 20
        self.assertEqual(f.cache_info().maxsize, 0)
        f_cnt = 0
        for i in range(5):
            self.assertEqual(f(), 20)
        self.assertEqual(f_cnt, 5)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 0)
        self.assertEqual(misses, 5)
        self.assertEqual(currsize, 0)

        # test size one
        @self.module.lru_cache(1)
        def f():
            # [FUNCTION DOC] HELPER: Counts calls in the enclosing f_cnt variable and returns 20.
            nonlocal f_cnt
            f_cnt += 1
            return 20
        self.assertEqual(f.cache_info().maxsize, 1)
        f_cnt = 0
        for i in range(5):
            self.assertEqual(f(), 20)
        self.assertEqual(f_cnt, 1)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 4)
        self.assertEqual(misses, 1)
        self.assertEqual(currsize, 1)

        # test size two
        @self.module.lru_cache(2)
        def f(x):
            # [FUNCTION DOC] HELPER: Counts calls in the enclosing f_cnt variable and returns x
            # times 10.
            nonlocal f_cnt
            f_cnt += 1
            return x*10
        self.assertEqual(f.cache_info().maxsize, 2)
        f_cnt = 0
        for x in 7, 9, 7, 9, 7, 9, 8, 8, 8, 9, 9, 9, 8, 8, 8, 7:
            #    *  *              *                          *
            self.assertEqual(f(x), x*10)
        self.assertEqual(f_cnt, 4)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(hits, 12)
        self.assertEqual(misses, 4)
        self.assertEqual(currsize, 2)

    def test_lru_no_args(self):
        # [FUNCTION DOC] Checks lru_cache used as a bare decorator defaults to maxsize 128 and
        # counts hits and misses.
        @self.module.lru_cache
        def square(x):
            # [FUNCTION DOC] HELPER: Returns x squared.
            return x ** 2

        self.assertEqual(list(map(square, [10, 20, 10])),
                         [100, 400, 100])
        self.assertEqual(square.cache_info().hits, 1)
        self.assertEqual(square.cache_info().misses, 2)
        self.assertEqual(square.cache_info().maxsize, 128)
        self.assertEqual(square.cache_info().currsize, 2)

    def test_lru_bug_35780(self):
        # [FUNCTION DOC] Regression test (bug 35780): a recursive call made while the cache is
        # full and being modified must not corrupt the cache; its size stays at maxsize.
        # C version of the lru_cache was not checking to see if
        # the user function call has already modified the cache
        # (this arises in recursive calls and in multi-threading).
        # This cause the cache to have orphan links not referenced
        # by the cache dictionary.

        once = True                 # Modified by f(x) below

        @self.module.lru_cache(maxsize=10)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns a string for x; on the first call with 20 it
            # recurses once (the enclosing once flag) so the cache is changed during the user
            # function call.
            nonlocal once
            rv = f'.{x}.'
            if x == 20 and once:
                once = False
                rv = f(x)
            return rv

        # Fill the cache
        for x in range(15):
            self.assertEqual(f(x), f'.{x}.')
        self.assertEqual(f.cache_info().currsize, 10)

        # Make a recursive call and make sure the cache remains full
        self.assertEqual(f(20), '.20.')
        self.assertEqual(f.cache_info().currsize, 10)

    def test_lru_bug_36650(self):
        # [FUNCTION DOC] Regression test (bug 36650): a call with an empty keyword dict must hit
        # the same entry as a call without keywords.
        # C version of lru_cache was treating a call with an empty **kwargs
        # dictionary as being distinct from a call with no keywords at all.
        # This did not result in an incorrect answer, but it did trigger
        # an unexpected cache miss.

        @self.module.lru_cache()
        def f(x):
            # [FUNCTION DOC] HELPER: Does nothing.
            pass

        f(0)
        f(0, **{})
        self.assertEqual(f.cache_info().hits, 1)

    def test_lru_hash_only_once(self):
        # [FUNCTION DOC] Checks the cache hashes each argument once per call: a mock argument's
        # hash count increases by one on insert and on a hit, not on eviction of another key, and
        # by one again on a miss; cache_info is verified after every step.
        # To protect against weird reentrancy bugs and to improve
        # efficiency when faced with slow __hash__ methods, the
        # LRU cache guarantees that it will only call __hash__
        # only once per use as an argument to the cached function.

        @self.module.lru_cache(maxsize=1)
        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns x times 3 plus y.
            return x * 3 + y

        # Simulate the integer 5
        mock_int = unittest.mock.Mock()
        mock_int.__mul__ = unittest.mock.Mock(return_value=15)
        mock_int.__hash__ = unittest.mock.Mock(return_value=999)

        # Add to cache:  One use as an argument gives one call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 1)
        self.assertEqual(f.cache_info(), (0, 1, 1, 1))

        # Cache hit: One use as an argument gives one additional call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 2)
        self.assertEqual(f.cache_info(), (1, 1, 1, 1))

        # Cache eviction: No use as an argument gives no additional call
        self.assertEqual(f(6, 2), 20)
        self.assertEqual(mock_int.__hash__.call_count, 2)
        self.assertEqual(f.cache_info(), (1, 2, 1, 1))

        # Cache miss: One use as an argument gives one additional call
        self.assertEqual(f(mock_int, 1), 16)
        self.assertEqual(mock_int.__hash__.call_count, 3)
        self.assertEqual(f.cache_info(), (1, 3, 1, 1))

    def test_lru_reentrancy_with_len(self):
        # [FUNCTION DOC] Checks the cache does not depend on the builtin len: temporarily replaces
        # the builtin len with a cached version, verifies results, and restores it in a finally.
        # Test to make sure the LRU cache code isn't thrown-off by
        # caching the built-in len() function.  Since len() can be
        # cached, we shouldn't use it inside the lru code itself.
        old_len = builtins.len
        try:
            builtins.len = self.module.lru_cache(4)(len)
            for i in [0, 0, 1, 2, 3, 3, 4, 5, 6, 1, 7, 2, 1]:
                self.assertEqual(len('abcdefghijklmn'[:i]), i)
        finally:
            builtins.len = old_len

    def test_lru_star_arg_handling(self):
        # [FUNCTION DOC] Regression check that varargs calls cache the star tuple and a single
        # tuple argument as different keys.
        # Test regression that arose in ea064ff3c10f
        @self.module.lru_cache()
        def f(*args):
            # [FUNCTION DOC] HELPER: Returns the positional arguments tuple.
            return args

        self.assertEqual(f(1, 2), (1, 2))
        self.assertEqual(f((1, 2)), ((1, 2),))

    def test_lru_type_error(self):
        # [FUNCTION DOC] Regression test (issue 28653): calling cached functions (unbounded and
        # bounded) with an unhashable list raises TypeError.
        # Regression test for issue #28653.
        # lru_cache was leaking when one of the arguments
        # wasn't cacheable.

        @self.module.lru_cache(maxsize=None)
        def infinite_cache(o):
            # [FUNCTION DOC] HELPER: Unbounded cached function that does nothing.
            pass

        @self.module.lru_cache(maxsize=10)
        def limited_cache(o):
            # [FUNCTION DOC] HELPER: Bounded cached function that does nothing.
            pass

        with self.assertRaises(TypeError):
            infinite_cache([])

        with self.assertRaises(TypeError):
            limited_cache([])

    def test_lru_with_maxsize_none(self):
        # [FUNCTION DOC] Checks an unbounded cache gives correct recursive Fibonacci values with
        # the expected hit and miss counts, and that clearing resets the info.
        @self.module.lru_cache(maxsize=None)
        def fib(n):
            # [FUNCTION DOC] HELPER: Cached recursive Fibonacci.
            if n < 2:
                return n
            return fib(n-1) + fib(n-2)
        self.assertEqual([fib(n) for n in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))

    def test_lru_with_maxsize_negative(self):
        # [FUNCTION DOC] Checks a negative maxsize is treated as 0: nothing is cached, all 300
        # calls miss, and reported maxsize is 0.
        @self.module.lru_cache(maxsize=-10)
        def eq(n):
            # [FUNCTION DOC] HELPER: Returns n.
            return n
        for i in (0, 1):
            self.assertEqual([eq(n) for n in range(150)], list(range(150)))
        self.assertEqual(eq.cache_info(),
            self.module._CacheInfo(hits=0, misses=300, maxsize=0, currsize=0))

    def test_lru_with_exceptions(self):
        # [FUNCTION DOC] Checks, for unbounded and bounded caches, that an exception from the user
        # function propagates with no chained context and that failures are not cached.
        # Verify that user_function exceptions get passed through without
        # creating a hard-to-read chained exception.
        # http://bugs.python.org/issue13177
        for maxsize in (None, 128):
            @self.module.lru_cache(maxsize)
            def func(i):
                # [FUNCTION DOC] HELPER: Returns the character of a short string at index i;
                # raises IndexError when out of range.
                return 'abc'[i]
            self.assertEqual(func(0), 'a')
            with self.assertRaises(IndexError) as cm:
                func(15)
            self.assertIsNone(cm.exception.__context__)
            # Verify that the previous exception did not result in a cached entry
            with self.assertRaises(IndexError):
                func(15)

    def test_lru_with_types(self):
        # [FUNCTION DOC] Checks typed caching separates int and float arguments (positional and
        # keyword) and counts four hits and four misses, for unbounded and bounded caches.
        for maxsize in (None, 128):
            @self.module.lru_cache(maxsize=maxsize, typed=True)
            def square(x):
                # [FUNCTION DOC] HELPER: Returns x times x.
                return x * x
            self.assertEqual(square(3), 9)
            self.assertEqual(type(square(3)), type(9))
            self.assertEqual(square(3.0), 9.0)
            self.assertEqual(type(square(3.0)), type(9.0))
            self.assertEqual(square(x=3), 9)
            self.assertEqual(type(square(x=3)), type(9))
            self.assertEqual(square(x=3.0), 9.0)
            self.assertEqual(type(square(x=3.0)), type(9.0))
            self.assertEqual(square.cache_info().hits, 4)
            self.assertEqual(square.cache_info().misses, 4)

    def test_lru_cache_typed_is_not_recursive(self):
        # [FUNCTION DOC] Checks typed caching only distinguishes top-level argument types: equal
        # values nested in tuples or tuple subclasses share a cache entry.
        cached = self.module.lru_cache(typed=True)(repr)

        self.assertEqual(cached(1), '1')
        self.assertEqual(cached(True), 'True')
        self.assertEqual(cached(1.0), '1.0')
        self.assertEqual(cached(0), '0')
        self.assertEqual(cached(False), 'False')
        self.assertEqual(cached(0.0), '0.0')

        self.assertEqual(cached((1,)), '(1,)')
        self.assertEqual(cached((True,)), '(1,)')
        self.assertEqual(cached((1.0,)), '(1,)')
        self.assertEqual(cached((0,)), '(0,)')
        self.assertEqual(cached((False,)), '(0,)')
        self.assertEqual(cached((0.0,)), '(0,)')

        class T(tuple):
            # [CLASS DOC] HELPER: Empty tuple subclass.
            pass

        self.assertEqual(cached(T((1,))), '(1,)')
        self.assertEqual(cached(T((True,))), '(1,)')
        self.assertEqual(cached(T((1.0,))), '(1,)')
        self.assertEqual(cached(T((0,))), '(0,)')
        self.assertEqual(cached(T((False,))), '(0,)')
        self.assertEqual(cached(T((0.0,))), '(0,)')

    def test_lru_with_keyword_args(self):
        # [FUNCTION DOC] Checks a cached recursive Fibonacci called through keyword arguments
        # gives the correct values and cache_info with default maxsize.
        @self.module.lru_cache()
        def fib(n):
            # [FUNCTION DOC] HELPER: Cached recursive Fibonacci calling itself with keyword args.
            if n < 2:
                return n
            return fib(n=n-1) + fib(n=n-2)
        self.assertEqual(
            [fib(n=number) for number in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610]
        )
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=128, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=128, currsize=0))

    def test_lru_with_keyword_args_maxsize_none(self):
        # [FUNCTION DOC] Checks keyword-argument calls with an unbounded cache give the correct
        # values and cache_info.
        @self.module.lru_cache(maxsize=None)
        def fib(n):
            # [FUNCTION DOC] HELPER: Cached recursive Fibonacci calling itself with keyword args.
            if n < 2:
                return n
            return fib(n=n-1) + fib(n=n-2)
        self.assertEqual([fib(n=number) for number in range(16)],
            [0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610])
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=28, misses=16, maxsize=None, currsize=16))
        fib.cache_clear()
        self.assertEqual(fib.cache_info(),
            self.module._CacheInfo(hits=0, misses=0, maxsize=None, currsize=0))

    def test_kwargs_order(self):
        # [FUNCTION DOC] Checks (PEP 468) that keyword order is part of the key: the same keywords
        # in a different order are two misses.
        # PEP 468: Preserving Keyword Argument Order
        @self.module.lru_cache(maxsize=10)
        def f(**kwargs):
            # [FUNCTION DOC] HELPER: Returns the keyword arguments as a list of pairs in
            # call order.
            return list(kwargs.items())
        self.assertEqual(f(a=1, b=2), [('a', 1), ('b', 2)])
        self.assertEqual(f(b=2, a=1), [('b', 2), ('a', 1)])
        self.assertEqual(f.cache_info(),
            self.module._CacheInfo(hits=0, misses=2, maxsize=10, currsize=2))

    def test_lru_cache_decoration(self):
        # [FUNCTION DOC] Checks the cache wrapper copies the attributes named in
        # WRAPPER_ASSIGNMENTS from the decorated function.
        def f(zomg: 'zomg_annotation'):
            # [FUNCTION DOC] HELPER: Documented, annotated function returning 42.
            """f doc string"""
            return 42
        g = self.module.lru_cache()(f)
        for attr in self.module.WRAPPER_ASSIGNMENTS:
            self.assertEqual(getattr(g, attr), getattr(f, attr))

    @threading_helper.requires_working_threading()
    def test_lru_cache_threaded(self):
        # [FUNCTION DOC] Stress test with threads: n threads fill the cache concurrently (the
        # Python version may have extra misses; the C version has exact counts), then a clearing
        # thread runs alongside filler threads. Uses a tiny switch interval, restored in finally.
        n, m = 5, 11
        def orig(x, y):
            # [FUNCTION DOC] HELPER: Returns three times x plus y.
            return 3 * x + y
        f = self.module.lru_cache(maxsize=n*m)(orig)
        hits, misses, maxsize, currsize = f.cache_info()
        self.assertEqual(currsize, 0)

        start = threading.Event()
        def full(k):
            # [FUNCTION DOC] HELPER: Waits for the start event and calls the cached function m
            # times for key k, checking each result against the original.
            start.wait(10)
            for _ in range(m):
                self.assertEqual(f(k, 0), orig(k, 0))

        def clear():
            # [FUNCTION DOC] HELPER: Waits for the start event and clears the cache 2*m times.
            start.wait(10)
            for _ in range(2*m):
                f.cache_clear()

        orig_si = sys.getswitchinterval()
        support.setswitchinterval(1e-6)
        try:
            # create n threads in order to fill cache
            threads = [threading.Thread(target=full, args=[k])
                       for k in range(n)]
            with threading_helper.start_threads(threads):
                start.set()

            hits, misses, maxsize, currsize = f.cache_info()
            if self.module is py_functools:
                # XXX: Why can be not equal?
                self.assertLessEqual(misses, n)
                self.assertLessEqual(hits, m*n - misses)
            else:
                self.assertEqual(misses, n)
                self.assertEqual(hits, m*n - misses)
            self.assertEqual(currsize, n)

            # create n threads in order to fill cache and 1 to clear it
            threads = [threading.Thread(target=clear)]
            threads += [threading.Thread(target=full, args=[k])
                        for k in range(n)]
            start.clear()
            with threading_helper.start_threads(threads):
                start.set()
        finally:
            sys.setswitchinterval(orig_si)

    @threading_helper.requires_working_threading()
    def test_lru_cache_threaded2(self):
        # [FUNCTION DOC] Checks simultaneous calls with identical arguments from n threads,
        # synchronized by barriers, are all counted as misses (nothing is a hit while all are
        # inside the function), and that size grows by one per round.
        # Simultaneous call with the same arguments
        n, m = 5, 7
        start = threading.Barrier(n+1)
        pause = threading.Barrier(n+1)
        stop = threading.Barrier(n+1)
        @self.module.lru_cache(maxsize=m*n)
        def f(x):
            # [FUNCTION DOC] HELPER: Blocks on the pause barrier, then returns three times x.
            pause.wait(10)
            return 3 * x
        self.assertEqual(f.cache_info(), (0, 0, m*n, 0))
        def test():
            # [FUNCTION DOC] HELPER: Thread body that, for each of m values, syncs on barriers and
            # checks the cached result.
            for i in range(m):
                start.wait(10)
                self.assertEqual(f(i), 3 * i)
                stop.wait(10)
        threads = [threading.Thread(target=test) for k in range(n)]
        with threading_helper.start_threads(threads):
            for i in range(m):
                start.wait(10)
                stop.reset()
                pause.wait(10)
                start.reset()
                stop.wait(10)
                pause.reset()
                self.assertEqual(f.cache_info(), (0, (i+1)*n, m*n, i+1))

    @threading_helper.requires_working_threading()
    def test_lru_cache_threaded3(self):
        # [FUNCTION DOC] Checks several threads calling a slow cached function with maxsize 2 all
        # get the correct result.
        @self.module.lru_cache(maxsize=2)
        def f(x):
            # [FUNCTION DOC] HELPER: Sleeps briefly and returns three times x.
            time.sleep(.01)
            return 3 * x
        def test(i, x):
            # [FUNCTION DOC] HELPER: Thread body checking the cached result for x, labelled
            # with i.
            self.assertEqual(f(x), 3 * x, i)
        threads = [threading.Thread(target=test, args=(i, v))
                   for i, v in enumerate([1, 2, 2, 3, 2])]
        with threading_helper.start_threads(threads):
            pass

    def test_need_for_rlock(self):
        # [FUNCTION DOC] Checks the cache lock is reentrant: an equality check triggered inside a
        # cache lookup calls the cached function again; with a plain lock this would deadlock.
        # This will deadlock on an LRU cache that uses a regular lock

        @self.module.lru_cache(maxsize=10)
        def test_func(x):
            # [FUNCTION DOC] HELPER: Cached function returning its argument.
            'Used to demonstrate a reentrant lru_cache call within a single thread'
            return x

        class DoubleEq:
            # [CLASS DOC] HELPER: Key type whose equality for value 2 re-enters the
            # cached function.
            'Demonstrate a reentrant lru_cache call within a single thread'
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores x.
                self.x = x
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Hashes to x.
                return self.x
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: When x is 2, calls the cached function with another key
                # (a re-entrant call); returns whether x values match.
                if self.x == 2:
                    test_func(DoubleEq(1))
                return self.x == other.x

        test_func(DoubleEq(1))                      # Load the cache
        test_func(DoubleEq(2))                      # Load the cache
        self.assertEqual(test_func(DoubleEq(2)),    # Trigger a re-entrant __eq__ call
                         DoubleEq(2))               # Verify the correct return value

    def test_lru_method(self):
        # [FUNCTION DOC] Checks lru_cache on a method of an int subclass: the cache is shared by
        # all instances (self is part of the key), with per-instance call counts and shared
        # cache_info through instances and the class.
        class X(int):
            # [CLASS DOC] HELPER: Int subclass with a call counter and a cached method returning x
            # times 10 plus its own value.
            f_cnt = 0
            @self.module.lru_cache(2)
            def f(self, x):
                # [FUNCTION DOC] HELPER: Increments the instance call counter and returns x times
                # 10 plus the integer value of self.
                self.f_cnt += 1
                return x*10+self
        a = X(5)
        b = X(5)
        c = X(7)
        self.assertEqual(X.f.cache_info(), (0, 0, 2, 0))

        for x in 1, 2, 2, 3, 1, 1, 1, 2, 3, 3:
            self.assertEqual(a.f(x), x*10 + 5)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 0, 0))
        self.assertEqual(X.f.cache_info(), (4, 6, 2, 2))

        for x in 1, 2, 1, 1, 1, 1, 3, 2, 2, 2:
            self.assertEqual(b.f(x), x*10 + 5)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 4, 0))
        self.assertEqual(X.f.cache_info(), (10, 10, 2, 2))

        for x in 2, 1, 1, 1, 1, 2, 1, 3, 2, 1:
            self.assertEqual(c.f(x), x*10 + 7)
        self.assertEqual((a.f_cnt, b.f_cnt, c.f_cnt), (6, 4, 5))
        self.assertEqual(X.f.cache_info(), (15, 15, 2, 2))

        self.assertEqual(a.f.cache_info(), X.f.cache_info())
        self.assertEqual(b.f.cache_info(), X.f.cache_info())
        self.assertEqual(c.f.cache_info(), X.f.cache_info())

    def test_pickle(self):
        # [FUNCTION DOC] Checks cached functions, cached methods and cached static methods pickle
        # by reference and unpickle to the identical object for every protocol.
        cls = self.__class__
        for f in cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth:
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                with self.subTest(proto=proto, func=f):
                    f_copy = pickle.loads(pickle.dumps(f, proto))
                    self.assertIs(f_copy, f)

    def test_copy(self):
        # [FUNCTION DOC] Checks shallow copy of cached functions (including a cached partial)
        # returns the same object.
        cls = self.__class__
        def orig(x, y):
            # [FUNCTION DOC] HELPER: Returns three times x plus y.
            return 3 * x + y
        part = self.module.partial(orig, 2)
        funcs = (cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth,
                 self.module.lru_cache(2)(part))
        for f in funcs:
            with self.subTest(func=f):
                f_copy = copy.copy(f)
                self.assertIs(f_copy, f)

    def test_deepcopy(self):
        # [FUNCTION DOC] Checks deep copy of cached functions (including a cached partial) returns
        # the same object.
        cls = self.__class__
        def orig(x, y):
            # [FUNCTION DOC] HELPER: Returns three times x plus y.
            return 3 * x + y
        part = self.module.partial(orig, 2)
        funcs = (cls.cached_func[0], cls.cached_meth, cls.cached_staticmeth,
                 self.module.lru_cache(2)(part))
        for f in funcs:
            with self.subTest(func=f):
                f_copy = copy.deepcopy(f)
                self.assertIs(f_copy, f)

    def test_lru_cache_parameters(self):
        # [FUNCTION DOC] Checks cache_parameters reports the maxsize and typed settings.
        @self.module.lru_cache(maxsize=2)
        def f():
            # [FUNCTION DOC] HELPER: Returns 1.
            return 1
        self.assertEqual(f.cache_parameters(), {'maxsize': 2, "typed": False})

        @self.module.lru_cache(maxsize=1000, typed=True)
        def f():
            # [FUNCTION DOC] HELPER: Returns 1.
            return 1
        self.assertEqual(f.cache_parameters(), {'maxsize': 1000, "typed": True})

    @support.suppress_immortalization()
    def test_lru_cache_weakrefable(self):
        # [FUNCTION DOC] Checks cached functions, methods and static methods can be weakly
        # referenced and are freed after the owners are deleted and a collection runs.
        @self.module.lru_cache
        def test_function(x):
            # [FUNCTION DOC] HELPER: Returns x.
            return x

        class A:
            # [CLASS DOC] HELPER: Class with a cached method and a cached static method.
            @self.module.lru_cache
            def test_method(self, x):
                # [FUNCTION DOC] HELPER: Returns the pair of self and x.
                return (self, x)

            @staticmethod
            @self.module.lru_cache
            def test_staticmethod(x):
                # [FUNCTION DOC] HELPER: Returns the pair of the enclosing test instance (not a
                # parameter) and x.
                return (self, x)

        refs = [weakref.ref(test_function),
                weakref.ref(A.test_method),
                weakref.ref(A.test_staticmethod)]

        for ref in refs:
            self.assertIsNotNone(ref())

        del A
        del test_function
        gc.collect()

        for ref in refs:
            self.assertIsNone(ref())

    def test_common_signatures(self):
        # [FUNCTION DOC] Checks the signature of a cached function matches the original and that
        # cache_info and cache_clear have empty signatures. The original is a one-line stub that
        # has no block of its own.
        def orig(a, /, b, c=True): ...
        lru = self.module.lru_cache(1)(orig)

        self.assertEqual(str(Signature.from_callable(lru)), '(a, /, b, c=True)')
        self.assertEqual(str(Signature.from_callable(lru.cache_info)), '()')
        self.assertEqual(str(Signature.from_callable(lru.cache_clear)), '()')

    @support.skip_on_s390x
    @unittest.skipIf(support.is_wasi, "WASI has limited C stack")
    def test_lru_recursion(self):
        # [FUNCTION DOC] Checks deep recursion through a cached Fibonacci works (outside debug
        # builds) and, for the C module, that very deep recursion raises RecursionError. Skipped
        # on s390x and WASI.

        @self.module.lru_cache
        def fib(n):
            # [FUNCTION DOC] HELPER: Cached recursive Fibonacci.
            if n <= 1:
                return n
            return fib(n-1) + fib(n-2)

        if not support.Py_DEBUG:
            depth = support.get_c_recursion_limit()*2//7
            with support.infinite_recursion():
                fib(depth)
        if self.module == c_functools:
            fib.cache_clear()
            with support.infinite_recursion():
                with self.assertRaises(RecursionError):
                    fib(10000)


@py_functools.lru_cache()
def py_cached_func(x, y):
    # [FUNCTION DOC] HELPER: Module-level function cached by the Python lru_cache; returns three
    # times x plus y.
    return 3 * x + y

if c_functools:
    @c_functools.lru_cache()
    def c_cached_func(x, y):
        # [FUNCTION DOC] HELPER: Module-level function cached by the C lru_cache; returns three
        # times x plus y.
        return 3 * x + y


class TestLRUPy(_TestLRUU, CPythonTestCase):
    # [CLASS DOC] Runs lru_cache tests with the Python implementation; provides the cached_func
    # tuple and a cached method and static method.
    module = py_functools
    cached_func = py_cached_func,

    @module.lru_cache()
    def cached_meth(self, x, y):
        # [FUNCTION DOC] HELPER: Cached method returning three times x plus y.
        return 3 * x + y

    @staticmethod
    @module.lru_cache()
    def cached_staticmeth(x, y):
        # [FUNCTION DOC] HELPER: Cached static method returning three times x plus y.
        return 3 * x + y


@unittest.skipUnless(c_functools, 'requires the C _functools module')
class TestLRUC(_TestLRUU, CPythonTestCase):
    # [CLASS DOC] Runs lru_cache tests with the C implementation (skipped when _functools is
    # missing); provides the cached_func tuple and a cached method and static method.
    if c_functools:
        module = c_functools
        cached_func = c_cached_func,

        @module.lru_cache()
        def cached_meth(self, x, y):
            # [FUNCTION DOC] HELPER: Cached method returning three times x plus y.
            return 3 * x + y

        @staticmethod
        @module.lru_cache()
        def cached_staticmeth(x, y):
            # [FUNCTION DOC] HELPER: Cached static method returning three times x plus y.
            return 3 * x + y


class TestSingleDispatch(CPythonTestCase):
    # [CLASS DOC] Tests for functools.singledispatch: registration, MRO and ABC resolution,
    # caching, annotations and methods. No shared state besides CPythonTestCase.
    def test_simple_overloads(self):
        # [FUNCTION DOC] Checks a base implementation plus a registered int function dispatches by
        # argument type.
        @functools.singledispatch
        def g(obj):
            # [FUNCTION DOC] HELPER: Default singledispatch implementation returning "base".
            return "base"
        def g_int(i):
            # [FUNCTION DOC] HELPER: Integer implementation returning "integer".
            return "integer"
        g.register(int, g_int)
        self.assertEqual(g("str"), "base")
        self.assertEqual(g(1), "integer")
        self.assertEqual(g([1,2,3]), "base")

    def test_mro(self):
        # [FUNCTION DOC] Checks dispatch follows the MRO: a diamond class D(C, B) with
        # registered A and B implementations resolves to B (C inherits A, B comes before A in
        # the linearization).
        @functools.singledispatch
        def g(obj):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        class A:
            # [CLASS DOC] HELPER: Root of the diamond hierarchy.
            pass
        class C(A):
            # [CLASS DOC] HELPER: Subclass of A.
            pass
        class B(A):
            # [CLASS DOC] HELPER: Subclass of A.
            pass
        class D(C, B):
            # [CLASS DOC] HELPER: Diamond subclass of C and B.
            pass
        def g_A(a):
            # [FUNCTION DOC] HELPER: Implementation for A returning "A".
            return "A"
        def g_B(b):
            # [FUNCTION DOC] HELPER: Implementation for B returning "B".
            return "B"
        g.register(A, g_A)
        g.register(B, g_B)
        self.assertEqual(g(A()), "A")
        self.assertEqual(g(B()), "B")
        self.assertEqual(g(C()), "A")
        self.assertEqual(g(D()), "B")

    def test_register_decorator(self):
        # [FUNCTION DOC] Checks register used as a decorator with a type: dispatch picks the int
        # implementation, dispatch() returns the registered function, and object and str resolve
        # to the same default.
        @functools.singledispatch
        def g(obj):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        @g.register(int)
        def g_int(i):
            # [FUNCTION DOC] HELPER: Int implementation returning a string with the number.
            return "int %s" % (i,)
        self.assertEqual(g(""), "base")
        self.assertEqual(g(12), "int 12")
        self.assertIs(g.dispatch(int), g_int)
        self.assertIs(g.dispatch(object), g.dispatch(str))
        # Note: in the assert above this is not g.
        # @singledispatch returns the wrapper.

    def test_wrapping_attributes(self):
        # [FUNCTION DOC] Checks the singledispatch wrapper keeps the function name and, unless
        # running with -OO, its docstring.
        @functools.singledispatch
        def g(obj):
            # [FUNCTION DOC] HELPER: Documented default implementation returning "Test".
            "Simple test"
            return "Test"
        self.assertEqual(g.__name__, "g")
        if sys.flags.optimize < 2:
            self.assertEqual(g.__doc__, "Simple test")

    @unittest.skipUnless(decimal, 'requires _decimal')
    @support.cpython_only
    def test_c_classes(self):
        # [FUNCTION DOC] Checks dispatch with classes from the C decimal module: a base exception
        # class implementation, then a more specific registration taking precedence. CPython only
        # and needs the C decimal module.
        @functools.singledispatch
        def g(obj):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        @g.register(decimal.DecimalException)
        def _(obj):
            # [FUNCTION DOC] HELPER: Implementation for decimal exceptions returning their args.
            return obj.args
        subn = decimal.Subnormal("Exponent < Emin")
        rnd = decimal.Rounded("Number got rounded")
        self.assertEqual(g(subn), ("Exponent < Emin",))
        self.assertEqual(g(rnd), ("Number got rounded",))
        @g.register(decimal.Subnormal)
        def _(obj):
            # [FUNCTION DOC] HELPER: Implementation for the Subnormal exception returning
            # a message.
            return "Too small to care."
        self.assertEqual(g(subn), "Too small to care.")
        self.assertEqual(g(rnd), ("Number got rounded",))

    def test_compose_mro(self):
        # [FUNCTION DOC] Checks the private _compose_mro helper produces the expected
        # linearization for built-in and abstract base classes regardless of the order of the
        # candidate ABCs (dict, ChainMap, defaultdict and subclasses with extra registrations or
        # call support).
        # None of the examples in this test depend on haystack ordering.
        c = collections.abc
        mro = functools._compose_mro
        bases = [c.Sequence, c.MutableMapping, c.Mapping, c.Set]
        for haystack in permutations(bases):
            m = mro(dict, haystack)
            self.assertEqual(m, [dict, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])
        bases = [c.Container, c.Mapping, c.MutableMapping, collections.OrderedDict]
        for haystack in permutations(bases):
            m = mro(collections.ChainMap, haystack)
            self.assertEqual(m, [collections.ChainMap, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])

        # If there's a generic function with implementations registered for
        # both Sized and Container, passing a defaultdict to it results in an
        # ambiguous dispatch which will cause a RuntimeError (see
        # test_mro_conflicts).
        bases = [c.Container, c.Sized, str]
        for haystack in permutations(bases):
            m = mro(collections.defaultdict, [c.Sized, c.Container, str])
            self.assertEqual(m, [collections.defaultdict, dict, c.Sized,
                                 c.Container, object])

        # MutableSequence below is registered directly on D. In other words, it
        # precedes MutableMapping which means single dispatch will always
        # choose MutableSequence here.
        class D(collections.defaultdict):
            # [CLASS DOC] HELPER: defaultdict subclass registered as a MutableSequence.
            pass
        c.MutableSequence.register(D)
        bases = [c.MutableSequence, c.MutableMapping]
        for haystack in permutations(bases):
            m = mro(D, haystack)
            self.assertEqual(m, [D, c.MutableSequence, c.Sequence, c.Reversible,
                                 collections.defaultdict, dict, c.MutableMapping, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable, c.Container,
                                 object])

        # Container and Callable are registered on different base classes and
        # a generic function supporting both should always pick the Callable
        # implementation if a C instance is passed.
        class C(collections.defaultdict):
            # [CLASS DOC] HELPER: Callable defaultdict subclass.
            def __call__(self):
                # [FUNCTION DOC] HELPER: Does nothing; makes instances Callable.
                pass
        bases = [c.Sized, c.Callable, c.Container, c.Mapping]
        for haystack in permutations(bases):
            m = mro(C, haystack)
            self.assertEqual(m, [C, c.Callable, collections.defaultdict, dict, c.Mapping,
                                 c.Collection, c.Sized, c.Iterable,
                                 c.Container, object])

    def test_register_abc(self):
        # [FUNCTION DOC] Checks registering ABCs step by step (Sized, MutableMapping, ChainMap,
        # MutableSequence, MutableSet, Mapping, Sequence, Set) and concrete types, verifying for a
        # dict, list, set, frozenset and tuple which implementation is chosen, favoring the most
        # specific one and ignoring irrelevant registrations.
        c = collections.abc
        d = {"a": "b"}
        l = [1, 2, 3]
        s = {object(), None}
        f = frozenset(s)
        t = (1, 2, 3)
        @functools.singledispatch
        def g(obj):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        self.assertEqual(g(d), "base")
        self.assertEqual(g(l), "base")
        self.assertEqual(g(s), "base")
        self.assertEqual(g(f), "base")
        self.assertEqual(g(t), "base")
        g.register(c.Sized, lambda obj: "sized")
        self.assertEqual(g(d), "sized")
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableMapping, lambda obj: "mutablemapping")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(collections.ChainMap, lambda obj: "chainmap")
        self.assertEqual(g(d), "mutablemapping")  # irrelevant ABCs registered
        self.assertEqual(g(l), "sized")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableSequence, lambda obj: "mutablesequence")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "sized")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.MutableSet, lambda obj: "mutableset")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.Mapping, lambda obj: "mapping")
        self.assertEqual(g(d), "mutablemapping")  # not specific enough
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sized")
        g.register(c.Sequence, lambda obj: "sequence")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "sized")
        self.assertEqual(g(t), "sequence")
        g.register(c.Set, lambda obj: "set")
        self.assertEqual(g(d), "mutablemapping")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(dict, lambda obj: "dict")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "mutablesequence")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(list, lambda obj: "list")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "mutableset")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(set, lambda obj: "concrete-set")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "set")
        self.assertEqual(g(t), "sequence")
        g.register(frozenset, lambda obj: "frozen-set")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "frozen-set")
        self.assertEqual(g(t), "sequence")
        g.register(tuple, lambda obj: "tuple")
        self.assertEqual(g(d), "dict")
        self.assertEqual(g(l), "list")
        self.assertEqual(g(s), "concrete-set")
        self.assertEqual(g(f), "frozen-set")
        self.assertEqual(g(t), "tuple")

    def test_c3_abc(self):
        # [FUNCTION DOC] Checks the private _c3_mro helper merges real bases and implicit ABCs
        # (Callable from a call method, Sized from a length method, Container by registration)
        # into the expected order for any ordering of ABCs; unrelated ABCs are ignored.
        c = collections.abc
        mro = functools._c3_mro
        class A(object):
            # [CLASS DOC] HELPER: Plain base class.
            pass
        class B(A):
            # [CLASS DOC] HELPER: Subclass of A defining length (implicitly Sized).
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0   # implies Sized
        @c.Container.register
        class C(object):
            # [CLASS DOC] HELPER: Plain class registered as a Container virtual subclass.
            pass
        class D(object):
            # [CLASS DOC] HELPER: Unrelated plain class.
            pass   # unrelated
        class X(D, C, B):
            # [CLASS DOC] HELPER: Class deriving from D, C and B and defining call
            # (implicitly Callable).
            def __call__(self):
                # [FUNCTION DOC] HELPER: Does nothing; makes instances Callable.
                pass   # implies Callable
        expected = [X, c.Callable, D, C, c.Container, B, c.Sized, A, object]
        for abcs in permutations([c.Sized, c.Callable, c.Container]):
            self.assertEqual(mro(X, abcs=abcs), expected)
        # unrelated ABCs don't appear in the resulting MRO
        many_abcs = [c.Mapping, c.Sized, c.Callable, c.Container, c.Iterable]
        self.assertEqual(mro(X, abcs=many_abcs), expected)

    def test_false_meta(self):
        # [FUNCTION DOC] Regression test (issue 23572): dispatch works for classes whose metaclass
        # defines a length method returning 0 (falsy classes).
        # see issue23572
        class MetaA(type):
            # [CLASS DOC] HELPER: Metaclass whose length is 0, so classes are falsy.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0
        class A(metaclass=MetaA):
            # [CLASS DOC] HELPER: Empty class using the falsy metaclass.
            pass
        class AA(A):
            # [CLASS DOC] HELPER: Subclass of A.
            pass
        @functools.singledispatch
        def fun(a):
            # [FUNCTION DOC] HELPER: Default implementation returning "base A".
            return 'base A'
        @fun.register(A)
        def _(a):
            # [FUNCTION DOC] HELPER: Implementation for A returning "fun A".
            return 'fun A'
        aa = AA()
        self.assertEqual(fun(aa), 'fun A')

    def test_mro_conflicts(self):
        # [FUNCTION DOC] Checks conflicts among implicitly or explicitly registered ABCs: more
        # specific registrations win, ambiguous registrations (for example Container versus
        # Iterable, or Container versus Sized) raise RuntimeError naming both classes in either
        # order, and explicit MRO membership takes precedence over inferred ABCs.
        c = collections.abc
        @functools.singledispatch
        def g(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        class O(c.Sized):
            # [CLASS DOC] HELPER: Explicit Sized subclass.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0
        o = O()
        self.assertEqual(g(o), "base")
        g.register(c.Iterable, lambda arg: "iterable")
        g.register(c.Container, lambda arg: "container")
        g.register(c.Sized, lambda arg: "sized")
        g.register(c.Set, lambda arg: "set")
        self.assertEqual(g(o), "sized")
        c.Iterable.register(O)
        self.assertEqual(g(o), "sized")   # because it's explicitly in __mro__
        c.Container.register(O)
        self.assertEqual(g(o), "sized")   # see above: Sized is in __mro__
        c.Set.register(O)
        self.assertEqual(g(o), "set")     # because c.Set is a subclass of
                                          # c.Sized and c.Container
        class P:
            # [CLASS DOC] HELPER: Empty plain class later registered with ABCs.
            pass
        p = P()
        self.assertEqual(g(p), "base")
        c.Iterable.register(P)
        self.assertEqual(g(p), "iterable")
        c.Container.register(P)
        with self.assertRaises(RuntimeError) as re_one:
            g(p)
        self.assertIn(
            str(re_one.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Iterable'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Iterable'> "
              "or <class 'collections.abc.Container'>")),
        )

        class Q(c.Sized):
            # [CLASS DOC] HELPER: Explicit Sized subclass.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0
        q = Q()
        self.assertEqual(g(q), "sized")
        c.Iterable.register(Q)
        self.assertEqual(g(q), "sized")   # because it's explicitly in __mro__
        c.Set.register(Q)
        self.assertEqual(g(q), "set")     # because c.Set is a subclass of
                                          # c.Sized and c.Iterable
        @functools.singledispatch
        def h(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        @h.register(c.Sized)
        def _(arg):
            # [FUNCTION DOC] HELPER: Sized implementation returning "sized".
            return "sized"
        @h.register(c.Container)
        def _(arg):
            # [FUNCTION DOC] HELPER: Container implementation returning "container".
            return "container"
        # Even though Sized and Container are explicit bases of MutableMapping,
        # this ABC is implicitly registered on defaultdict which makes all of
        # MutableMapping's bases implicit as well from defaultdict's
        # perspective.
        with self.assertRaises(RuntimeError) as re_two:
            h(collections.defaultdict(lambda: 0))
        self.assertIn(
            str(re_two.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Sized'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Sized'> "
              "or <class 'collections.abc.Container'>")),
        )
        class R(collections.defaultdict):
            # [CLASS DOC] HELPER: defaultdict subclass registered as a MutableSequence.
            pass
        c.MutableSequence.register(R)
        @functools.singledispatch
        def i(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        @i.register(c.MutableMapping)
        def _(arg):
            # [FUNCTION DOC] HELPER: MutableMapping implementation returning "mapping".
            return "mapping"
        @i.register(c.MutableSequence)
        def _(arg):
            # [FUNCTION DOC] HELPER: MutableSequence implementation returning "sequence".
            return "sequence"
        r = R()
        self.assertEqual(i(r), "sequence")
        class S:
            # [CLASS DOC] HELPER: Empty plain class.
            pass
        class T(S, c.Sized):
            # [CLASS DOC] HELPER: Subclass of S and the Sized ABC.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0
        t = T()
        self.assertEqual(h(t), "sized")
        c.Container.register(T)
        self.assertEqual(h(t), "sized")   # because it's explicitly in the MRO
        class U:
            # [CLASS DOC] HELPER: Plain class with a length method (implicitly Sized).
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0
        u = U()
        self.assertEqual(h(u), "sized")   # implicit Sized subclass inferred
                                          # from the existence of __len__()
        c.Container.register(U)
        # There is no preference for registered versus inferred ABCs.
        with self.assertRaises(RuntimeError) as re_three:
            h(u)
        self.assertIn(
            str(re_three.exception),
            (("Ambiguous dispatch: <class 'collections.abc.Container'> "
              "or <class 'collections.abc.Sized'>"),
             ("Ambiguous dispatch: <class 'collections.abc.Sized'> "
              "or <class 'collections.abc.Container'>")),
        )
        class V(c.Sized, S):
            # [CLASS DOC] HELPER: Subclass of the Sized ABC and S.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Returns 0.
                return 0
        @functools.singledispatch
        def j(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        @j.register(S)
        def _(arg):
            # [FUNCTION DOC] HELPER: Implementation for S returning "s".
            return "s"
        @j.register(c.Container)
        def _(arg):
            # [FUNCTION DOC] HELPER: Container implementation returning "container".
            return "container"
        v = V()
        self.assertEqual(j(v), "s")
        c.Container.register(V)
        self.assertEqual(j(v), "container")   # because it ends up right after
                                              # Sized in the MRO

    def test_cache_invalidation(self):
        # [FUNCTION DOC] Checks the dispatch cache: swaps in a tracing dictionary for the
        # weak-key dictionary and verifies which classes are cached, how often the cache is
        # read/written, and that registering new implementations, registering ABC virtual
        # subclasses, and an explicit cache clear invalidate it as expected. The weakref
        # attribute is restored on exit.
        from collections import UserDict
        import weakref

        class TracingDict(UserDict):
            # [CLASS DOC] HELPER: UserDict recording the keys of each set and get operation in the
            # lists set_ops and get_ops, used in place of the dispatch cache.
            def __init__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Initializes the user dict and creates empty lists of
                # recorded set and get operations.
                super(TracingDict, self).__init__(*args, **kwargs)
                self.set_ops = []
                self.get_ops = []
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Looks up the key (KeyError if missing); on success
                # records it in get_ops and returns the value.
                result = self.data[key]
                self.get_ops.append(key)
                return result
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Records the key in set_ops and stores the value.
                self.set_ops.append(key)
                self.data[key] = value
            def clear(self):
                # [FUNCTION DOC] HELPER: Empties the data without recording anything.
                self.data.clear()

        td = TracingDict()
        with support.swap_attr(weakref, "WeakKeyDictionary", lambda: td):
            c = collections.abc
            @functools.singledispatch
            def g(arg):
                # [FUNCTION DOC] HELPER: Default implementation returning "base".
                return "base"
            d = {}
            l = []
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "base")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [])
            self.assertEqual(td.set_ops, [dict])
            self.assertEqual(td.data[dict], g.registry[object])
            self.assertEqual(g(l), "base")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [])
            self.assertEqual(td.set_ops, [dict, list])
            self.assertEqual(td.data[dict], g.registry[object])
            self.assertEqual(td.data[list], g.registry[object])
            self.assertEqual(td.data[dict], td.data[list])
            self.assertEqual(g(l), "base")
            self.assertEqual(g(d), "base")
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list])
            g.register(list, lambda arg: "list")
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "base")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict])
            self.assertEqual(td.data[dict],
                             functools._find_impl(dict, g.registry))
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list])
            self.assertEqual(td.data[list],
                             functools._find_impl(list, g.registry))
            class X:
                # [CLASS DOC] HELPER: Empty class registered with ABCs to test cache invalidation.
                pass
            c.MutableMapping.register(X)   # Will not invalidate the cache,
                                           # not using ABCs yet.
            self.assertEqual(g(d), "base")
            self.assertEqual(g(l), "list")
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list])
            g.register(c.Sized, lambda arg: "sized")
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "sized")
            self.assertEqual(len(td), 1)
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict])
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            self.assertEqual(td.get_ops, [list, dict, dict, list])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            self.assertEqual(g(l), "list")
            self.assertEqual(g(d), "sized")
            self.assertEqual(td.get_ops, [list, dict, dict, list, list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            g.dispatch(list)
            g.dispatch(dict)
            self.assertEqual(td.get_ops, [list, dict, dict, list, list, dict,
                                          list, dict])
            self.assertEqual(td.set_ops, [dict, list, dict, list, dict, list])
            c.MutableSet.register(X)       # Will invalidate the cache.
            self.assertEqual(len(td), 2)   # Stale cache.
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 1)
            g.register(c.MutableMapping, lambda arg: "mutablemapping")
            self.assertEqual(len(td), 0)
            self.assertEqual(g(d), "mutablemapping")
            self.assertEqual(len(td), 1)
            self.assertEqual(g(l), "list")
            self.assertEqual(len(td), 2)
            g.register(dict, lambda arg: "dict")
            self.assertEqual(g(d), "dict")
            self.assertEqual(g(l), "list")
            g._clear_cache()
            self.assertEqual(len(td), 0)

    def test_annotations(self):
        # [FUNCTION DOC] Checks register infers the dispatch type from the first parameter's
        # annotation (a class or a string forward reference), and that a class registered as a
        # callable needs an explicit type argument.
        @functools.singledispatch
        def i(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        @i.register
        def _(arg: collections.abc.Mapping):
            # [FUNCTION DOC] HELPER: Mapping-annotated implementation returning "mapping".
            return "mapping"
        @i.register
        def _(arg: "collections.abc.Sequence"):
            # [FUNCTION DOC] HELPER: Sequence-annotated implementation (string annotation)
            # returning "sequence".
            return "sequence"
        self.assertEqual(i(None), "base")
        self.assertEqual(i({"a": 1}), "mapping")
        self.assertEqual(i([1, 2, 3]), "sequence")
        self.assertEqual(i((1, 2, 3)), "sequence")
        self.assertEqual(i("str"), "sequence")

        # Registering classes as callables doesn't work with annotations,
        # you need to pass the type explicitly.
        @i.register(str)
        class _:
            # [CLASS DOC] HELPER: Callable class registered for str; stores its argument and
            # compares equal to values equal to it.
            def __init__(self, arg):
                # [FUNCTION DOC] HELPER: Stores the argument.
                self.arg = arg

            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Equal if the stored argument equals the other value.
                return self.arg == other
        self.assertEqual(i("str"), "str")

    def test_method_register(self):
        # [FUNCTION DOC] Checks singledispatchmethod on instance methods: int, str and default
        # implementations set an arg attribute on the instance and do not affect other instances.
        class A:
            # [CLASS DOC] HELPER: Class with a singledispatchmethod and int and str registrations
            # that record which one ran in an instance attribute.
            @functools.singledispatchmethod
            def t(self, arg):
                # [FUNCTION DOC] HELPER: Default implementation storing "base" in the
                # arg attribute.
                self.arg = "base"
            @t.register(int)
            def _(self, arg):
                # [FUNCTION DOC] HELPER: Int implementation storing "int".
                self.arg = "int"
            @t.register(str)
            def _(self, arg):
                # [FUNCTION DOC] HELPER: Str implementation storing "str".
                self.arg = "str"
        a = A()

        a.t(0)
        self.assertEqual(a.arg, "int")
        aa = A()
        self.assertFalse(hasattr(aa, 'arg'))
        a.t('')
        self.assertEqual(a.arg, "str")
        aa = A()
        self.assertFalse(hasattr(aa, 'arg'))
        a.t(0.0)
        self.assertEqual(a.arg, "base")
        aa = A()
        self.assertFalse(hasattr(aa, 'arg'))

    def test_staticmethod_register(self):
        # [FUNCTION DOC] Checks singledispatchmethod over static methods dispatches by argument
        # type when called on the class.
        class A:
            # [CLASS DOC] HELPER: Class with a static singledispatch method and int and
            # str variants.
            @functools.singledispatchmethod
            @staticmethod
            def t(arg):
                # [FUNCTION DOC] HELPER: Default static implementation returning the argument.
                return arg
            @t.register(int)
            @staticmethod
            def _(arg):
                # [FUNCTION DOC] HELPER: Int static implementation returning whether the argument
                # is an int.
                return isinstance(arg, int)
            @t.register(str)
            @staticmethod
            def _(arg):
                # [FUNCTION DOC] HELPER: Str static implementation returning whether the argument
                # is a str.
                return isinstance(arg, str)
        a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(''))
        self.assertEqual(A.t(0.0), 0.0)

    def test_slotted_class(self):
        # [FUNCTION DOC] Checks singledispatchmethod works on a class using slots (no instance
        # dict), with dispatch on an annotated parameter.
        class Slot:
            # [CLASS DOC] HELPER: Slotted class with a singledispatchmethod.
            __slots__ = ('a', 'b')
            @functools.singledispatchmethod
            def go(self, item, arg):
                # [FUNCTION DOC] HELPER: Default implementation that does nothing.
                pass

            @go.register
            def _(self, item: int, arg):
                # [FUNCTION DOC] HELPER: Int-annotated implementation returning the sum of item
                # and arg.
                return item + arg

        s = Slot()
        self.assertEqual(s.go(1, 1), 2)

    def test_classmethod_slotted_class(self):
        # [FUNCTION DOC] Checks singledispatchmethod over class methods on a slotted class, called
        # from both instance and class.
        class Slot:
            # [CLASS DOC] HELPER: Slotted class with a class-method singledispatchmethod.
            __slots__ = ('a', 'b')
            @functools.singledispatchmethod
            @classmethod
            def go(cls, item, arg):
                # [FUNCTION DOC] HELPER: Default class-method implementation that does nothing.
                pass

            @go.register
            @classmethod
            def _(cls, item: int, arg):
                # [FUNCTION DOC] HELPER: Int-annotated class-method implementation returning
                # the sum.
                return item + arg

        s = Slot()
        self.assertEqual(s.go(1, 1), 2)
        self.assertEqual(Slot.go(1, 1), 2)

    def test_staticmethod_slotted_class(self):
        # [FUNCTION DOC] Checks singledispatchmethod over static methods on a slotted class,
        # called from both class and instance.
        class A:
            # [CLASS DOC] HELPER: Slotted class with a static singledispatch method and int and
            # str variants.
            __slots__ = ['a']
            @functools.singledispatchmethod
            @staticmethod
            def t(arg):
                # [FUNCTION DOC] HELPER: Default static implementation returning the argument.
                return arg
            @t.register(int)
            @staticmethod
            def _(arg):
                # [FUNCTION DOC] HELPER: Int static implementation returning whether the argument
                # is an int.
                return isinstance(arg, int)
            @t.register(str)
            @staticmethod
            def _(arg):
                # [FUNCTION DOC] HELPER: Str static implementation returning whether the argument
                # is a str.
                return isinstance(arg, str)
        a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(''))
        self.assertEqual(A.t(0.0), 0.0)
        self.assertTrue(a.t(0))
        self.assertTrue(a.t(''))
        self.assertEqual(a.t(0.0), 0.0)

    def test_assignment_behavior(self):
        # [FUNCTION DOC] Checks (gh-106448) that setting an attribute on a bound dispatch method
        # of one instance does not leak to the method of another instance (AttributeError).
        # see gh-106448
        class A:
            # [CLASS DOC] HELPER: Class with a singledispatchmethod.
            @functools.singledispatchmethod
            def t(arg):
                # [FUNCTION DOC] HELPER: Default implementation returning its argument.
                return arg

        a = A()
        a.t.foo = 'bar'
        a2 = A()
        with self.assertRaises(AttributeError):
            a2.t.foo

    def test_classmethod_register(self):
        # [FUNCTION DOC] Checks singledispatchmethod over class methods creates instances of the
        # class holding the name of the chosen implementation.
        class A:
            # [CLASS DOC] HELPER: Class storing its constructor argument, with a class-method
            # dispatch method whose variants build instances.
            def __init__(self, arg):
                # [FUNCTION DOC] HELPER: Stores the argument.
                self.arg = arg

            @functools.singledispatchmethod
            @classmethod
            def t(cls, arg):
                # [FUNCTION DOC] HELPER: Default class-method implementation creating an instance
                # with "base".
                return cls("base")
            @t.register(int)
            @classmethod
            def _(cls, arg):
                # [FUNCTION DOC] HELPER: Int class-method implementation creating an instance
                # with "int".
                return cls("int")
            @t.register(str)
            @classmethod
            def _(cls, arg):
                # [FUNCTION DOC] HELPER: Str class-method implementation creating an instance
                # with "str".
                return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t('').arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_callable_register(self):
        # [FUNCTION DOC] Checks register can be applied to class-method implementations after the
        # class is defined, via the class attribute.
        class A:
            # [CLASS DOC] HELPER: Class storing its constructor argument, with a class-method
            # dispatch method.
            def __init__(self, arg):
                # [FUNCTION DOC] HELPER: Stores the argument.
                self.arg = arg

            @functools.singledispatchmethod
            @classmethod
            def t(cls, arg):
                # [FUNCTION DOC] HELPER: Default class-method implementation creating an instance
                # with "base".
                return cls("base")

        @A.t.register(int)
        @classmethod
        def _(cls, arg):
            # [FUNCTION DOC] HELPER: Int class-method implementation creating an instance
            # with "int".
            return cls("int")
        @A.t.register(str)
        @classmethod
        def _(cls, arg):
            # [FUNCTION DOC] HELPER: Str class-method implementation creating an instance
            # with "str".
            return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t('').arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_abstractmethod_register(self):
        # [FUNCTION DOC] Checks singledispatchmethod propagates the abstract flag from an abstract
        # method so the class cannot be instantiated (TypeError).
        class Abstract(metaclass=abc.ABCMeta):
            # [CLASS DOC] HELPER: ABC with an abstract singledispatchmethod.

            @functools.singledispatchmethod
            @abc.abstractmethod
            def add(self, x, y):
                # [FUNCTION DOC] HELPER: Abstract placeholder method.
                pass

        self.assertTrue(Abstract.add.__isabstractmethod__)
        self.assertTrue(Abstract.__dict__['add'].__isabstractmethod__)

        with self.assertRaises(TypeError):
            Abstract()

    def test_type_ann_register(self):
        # [FUNCTION DOC] Checks singledispatchmethod picks implementations from the annotation of
        # the first non-self parameter (int, str) with a default for other types.
        class A:
            # [CLASS DOC] HELPER: Class with a dispatch method and annotation-registered int and
            # str variants returning their names.
            @functools.singledispatchmethod
            def t(self, arg):
                # [FUNCTION DOC] HELPER: Default implementation returning "base".
                return "base"
            @t.register
            def _(self, arg: int):
                # [FUNCTION DOC] HELPER: Int-annotated implementation returning "int".
                return "int"
            @t.register
            def _(self, arg: str):
                # [FUNCTION DOC] HELPER: Str-annotated implementation returning "str".
                return "str"
        a = A()

        self.assertEqual(a.t(0), "int")
        self.assertEqual(a.t(''), "str")
        self.assertEqual(a.t(0.0), "base")

    def test_staticmethod_type_ann_register(self):
        # [FUNCTION DOC] Checks annotation-based registration of static methods under a
        # singledispatchmethod, called through the class.
        class A:
            # [CLASS DOC] HELPER: Class with a static dispatch method and
            # annotation-registered variants.
            @functools.singledispatchmethod
            @staticmethod
            def t(arg):
                # [FUNCTION DOC] HELPER: Default static implementation returning the argument.
                return arg
            @t.register
            @staticmethod
            def _(arg: int):
                # [FUNCTION DOC] HELPER: Int-annotated static implementation returning whether it
                # is an int.
                return isinstance(arg, int)
            @t.register
            @staticmethod
            def _(arg: str):
                # [FUNCTION DOC] HELPER: Str-annotated static implementation returning whether it
                # is a str.
                return isinstance(arg, str)
        a = A()

        self.assertTrue(A.t(0))
        self.assertTrue(A.t(''))
        self.assertEqual(A.t(0.0), 0.0)

    def test_classmethod_type_ann_register(self):
        # [FUNCTION DOC] Checks annotation-based registration of class methods under a
        # singledispatchmethod; each variant builds an instance recording its name.
        class A:
            # [CLASS DOC] HELPER: Class storing its constructor argument with a class-method
            # dispatch method and annotation-registered variants.
            def __init__(self, arg):
                # [FUNCTION DOC] HELPER: Stores the argument.
                self.arg = arg

            @functools.singledispatchmethod
            @classmethod
            def t(cls, arg):
                # [FUNCTION DOC] HELPER: Default class-method implementation creating an instance
                # with "base".
                return cls("base")
            @t.register
            @classmethod
            def _(cls, arg: int):
                # [FUNCTION DOC] HELPER: Int-annotated implementation creating an instance
                # with "int".
                return cls("int")
            @t.register
            @classmethod
            def _(cls, arg: str):
                # [FUNCTION DOC] HELPER: Str-annotated implementation creating an instance
                # with "str".
                return cls("str")

        self.assertEqual(A.t(0).arg, "int")
        self.assertEqual(A.t('').arg, "str")
        self.assertEqual(A.t(0.0).arg, "base")

    def test_method_wrapping_attributes(self):
        # [FUNCTION DOC] Checks that bound and unbound singledispatchmethod objects (instance,
        # class and static variants) expose the qualified name, docstring (when docstrings are
        # kept), annotations and short name of the wrapped function.
        class A:
            # [CLASS DOC] HELPER: Class with documented, annotated dispatch methods of instance,
            # class and static kinds.
            @functools.singledispatchmethod
            def func(self, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Returns the argument as a string.
                """My function docstring"""
                return str(arg)
            @functools.singledispatchmethod
            @classmethod
            def cls_func(cls, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Class-method variant returning the argument as a string.
                """My function docstring"""
                return str(arg)
            @functools.singledispatchmethod
            @staticmethod
            def static_func(arg: int) -> str:
                # [FUNCTION DOC] HELPER: Static variant returning the argument as a string.
                """My function docstring"""
                return str(arg)

        prefix = A.__qualname__ + '.'
        for meth in (
            A.func,
            A().func,
            A.cls_func,
            A().cls_func,
            A.static_func,
            A().static_func
        ):
            with self.subTest(meth=meth):
                self.assertEqual(meth.__qualname__, prefix + meth.__name__)
                self.assertEqual(meth.__doc__,
                                 ('My function docstring'
                                  if support.HAVE_PY_DOCSTRINGS
                                  else None))
                self.assertEqual(meth.__annotations__['arg'], int)

        self.assertEqual(A.func.__name__, 'func')
        self.assertEqual(A().func.__name__, 'func')
        self.assertEqual(A.cls_func.__name__, 'cls_func')
        self.assertEqual(A().cls_func.__name__, 'cls_func')
        self.assertEqual(A.static_func.__name__, 'static_func')
        self.assertEqual(A().static_func.__name__, 'static_func')

    def test_double_wrapped_methods(self):
        # [FUNCTION DOC] Checks singledispatchmethod over class methods that are wrapped by
        # another decorator (a context manager, or a decorator that rewraps the underlying
        # function): behavior matches the same methods without dispatch, and name, docstring and
        # annotations are preserved on class and instance access.
        def classmethod_friendly_decorator(func):
            # [FUNCTION DOC] HELPER: Decorator for class methods; unwraps the underlying function
            # and returns a new class method that forwards to it, keeping its metadata.
            wrapped = func.__func__
            @classmethod
            @functools.wraps(wrapped)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Forwards all arguments to the wrapped function.
                return wrapped(*args, **kwargs)
            return wrapper

        class WithoutSingleDispatch:
            # [CLASS DOC] HELPER: Reference class without dispatch, with a class-method context
            # manager and a class method decorated by the wrapping decorator.
            @classmethod
            @contextlib.contextmanager
            def cls_context_manager(cls, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Context manager yielding the argument as a string; a
                # return in the cleanup block discards exceptions.
                try:
                    yield str(arg)
                finally:
                    return 'Done'

            @classmethod_friendly_decorator
            @classmethod
            def decorated_classmethod(cls, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Returns the argument as a string.
                return str(arg)

        class WithSingleDispatch:
            # [CLASS DOC] HELPER: Same methods as the reference class but wrapped in a
            # singledispatchmethod, with docstrings.
            @functools.singledispatchmethod
            @classmethod
            @contextlib.contextmanager
            def cls_context_manager(cls, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Context manager yielding the argument as a string.
                """My function docstring"""
                try:
                    yield str(arg)
                finally:
                    return 'Done'

            @functools.singledispatchmethod
            @classmethod_friendly_decorator
            @classmethod
            def decorated_classmethod(cls, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Returns the argument as a string.
                """My function docstring"""
                return str(arg)

        # These are sanity checks
        # to test the test itself is working as expected
        with WithoutSingleDispatch.cls_context_manager(5) as foo:
            without_single_dispatch_foo = foo

        with WithSingleDispatch.cls_context_manager(5) as foo:
            single_dispatch_foo = foo

        self.assertEqual(without_single_dispatch_foo, single_dispatch_foo)
        self.assertEqual(single_dispatch_foo, '5')

        self.assertEqual(
            WithoutSingleDispatch.decorated_classmethod(5),
            WithSingleDispatch.decorated_classmethod(5)
        )

        self.assertEqual(WithSingleDispatch.decorated_classmethod(5), '5')

        # Behavioural checks now follow
        for method_name in ('cls_context_manager', 'decorated_classmethod'):
            with self.subTest(method=method_name):
                self.assertEqual(
                    getattr(WithSingleDispatch, method_name).__name__,
                    getattr(WithoutSingleDispatch, method_name).__name__
                )

                self.assertEqual(
                    getattr(WithSingleDispatch(), method_name).__name__,
                    getattr(WithoutSingleDispatch(), method_name).__name__
                )

        for meth in (
            WithSingleDispatch.cls_context_manager,
            WithSingleDispatch().cls_context_manager,
            WithSingleDispatch.decorated_classmethod,
            WithSingleDispatch().decorated_classmethod
        ):
            with self.subTest(meth=meth):
                self.assertEqual(meth.__doc__,
                                 ('My function docstring'
                                  if support.HAVE_PY_DOCSTRINGS
                                  else None))
                self.assertEqual(meth.__annotations__['arg'], int)

        self.assertEqual(
            WithSingleDispatch.cls_context_manager.__name__,
            'cls_context_manager'
        )
        self.assertEqual(
            WithSingleDispatch().cls_context_manager.__name__,
            'cls_context_manager'
        )
        self.assertEqual(
            WithSingleDispatch.decorated_classmethod.__name__,
            'decorated_classmethod'
        )
        self.assertEqual(
            WithSingleDispatch().decorated_classmethod.__name__,
            'decorated_classmethod'
        )

    def test_invalid_registrations(self):
        # [FUNCTION DOC] Checks register raises TypeError with specific messages for a non-class
        # first argument, a function without annotation, a generic collection annotation (typing
        # Iterable), and a Union annotation with a non-class member.
        msg_prefix = "Invalid first argument to `register()`: "
        msg_suffix = (
            ". Use either `@register(some_class)` or plain `@register` on an "
            "annotated function."
        )
        @functools.singledispatch
        def i(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "base".
            return "base"
        with self.assertRaises(TypeError) as exc:
            @i.register(42)
            def _(arg):
                # [FUNCTION DOC] HELPER: Implementation whose registration is rejected
                # (non-type argument).
                return "I annotated with a non-type"
        self.assertTrue(str(exc.exception).startswith(msg_prefix + "42"))
        self.assertTrue(str(exc.exception).endswith(msg_suffix))
        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg):
                # [FUNCTION DOC] HELPER: Unannotated implementation whose registration
                # is rejected.
                return "I forgot to annotate"
        self.assertTrue(str(exc.exception).startswith(msg_prefix +
            "<function TestSingleDispatch.test_invalid_registrations.<locals>._"
        ))
        self.assertTrue(str(exc.exception).endswith(msg_suffix))

        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg: typing.Iterable[str]):
                # [FUNCTION DOC] HELPER: Implementation annotated with a generic; registration
                # is rejected.
                # At runtime, dispatching on generics is impossible.
                # When registering implementations with singledispatch, avoid
                # types from `typing`. Instead, annotate with regular types
                # or ABCs.
                return "I annotated with a generic collection"
        self.assertTrue(str(exc.exception).startswith(
            "Invalid annotation for 'arg'."
        ))
        self.assertTrue(str(exc.exception).endswith(
            'typing.Iterable[str] is not a class.'
        ))

        with self.assertRaises(TypeError) as exc:
            @i.register
            def _(arg: typing.Union[int, typing.Iterable[str]]):
                # [FUNCTION DOC] HELPER: Implementation annotated with an invalid Union;
                # registration is rejected.
                return "Invalid Union"
        self.assertTrue(str(exc.exception).startswith(
            "Invalid annotation for 'arg'."
        ))
        self.assertTrue(str(exc.exception).endswith(
            'typing.Union[int, typing.Iterable[str]] not all arguments are classes.'
        ))

    def test_invalid_positional_argument(self):
        # [FUNCTION DOC] Checks a singledispatch function called without positional arguments
        # (none, or only keywords) raises TypeError saying at least one is required.
        @functools.singledispatch
        def f(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Accepts any arguments and does nothing.
            pass
        msg = 'f requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            f()
        msg = 'f requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            f(a=1)

    def test_invalid_positional_argument_singledispatchmethod(self):
        # [FUNCTION DOC] Checks the same TypeError for a singledispatchmethod called with no
        # positional argument.
        class A:
            # [CLASS DOC] HELPER: Class with a dispatch method accepting any arguments.
            @functools.singledispatchmethod
            def t(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Accepts any arguments and does nothing.
                pass
        msg = 't requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            A().t()
        msg = 't requires at least 1 positional argument'
        with self.assertRaisesRegex(TypeError, msg):
            A().t(a=1)

    def test_union(self):
        # [FUNCTION DOC] Checks annotations using typing Union and the pipe union syntax register
        # every member type, with a default for others.
        @functools.singledispatch
        def f(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        @f.register
        def _(arg: typing.Union[str, bytes]):
            # [FUNCTION DOC] HELPER: Implementation for str and bytes returning "typing.Union".
            return "typing.Union"

        @f.register
        def _(arg: int | float):
            # [FUNCTION DOC] HELPER: Implementation for int and float returning "types.UnionType".
            return "types.UnionType"

        self.assertEqual(f([]), "default")
        self.assertEqual(f(""), "typing.Union")
        self.assertEqual(f(b""), "typing.Union")
        self.assertEqual(f(1), "types.UnionType")
        self.assertEqual(f(1.0), "types.UnionType")

    def test_union_conflict(self):
        # [FUNCTION DOC] Checks that when two union annotations overlap on str, the later
        # registration wins for str, and non-overlapping members keep their implementation.
        @functools.singledispatch
        def f(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        @f.register
        def _(arg: typing.Union[str, bytes]):
            # [FUNCTION DOC] HELPER: Implementation for str and bytes returning "typing.Union".
            return "typing.Union"

        @f.register
        def _(arg: int | str):
            # [FUNCTION DOC] HELPER: Implementation for int and str returning "types.UnionType".
            return "types.UnionType"

        self.assertEqual(f([]), "default")
        self.assertEqual(f(""), "types.UnionType")  # last one wins
        self.assertEqual(f(b""), "typing.Union")
        self.assertEqual(f(1), "types.UnionType")

    def test_union_None(self):
        # [FUNCTION DOC] Checks unions containing None register the NoneType, for both the typing
        # Optional form and the pipe syntax.
        @functools.singledispatch
        def typing_union(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        @typing_union.register
        def _(arg: typing.Union[str, None]):
            # [FUNCTION DOC] HELPER: Implementation for str or None returning "typing.Union".
            return "typing.Union"

        self.assertEqual(typing_union(1), "default")
        self.assertEqual(typing_union(""), "typing.Union")
        self.assertEqual(typing_union(None), "typing.Union")

        @functools.singledispatch
        def types_union(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        @types_union.register
        def _(arg: int | None):
            # [FUNCTION DOC] HELPER: Implementation for int or None returning "types.UnionType".
            return "types.UnionType"

        self.assertEqual(types_union(""), "default")
        self.assertEqual(types_union(1), "types.UnionType")
        self.assertEqual(types_union(None), "types.UnionType")

    def test_register_genericalias(self):
        # [FUNCTION DOC] Checks passing a parametrized generic (builtin or typing, alone or in a
        # union) plus a function to register raises TypeError, and dispatch is unchanged.
        @functools.singledispatch
        def f(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int], lambda arg: "types.GenericAlias")
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[int], lambda arg: "typing.GenericAlias")
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int] | str, lambda arg: "types.UnionTypes(types.GenericAlias)")
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[float] | bytes, lambda arg: "typing.Union[typing.GenericAlias]")

        self.assertEqual(f([1]), "default")
        self.assertEqual(f([1.0]), "default")
        self.assertEqual(f(""), "default")
        self.assertEqual(f(b""), "default")

    def test_register_genericalias_decorator(self):
        # [FUNCTION DOC] Checks passing a parametrized generic alone to register (decorator form)
        # raises TypeError.
        @functools.singledispatch
        def f(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int])
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[int])
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(list[int] | str)
        with self.assertRaisesRegex(TypeError, "Invalid first argument to "):
            f.register(typing.List[int] | str)

    def test_register_genericalias_annotation(self):
        # [FUNCTION DOC] Checks registering functions whose parameter annotation is a parametrized
        # generic (or a union containing one) raises TypeError mentioning the invalid annotation,
        # and dispatch is unchanged.
        @functools.singledispatch
        def f(arg):
            # [FUNCTION DOC] HELPER: Default implementation returning "default".
            return "default"

        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: list[int]):
                # [FUNCTION DOC] HELPER: Implementation rejected due to its generic
                # list annotation.
                return "types.GenericAlias"
        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: typing.List[float]):
                # [FUNCTION DOC] HELPER: Implementation rejected due to its typing
                # List annotation.
                return "typing.GenericAlias"
        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: list[int] | str):
                # [FUNCTION DOC] HELPER: Implementation rejected due to a union with a generic.
                return "types.UnionType(types.GenericAlias)"
        with self.assertRaisesRegex(TypeError, "Invalid annotation for 'arg'"):
            @f.register
            def _(arg: typing.List[float] | bytes):
                # [FUNCTION DOC] HELPER: Implementation rejected due to a typing union with
                # a generic.
                return "typing.Union[typing.GenericAlias]"

        self.assertEqual(f([1]), "default")
        self.assertEqual(f([1.0]), "default")
        self.assertEqual(f(""), "default")
        self.assertEqual(f(b""), "default")

    def test_method_equal_instances(self):
        # [FUNCTION DOC] Regression test (gh-127750): two distinct instances that compare equal
        # and hash the same must each get their own self in the dispatched method.
        # gh-127750: Reference to self was cached
        class A:
            # [CLASS DOC] HELPER: Class whose instances are all equal with hash 1, with a dispatch
            # method returning the instance.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Always returns True.
                return True
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Always returns 1.
                return 1
            @functools.singledispatchmethod
            def t(self, arg):
                # [FUNCTION DOC] HELPER: Default implementation returning self.
                return self

        a = A()
        b = A()
        self.assertIs(a.t(1), a)
        self.assertIs(b.t(2), b)

    def test_method_bad_hash(self):
        # [FUNCTION DOC] Checks that calling, hashing and comparing a dispatch method never calls
        # the instance's equality or hash (which raise if called).
        class A:
            # [CLASS DOC] HELPER: Class whose equality and hash raise AssertionError, with a
            # dispatch method that does nothing.
            def __eq__(self, other):
                # [FUNCTION DOC] HELPER: Raises AssertionError.
                raise AssertionError
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Raises AssertionError.
                raise AssertionError
            @functools.singledispatchmethod
            def t(self, arg):
                # [FUNCTION DOC] HELPER: Default implementation that does nothing.
                pass

        # Should not raise
        A().t(1)
        hash(A().t)
        A().t == A().t

    def test_method_no_reference_loops(self):
        # [FUNCTION DOC] Regression test (gh-127750): a bound dispatch method must not keep a
        # strong reference to the instance; the weak reference dies after deleting the instance,
        # whether or not the bound method outlives it.
        # gh-127750: Created a strong reference to self
        class A:
            # [CLASS DOC] HELPER: Class with a dispatch method returning a weak reference to self.
            @functools.singledispatchmethod
            def t(self, arg):
                # [FUNCTION DOC] HELPER: Default implementation returning a weak reference
                # to self.
                return weakref.ref(self)

        a = A()
        r = a.t(1)
        self.assertIsNotNone(r())
        del a  # delete a after a.t
        if not support.check_impl_detail(cpython=True):
            support.gc_collect()
        self.assertIsNone(r())

        a = A()
        t = a.t
        del a # delete a before a.t
        support.gc_collect()
        r = t(1)
        self.assertIsNotNone(r())
        del t
        if not support.check_impl_detail(cpython=True):
            support.gc_collect()
        self.assertIsNone(r())

    def test_signatures(self):
        # [FUNCTION DOC] Checks the signature of a singledispatch function is that of the base
        # function, not of registered implementations.
        @functools.singledispatch
        def func(item, arg: int) -> str:
            # [FUNCTION DOC] HELPER: Base implementation returning the item as a string.
            return str(item)
        @func.register
        def _(item: int, arg: bytes) -> str:
            # [FUNCTION DOC] HELPER: Implementation for int returning the item as a string.
            return str(item)

        self.assertEqual(str(Signature.from_callable(func)),
                         '(item, arg: int) -> str')

    def test_method_signatures(self):
        # [FUNCTION DOC] Checks signatures of singledispatchmethod objects (class access, instance
        # access, class-method and static variants) match the base function.
        # KNOWN ISSUE: the class-method and static-method variants register implementations with
        # func.register rather than their own cls_func / static_func registrars, so those extra
        # registrations attach to func (likely an upstream copy-paste quirk).
        class A:
            # [CLASS DOC] HELPER: Class with plain, class-method and dispatch-method variants used
            # to compare introspected signatures.
            def m(self, item, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Plain method returning the item as a string.
                return str(item)
            @classmethod
            def cm(cls, item, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Class method returning the item as a string.
                return str(item)
            @functools.singledispatchmethod
            def func(self, item, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Base dispatch method returning the item as a string.
                return str(item)
            @func.register
            def _(self, item, arg: bytes) -> str:
                # [FUNCTION DOC] HELPER: Bytes-annotated variant returning the item as a string.
                return str(item)

            @functools.singledispatchmethod
            @classmethod
            def cls_func(cls, item, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Base class-method dispatch returning the argument as
                # a string.
                return str(arg)
            @func.register
            @classmethod
            def _(cls, item, arg: bytes) -> str:
                # [FUNCTION DOC] HELPER: Class-method variant returning the item as a string.
                return str(item)

            @functools.singledispatchmethod
            @staticmethod
            def static_func(item, arg: int) -> str:
                # [FUNCTION DOC] HELPER: Base static dispatch returning the argument as a string.
                return str(arg)
            @func.register
            @staticmethod
            def _(item, arg: bytes) -> str:
                # [FUNCTION DOC] HELPER: Static variant returning the item as a string.
                return str(item)

        self.assertEqual(str(Signature.from_callable(A.func)),
                         '(self, item, arg: int) -> str')
        self.assertEqual(str(Signature.from_callable(A().func)),
                         '(self, item, arg: int) -> str')
        self.assertEqual(str(Signature.from_callable(A.cls_func)),
                         '(cls, item, arg: int) -> str')
        self.assertEqual(str(Signature.from_callable(A.static_func)),
                         '(item, arg: int) -> str')


class CachedCostItem:
    # [CLASS DOC] HELPER: Fixture with a class-level cost counter, an instance lock and a
    # cached_property cost that increments the counter under the lock on first access.
    _cost = 1

    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates a reentrant lock stored on the instance.
        self.lock = py_functools.RLock()

    @py_functools.cached_property
    def cost(self):
        # [FUNCTION DOC] HELPER: Increments the cost counter under the lock and returns it; its
        # result is cached by the property after the first access.
        """The cost of the item."""
        with self.lock:
            self._cost += 1
        return self._cost


class OptionallyCachedCostItem:
    # [CLASS DOC] HELPER: Fixture exposing the same cost computation as an ordinary method and as
    # a cached_property under a different name.
    _cost = 1

    def get_cost(self):
        # [FUNCTION DOC] HELPER: Increments the cost counter and returns it (not cached itself).
        """The cost of the item."""
        self._cost += 1
        return self._cost

    cached_cost = py_functools.cached_property(get_cost)


class CachedCostItemWithSlots:
    # [CLASS DOC] HELPER: Fixture with slots and so no instance dict; a cached_property on it
    # cannot store its cache.
    __slots__ = ('_cost')

    def __init__(self):
        # [FUNCTION DOC] HELPER: Sets the cost value to 1.
        self._cost = 1

    @py_functools.cached_property
    def cost(self):
        # [FUNCTION DOC] HELPER: Always raises RuntimeError; never reached since caching
        # fails first.
        raise RuntimeError('never called, slots not supported')


class TestCachedProperty(CPythonTestCase):
    # [CLASS DOC] Tests for functools.cached_property using the CachedCostItem fixtures.
    def test_cached(self):
        # [FUNCTION DOC] Checks the property value is computed once and cached.
        item = CachedCostItem()
        self.assertEqual(item.cost, 2)
        self.assertEqual(item.cost, 2) # not 3

    def test_cached_attribute_name_differs_from_func_name(self):
        # [FUNCTION DOC] Checks a cached property under a different name than the function is
        # cached independently of the plain method.
        item = OptionallyCachedCostItem()
        self.assertEqual(item.get_cost(), 2)
        self.assertEqual(item.cached_cost, 3)
        self.assertEqual(item.get_cost(), 4)
        self.assertEqual(item.cached_cost, 3)

    def test_object_with_slots(self):
        # [FUNCTION DOC] Checks that a cached_property on a slotted class raises TypeError saying
        # there is no instance dict to cache in.
        item = CachedCostItemWithSlots()
        with self.assertRaisesRegex(
                TypeError,
                "No '__dict__' attribute on 'CachedCostItemWithSlots' instance to cache 'cost' property.",
        ):
            item.cost

    def test_immutable_dict(self):
        # [FUNCTION DOC] Checks that a cached_property on a metaclass raises TypeError when the
        # class dict does not support item assignment.
        class MyMeta(type):
            # [CLASS DOC] HELPER: Metaclass with a cached_property named prop.
            @py_functools.cached_property
            def prop(self):
                # [FUNCTION DOC] HELPER: Returns True.
                return True

        class MyClass(metaclass=MyMeta):
            # [CLASS DOC] HELPER: Empty class using the metaclass.
            pass

        with self.assertRaisesRegex(
            TypeError,
            "The '__dict__' attribute on 'MyMeta' instance does not support item assignment for caching 'prop' property.",
        ):
            MyClass.prop

    def test_reuse_different_names(self):
        # [FUNCTION DOC] Checks assigning one cached_property to two names in a class body raises
        # TypeError with a message naming both names.
        """Disallow this case because decorated function a would not be cached."""
        with self.assertRaises(TypeError) as ctx:
            class ReusedCachedProperty:
                # [CLASS DOC] HELPER: Class whose body reuses a cached_property under a
                # second name.
                @py_functools.cached_property
                def a(self):
                    # [FUNCTION DOC] HELPER: Does nothing.
                    pass

                b = a

        self.assertEqual(
            str(ctx.exception),
            str(TypeError("Cannot assign the same cached_property to two different names ('a' and 'b')."))
        )

    def test_reuse_same_name(self):
        # [FUNCTION DOC] Checks one cached_property reused under the same name in two classes
        # caches separately for each instance.
        """Reusing a cached_property on different classes under the same name is OK."""
        counter = 0

        @py_functools.cached_property
        def _cp(_self):
            # [FUNCTION DOC] HELPER: Increments the enclosing counter and returns it.
            nonlocal counter
            counter += 1
            return counter

        class A:
            # [CLASS DOC] HELPER: Class using the shared cached property under the name cp.
            cp = _cp

        class B:
            # [CLASS DOC] HELPER: Another class using the shared cached property under the
            # name cp.
            cp = _cp

        a = A()
        b = B()

        self.assertEqual(a.cp, 1)
        self.assertEqual(b.cp, 2)
        self.assertEqual(a.cp, 1)

    def test_set_name_not_called(self):
        # [FUNCTION DOC] Checks using a cached_property attached to a class after creation (no
        # naming hook run) raises TypeError on access.
        cp = py_functools.cached_property(lambda s: None)
        class Foo:
            # [CLASS DOC] HELPER: Empty class that receives the property afterwards.
            pass

        Foo.cp = cp

        with self.assertRaisesRegex(
                TypeError,
                "Cannot use cached_property instance without calling __set_name__ on it.",
        ):
            Foo().cp

    def test_access_from_class(self):
        # [FUNCTION DOC] Checks class-level access returns the cached_property object itself.
        self.assertIsInstance(CachedCostItem.cost, py_functools.cached_property)

    def test_doc(self):
        # [FUNCTION DOC] Checks the property exposes the function's docstring (when kept).
        self.assertEqual(CachedCostItem.cost.__doc__,
                         ("The cost of the item."
                          if support.HAVE_PY_DOCSTRINGS
                          else None))

    def test_module(self):
        # [FUNCTION DOC] Checks the property's module matches the defining class's module.
        self.assertEqual(CachedCostItem.cost.__module__, CachedCostItem.__module__)

    def test_subclass_with___set__(self):
        # [FUNCTION DOC] Checks caching still works for a cached_property subclass that defines a
        # set method raising AttributeError (so it is a data descriptor): later changes to the
        # source attribute are not seen after the first read.
        """Caching still works for a subclass defining __set__."""
        class readonly_cached_property(py_functools.cached_property):
            # [CLASS DOC] HELPER: cached_property subclass whose set method raises AttributeError.
            def __set__(self, obj, value):
                # [FUNCTION DOC] HELPER: Always raises AttributeError (read-only property).
                raise AttributeError("read only property")

        class Test:
            # [CLASS DOC] HELPER: Class holding a value read through the read-only
            # cached property.
            def __init__(self, prop):
                # [FUNCTION DOC] HELPER: Stores the value in a private attribute.
                self._prop = prop

            @readonly_cached_property
            def prop(self):
                # [FUNCTION DOC] HELPER: Returns the stored private value.
                return self._prop

        t = Test(1)
        self.assertEqual(t.prop, 1)
        t._prop = 999
        self.assertEqual(t.prop, 1)


if __name__ == '__main__':
    run_tests()
