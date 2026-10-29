# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa
# [MODULE DOC] Copy of the CPython 3.13 test_builtin test suite adapted for the Dynamo test harness
# (classes derive from the CPython test case base, some tests are wrapped to not error on graph
# breaks or skipped under Dynamo).
# Contents: sequence and error fixtures, then test cases for builtin functions, breakpoint,
# terminal-based input, sorted, shutdown, immortal objects and type creation.
# Import-time side effects: imports the test support helpers from the standard library test
# package, tries to import pty and signal (falls back to none), and computes a double-rounding flag
# and a global value used by tests. The main block runs the test runner.

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_builtin.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    skipIfTorchDynamo,
    run_tests,
)

# ======= END DYNAMO PATCH =======

# Python test set -- built-in functions

import ast
import asyncio
import builtins
import collections
import contextlib
import decimal
import fractions
import gc
import io
import locale
import math
import os
import pickle
import platform
import random
import re
import sys
import traceback
import types
import typing
import unittest
import warnings
from contextlib import ExitStack
from functools import partial
from inspect import CO_COROUTINE
from itertools import product
from textwrap import dedent
from types import AsyncGeneratorType, FunctionType, CellType
from operator import neg
from test import support
from test.support import (cpython_only, swap_attr, maybe_get_event_loop_policy)
from test.support.import_helper import import_module
from test.support.os_helper import (EnvironmentVarGuard, TESTFN, unlink)
from test.support.script_helper import assert_python_ok
from test.support.warnings_helper import check_warnings
from test.support import requires_IEEE_754
from unittest.mock import MagicMock, patch
try:
    import pty, signal
except ImportError:
    pty = signal = None


# Detect evidence of double-rounding: sum() does not always
# get improved accuracy on machines that suffer from double rounding.
x, y = 1e16, 2.9999 # use temporary values to defeat peephole optimizer
HAVE_DOUBLE_ROUNDING = (x + y == 1e16 + 4)

# used as proof of globals being used
A_GLOBAL_VALUE = 123

class Squares:
    # [CLASS DOC] HELPER: Sequence fixture yielding squares of 0..max-1, computed lazily and cached
    # in sofar. Supports indexing and a length that reflects only computed items.

    def __init__(self, max):
        # [FUNCTION DOC] HELPER: Store the upper bound max and start with an empty
        # cache list sofar.
        self.max = max
        self.sofar = []

    # [FUNCTION DOC] HELPER: Return how many squares have been computed so far (not the bound).
    def __len__(self): return len(self.sofar)

    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the i-th square, lazily extending the cache; raises
        # IndexError when i is outside 0..max-1.
        if not 0 <= i < self.max: raise IndexError
        n = len(self.sofar)
        while n <= i:
            self.sofar.append(n*n)
            n += 1
        return self.sofar[i]

class StrSquares:
    # [CLASS DOC] HELPER: Sequence fixture like Squares but its items are the squares converted to
    # strings. Attributes: max (bound), sofar (cache list).

    def __init__(self, max):
        # [FUNCTION DOC] HELPER: Store the bound max and an empty cache list.
        self.max = max
        self.sofar = []

    def __len__(self):
        # [FUNCTION DOC] HELPER: Return the number of items computed so far, not max.
        return len(self.sofar)

    def __getitem__(self, i):
        # [FUNCTION DOC] HELPER: Return the i-th square as a string, lazily extending the cache;
        # raises IndexError when i is outside 0..max-1.
        if not 0 <= i < self.max:
            raise IndexError
        n = len(self.sofar)
        while n <= i:
            self.sofar.append(str(n*n))
            n += 1
        return self.sofar[i]

class BitBucket:
    # [CLASS DOC] HELPER: Write-only sink standing in for stdout so echoed prompts are discarded.
    def write(self, line):
        # [FUNCTION DOC] HELPER: Accept and discard a line of text.
        pass

test_conv_no_sign = [
        ('0', 0),
        ('1', 1),
        ('9', 9),
        ('10', 10),
        ('99', 99),
        ('100', 100),
        ('314', 314),
        (' 314', 314),
        ('314 ', 314),
        ('  \t\t  314  \t\t  ', 314),
        (repr(sys.maxsize), sys.maxsize),
        ('  1x', ValueError),
        ('  1  ', 1),
        ('  1\02  ', ValueError),
        ('', ValueError),
        (' ', ValueError),
        ('  \t\t  ', ValueError),
        (str(br'\u0663\u0661\u0664 ','raw-unicode-escape'), 314),
        (chr(0x200), ValueError),
]

test_conv_sign = [
        ('0', 0),
        ('1', 1),
        ('9', 9),
        ('10', 10),
        ('99', 99),
        ('100', 100),
        ('314', 314),
        (' 314', ValueError),
        ('314 ', 314),
        ('  \t\t  314  \t\t  ', ValueError),
        (repr(sys.maxsize), sys.maxsize),
        ('  1x', ValueError),
        ('  1  ', ValueError),
        ('  1\02  ', ValueError),
        ('', ValueError),
        (' ', ValueError),
        ('  \t\t  ', ValueError),
        (str(br'\u0663\u0661\u0664 ','raw-unicode-escape'), 314),
        (chr(0x200), ValueError),
]

class TestFailingBool:
    # [CLASS DOC] HELPER: Fixture whose truth-value test always raises RuntimeError.
    def __bool__(self):
        # [FUNCTION DOC] HELPER: Always raises RuntimeError.
        raise RuntimeError

class TestFailingIter:
    # [CLASS DOC] HELPER: Fixture whose iteration always raises RuntimeError.
    def __iter__(self):
        # [FUNCTION DOC] HELPER: Always raises RuntimeError.
        raise RuntimeError

def filter_char(arg):
    # [FUNCTION DOC] HELPER: Predicate used with filter: true when the character sorts after the
    # letter d.
    return ord(arg) > ord("d")

def map_char(arg):
    # [FUNCTION DOC] HELPER: Mapping function used with map: return the character whose code point
    # is one greater.
    return chr(ord(arg)+1)

class BuiltinTest(CPythonTestCase):
    # [CLASS DOC] Test case checking the behavior of CPython builtin functions (abs, compile, eval,
    # exec, zip, round, sum, and so on) under the Dynamo CPython test harness. Each test method
    # targets one builtin.
    # Helper to check picklability
    def check_iter_pickle(self, it, seq, proto):
        # [FUNCTION DOC] HELPER: Assert an iterator survives pickling at the given protocol: the
        # round trip keeps the type and yields the expected sequence, and after consuming one item
        # the round trip yields the remainder.
        itorg = it
        d = pickle.dumps(it, proto)
        it = pickle.loads(d)
        self.assertEqual(type(itorg), type(it))
        self.assertEqual(list(it), seq)

        #test the iterator after dropping one from it
        it = pickle.loads(d)
        try:
            next(it)
        except StopIteration:
            return
        d = pickle.dumps(it, proto)
        it = pickle.loads(d)
        self.assertEqual(list(it), seq[1:])

    @torch._dynamo.error_on_graph_break(False)
    def test_import(self):
        # [FUNCTION DOC] Check __import__: valid names, keyword arguments, and errors for missing
        # modules, bad argument types, empty names, duplicate arguments, relative import outside a
        # package (warns), and embedded null characters.
        __import__('sys')
        __import__('time')
        __import__('string')
        __import__(name='sys')
        __import__(name='time', level=0)
        self.assertRaises(ModuleNotFoundError, __import__, 'spamspam')
        self.assertRaises(TypeError, __import__, 1, 2, 3, 4)
        self.assertRaises(ValueError, __import__, '')
        self.assertRaises(TypeError, __import__, 'sys', name='sys')
        # Relative import outside of a package with no __package__ or __spec__ (bpo-37409).
        with self.assertWarns(ImportWarning):
            self.assertRaises(ImportError, __import__, '',
                              {'__package__': None, '__spec__': None, '__name__': '__main__'},
                              locals={}, fromlist=('foo',), level=1)
        # embedded null character
        self.assertRaises(ModuleNotFoundError, __import__, 'string\x00')

    def test_abs(self):
        # [FUNCTION DOC] Check abs on ints, floats, bools, objects defining the abs hook; raises
        # TypeError for strings, None and missing arguments.
        # int
        self.assertEqual(abs(0), 0)
        self.assertEqual(abs(1234), 1234)
        self.assertEqual(abs(-1234), 1234)
        self.assertTrue(abs(-sys.maxsize-1) > 0)
        # float
        self.assertEqual(abs(0.0), 0.0)
        self.assertEqual(abs(3.14), 3.14)
        self.assertEqual(abs(-3.14), 3.14)
        # str
        self.assertRaises(TypeError, abs, 'a')
        # bool
        self.assertEqual(abs(True), 1)
        self.assertEqual(abs(False), 0)
        # other
        self.assertRaises(TypeError, abs)
        self.assertRaises(TypeError, abs, None)
        class AbsClass(object):
            # [CLASS DOC] HELPER: Fixture whose abs hook returns the constant minus five.
            def __abs__(self):
                # [FUNCTION DOC] HELPER: Return the constant minus five.
                return -5
        self.assertEqual(abs(AbsClass()), -5)

    def test_all(self):
        # [FUNCTION DOC] Check all: truth results, TypeError for non-iterables and wrong argument
        # counts, errors from truth tests and iteration propagate, empty input is true, and
        # evaluation short-circuits.
        self.assertEqual(all([2, 4, 6]), True)
        self.assertEqual(all([2, None, 6]), False)
        self.assertRaises(RuntimeError, all, [2, TestFailingBool(), 6])
        self.assertRaises(RuntimeError, all, TestFailingIter())
        self.assertRaises(TypeError, all, 10)               # Non-iterable
        self.assertRaises(TypeError, all)                   # No args
        self.assertRaises(TypeError, all, [2, 4, 6], [])    # Too many args
        self.assertEqual(all([]), True)                     # Empty iterator
        self.assertEqual(all([0, TestFailingBool()]), False)# Short-circuit
        S = [50, 60]
        self.assertEqual(all(x > 42 for x in S), True)
        S = [50, 40, 60]
        self.assertEqual(all(x > 42 for x in S), False)

    def test_any(self):
        # [FUNCTION DOC] Check any: truth results, TypeError for non-iterables and wrong argument
        # counts, errors from truth tests and iteration propagate, empty input is false, and
        # evaluation short-circuits.
        self.assertEqual(any([None, None, None]), False)
        self.assertEqual(any([None, 4, None]), True)
        self.assertRaises(RuntimeError, any, [None, TestFailingBool(), 6])
        self.assertRaises(RuntimeError, any, TestFailingIter())
        self.assertRaises(TypeError, any, 10)               # Non-iterable
        self.assertRaises(TypeError, any)                   # No args
        self.assertRaises(TypeError, any, [2, 4, 6], [])    # Too many args
        self.assertEqual(any([]), False)                    # Empty iterator
        self.assertEqual(any([1, TestFailingBool()]), True) # Short-circuit
        S = [40, 60, 30]
        self.assertEqual(any(x > 42 for x in S), True)
        S = [10, 20, 30]
        self.assertEqual(any(x > 42 for x in S), False)

    def test_ascii(self):
        # [FUNCTION DOC] Check ascii on basic values, self-referencing containers (rendered with an
        # ellipsis marker), and that for strings it matches repr when escaped characters are
        # involved, including surrogates and wide characters.
        self.assertEqual(ascii(''), '\'\'')
        self.assertEqual(ascii(0), '0')
        self.assertEqual(ascii(()), '()')
        self.assertEqual(ascii([]), '[]')
        self.assertEqual(ascii({}), '{}')
        a = []
        a.append(a)
        self.assertEqual(ascii(a), '[[...]]')
        a = {}
        a[0] = a
        self.assertEqual(ascii(a), '{0: {...}}')
        # Advanced checks for unicode strings
        def _check_uni(s):
            # [FUNCTION DOC] HELPER: Assert that ascii of a string equals its repr.
            self.assertEqual(ascii(s), repr(s))
        _check_uni("'")
        _check_uni('"')
        _check_uni('"\'')
        _check_uni('\0')
        _check_uni('\r\n\t .')
        # Unprintable non-ASCII characters
        _check_uni('\x85')
        _check_uni('\u1fff')
        _check_uni('\U00012fff')
        # Lone surrogates
        _check_uni('\ud800')
        _check_uni('\udfff')
        # Issue #9804: surrogates should be joined even for printable
        # wide characters (UCS-2 builds).
        self.assertEqual(ascii('\U0001d121'), "'\\U0001d121'")
        # All together
        s = "'\0\"\n\r\t abcd\x85é\U00012fff\uD800\U0001D121xxx."
        self.assertEqual(ascii(s),
            r"""'\'\x00"\n\r\t abcd\x85\xe9\U00012fff\ud800\U0001d121xxx.'""")

    def test_neg(self):
        # [FUNCTION DOC] Check that negating the minimum machine-size integer gives a value one
        # above the maximum and the operand stays an int.
        x = -sys.maxsize-1
        self.assertTrue(isinstance(x, int))
        self.assertEqual(-x, sys.maxsize+1)

    def test_callable(self):
        # [FUNCTION DOC] Check callable on builtins, strings, lambdas, functions, classes, bound
        # methods and instances; the call hook is looked up on the class, not the instance.
        self.assertTrue(callable(len))
        self.assertFalse(callable("a"))
        self.assertTrue(callable(callable))
        self.assertTrue(callable(lambda x, y: x + y))
        self.assertFalse(callable(__builtins__))
        # [FUNCTION DOC] HELPER: Empty function fixture.
        def f(): pass
        self.assertTrue(callable(f))

        class C1:
            # [CLASS DOC] HELPER: Fixture class with a method but no call hook.
            # [FUNCTION DOC] HELPER: Empty method fixture.
            def meth(self): pass
        self.assertTrue(callable(C1))
        c = C1()
        self.assertTrue(callable(c.meth))
        self.assertFalse(callable(c))

        # __call__ is looked up on the class, not the instance
        c.__call__ = None
        self.assertFalse(callable(c))
        c.__call__ = lambda self: 0
        self.assertFalse(callable(c))
        del c.__call__
        self.assertFalse(callable(c))

        class C2(object):
            # [CLASS DOC] HELPER: Fixture class defining a call hook.
            # [FUNCTION DOC] HELPER: Empty call hook.
            def __call__(self): pass
        c2 = C2()
        self.assertTrue(callable(c2))
        c2.__call__ = None
        self.assertTrue(callable(c2))
        # [CLASS DOC] HELPER: Subclass of C2 used to check inherited call hook.
        class C3(C2): pass
        c3 = C3()
        self.assertTrue(callable(c3))

    def test_chr(self):
        # [FUNCTION DOC] Check chr: known code points, boundaries of the Unicode range, TypeError
        # for non-int or missing arguments, ValueError for out-of-range values.
        self.assertEqual(chr(0), '\0')
        self.assertEqual(chr(32), ' ')
        self.assertEqual(chr(65), 'A')
        self.assertEqual(chr(97), 'a')
        self.assertEqual(chr(0xff), '\xff')
        self.assertRaises(TypeError, chr)
        self.assertRaises(TypeError, chr, 65.0)
        self.assertEqual(chr(0x0000FFFF), "\U0000FFFF")
        self.assertEqual(chr(0x00010000), "\U00010000")
        self.assertEqual(chr(0x00010001), "\U00010001")
        self.assertEqual(chr(0x000FFFFE), "\U000FFFFE")
        self.assertEqual(chr(0x000FFFFF), "\U000FFFFF")
        self.assertEqual(chr(0x00100000), "\U00100000")
        self.assertEqual(chr(0x00100001), "\U00100001")
        self.assertEqual(chr(0x0010FFFE), "\U0010FFFE")
        self.assertEqual(chr(0x0010FFFF), "\U0010FFFF")
        self.assertRaises(ValueError, chr, -1)
        self.assertRaises(ValueError, chr, 0x00110000)
        self.assertRaises(ValueError, chr, 1<<24)
        self.assertRaises(ValueError, chr, 2**32-1)
        self.assertRaises(ValueError, chr, -2**32)
        self.assertRaises(ValueError, chr, 2**1000)
        self.assertRaises(ValueError, chr, -2**1000)

    def test_cmp(self):
        # [FUNCTION DOC] Check that the Python 2 builtin cmp is absent from the builtins module.
        self.assertTrue(not hasattr(builtins, "cmp"))

    @torch._dynamo.error_on_graph_break(False)
    def test_compile(self):
        # [FUNCTION DOC] Check compile: argument forms (bytes with BOM, memoryview, keywords),
        # errors for bad mode, flags, argument combinations and null characters, and the optimize
        # levels for source text and AST input (assert stripping, docstring stripping, debug flag).
        compile('print(1)\n', '', 'exec')
        bom = b'\xef\xbb\xbf'
        compile(bom + b'print(1)\n', '', 'exec')
        compile(source='pass', filename='?', mode='exec')
        compile(dont_inherit=False, filename='tmp', source='0', mode='eval')
        compile('pass', '?', dont_inherit=True, mode='exec')
        compile(memoryview(b"text"), "name", "exec")
        self.assertRaises(TypeError, compile)
        self.assertRaises(ValueError, compile, 'print(42)\n', '<string>', 'badmode')
        self.assertRaises(ValueError, compile, 'print(42)\n', '<string>', 'single', 0xff)
        self.assertRaises(TypeError, compile, 'pass', '?', 'exec',
                          mode='eval', source='0', filename='tmp')
        compile('print("\xe5")\n', '', 'exec')
        self.assertRaises(SyntaxError, compile, chr(0), 'f', 'exec')
        self.assertRaises(ValueError, compile, str('a = 1'), 'f', 'bad')

        # test the optimize argument

        codestr = '''def f():
        """doc"""
        debug_enabled = False
        if __debug__:
            debug_enabled = True
        try:
            assert False
        except AssertionError:
            return (True, f.__doc__, debug_enabled, __debug__)
        else:
            return (False, f.__doc__, debug_enabled, __debug__)
        '''
        # [FUNCTION DOC] HELPER: One-line function fixture with a docstring, used to get the
        # expected docstring value.
        def f(): """doc"""
        values = [(-1, __debug__, f.__doc__, __debug__, __debug__),
                  (0, True, 'doc', True, True),
                  (1, False, 'doc', False, False),
                  (2, False, None, False, False)]
        for optval, *expected in values:
            with self.subTest(optval=optval):
            # test both direct compilation and compilation via AST
                codeobjs = []
                codeobjs.append(compile(codestr, "<test>", "exec", optimize=optval))
                tree = ast.parse(codestr)
                codeobjs.append(compile(tree, "<test>", "exec", optimize=optval))
                for code in codeobjs:
                    ns = {}
                    exec(code, ns)
                    rv = ns['f']()
                    self.assertEqual(rv, tuple(expected))

    @torch._dynamo.error_on_graph_break(False)
    def test_compile_top_level_await_no_coro(self):
        # [FUNCTION DOC] Check that top-level code without await (functions, comprehensions,
        # generator expressions) compiled with the allow-top-level-await flag does not get the
        # coroutine flag, in single and exec modes.
        """Make sure top level non-await codes get the correct coroutine flags"""
        modes = ('single', 'exec')
        code_samples = [
            '''def f():pass\n''',
            '''[x for x in l]''',
            '''{x for x in l}''',
            '''(x for x in l)''',
            '''{x:x for x in l}''',
        ]
        for mode, code_sample in product(modes, code_samples):
            source = dedent(code_sample)
            co = compile(source,
                            '?',
                            mode,
                            flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

            self.assertNotEqual(co.co_flags & CO_COROUTINE, CO_COROUTINE,
                                msg=f"source={source} mode={mode}")


    @unittest.skipIf(
        support.is_emscripten or support.is_wasi,
        "socket.accept is broken"
    )
    @torch._dynamo.error_on_graph_break(False)
    def test_compile_top_level_await(self):
        # [FUNCTION DOC] Check that code using top-level await, async for, async with and async
        # comprehensions fails to compile without the allow-top-level-await flag, compiles with it
        # with the coroutine flag set, and runs both via a function type wrapper and by evaluating
        # the code object under an event loop. Restores the event loop policy afterwards.
        # KNOWN ISSUE: the skip reason says 'socket.accept is broken' but this test uses no
        # sockets; the message looks copied from another test.
        """Test whether code some top level await can be compiled.

        Make sure it compiles only with the PyCF_ALLOW_TOP_LEVEL_AWAIT flag
        set, and make sure the generated code object has the CO_COROUTINE flag
        set in order to execute it with  `await eval(.....)` instead of exec,
        or via a FunctionType.
        """

        # helper function just to check we can run top=level async-for
        async def arange(n):
            # [FUNCTION DOC] HELPER: Async generator yielding integers from zero to n-1.
            for i in range(n):
                yield i

        modes = ('single', 'exec')
        code_samples = [
            '''a = await asyncio.sleep(0, result=1)''',
            '''async for i in arange(1):
                   a = 1''',
            '''async with asyncio.Lock() as l:
                   a = 1''',
            '''a = [x async for x in arange(2)][1]''',
            '''a = 1 in {x async for x in arange(2)}''',
            '''a = {x:1 async for x in arange(1)}[0]''',
            '''a = [x async for x in arange(2) async for x in arange(2)][1]''',
            '''a = [x async for x in (x async for x in arange(5))][1]''',
            '''a, = [1 for x in {x async for x in arange(1)}]''',
            '''a = [await asyncio.sleep(0, x) async for x in arange(2)][1]'''
        ]
        policy = maybe_get_event_loop_policy()
        try:
            for mode, code_sample in product(modes, code_samples):
                source = dedent(code_sample)
                with self.assertRaises(
                        SyntaxError, msg=f"source={source} mode={mode}"):
                    compile(source, '?', mode)

                co = compile(source,
                             '?',
                             mode,
                             flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)

                self.assertEqual(co.co_flags & CO_COROUTINE, CO_COROUTINE,
                                 msg=f"source={source} mode={mode}")

                # test we can create and  advance a function type
                globals_ = {'asyncio': asyncio, 'a': 0, 'arange': arange}
                async_f = FunctionType(co, globals_)
                asyncio.run(async_f())
                self.assertEqual(globals_['a'], 1)

                # test we can await-eval,
                globals_ = {'asyncio': asyncio, 'a': 0, 'arange': arange}
                asyncio.run(eval(co, globals_))
                self.assertEqual(globals_['a'], 1)
        finally:
            asyncio.set_event_loop_policy(policy)

    @torch._dynamo.error_on_graph_break(False)
    def test_compile_top_level_await_invalid_cases(self):
        # [FUNCTION DOC] Check that await and async constructs placed inside ordinary function
        # definitions raise SyntaxError even with the allow-top-level-await flag. Restores the
        # event loop policy afterwards.
         # helper function just to check we can run top=level async-for
        async def arange(n):
            # [FUNCTION DOC] HELPER: Async generator yielding integers from zero to n-1.
            for i in range(n):
                yield i

        modes = ('single', 'exec')
        code_samples = [
            '''def f():  await arange(10)\n''',
            '''def f():  [x async for x in arange(10)]\n''',
            '''def f():  [await x async for x in arange(10)]\n''',
            '''def f():
                   async for i in arange(1):
                       a = 1
            ''',
            '''def f():
                   async with asyncio.Lock() as l:
                       a = 1
            '''
        ]
        policy = maybe_get_event_loop_policy()
        try:
            for mode, code_sample in product(modes, code_samples):
                source = dedent(code_sample)
                with self.assertRaises(
                        SyntaxError, msg=f"source={source} mode={mode}"):
                    compile(source, '?', mode)

                with self.assertRaises(
                        SyntaxError, msg=f"source={source} mode={mode}"):
                    co = compile(source,
                             '?',
                             mode,
                             flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        finally:
            asyncio.set_event_loop_policy(policy)


    @torch._dynamo.error_on_graph_break(False)
    def test_compile_async_generator(self):
        # [FUNCTION DOC] Check that an async generator function compiled with the allow-top-level-
        # await flag is still an async generator and not marked as a coroutine.
        """
        With the PyCF_ALLOW_TOP_LEVEL_AWAIT flag added in 3.8, we want to
        make sure AsyncGenerators are still properly not marked with the
        CO_COROUTINE flag.
        """
        code = dedent("""async def ticker():
                for i in range(10):
                    yield i
                    await asyncio.sleep(0)""")

        co = compile(code, '?', 'exec', flags=ast.PyCF_ALLOW_TOP_LEVEL_AWAIT)
        glob = {}
        exec(co, glob)
        self.assertEqual(type(glob['ticker']()), AsyncGeneratorType)

    @torch._dynamo.error_on_graph_break(False)
    def test_compile_ast(self):
        # [FUNCTION DOC] Check compile with the only-AST and optimized-AST flags: the raw tree
        # keeps the addition node, the optimized trees (from text and from a parsed tree) fold it
        # into a constant.
        args = ("a*(1+2)", "f.py", "exec")
        raw = compile(*args, flags = ast.PyCF_ONLY_AST).body[0]
        opt1 = compile(*args, flags = ast.PyCF_OPTIMIZED_AST).body[0]
        opt2 = compile(ast.parse(args[0]), *args[1:], flags = ast.PyCF_OPTIMIZED_AST).body[0]

        for tree in (raw, opt1, opt2):
            self.assertIsInstance(tree.value, ast.BinOp)
            self.assertIsInstance(tree.value.op, ast.Mult)
            self.assertIsInstance(tree.value.left, ast.Name)
            self.assertEqual(tree.value.left.id, 'a')

        raw_right = raw.value.right  # expect BinOp(1, '+', 2)
        self.assertIsInstance(raw_right, ast.BinOp)
        self.assertIsInstance(raw_right.left, ast.Constant)
        self.assertEqual(raw_right.left.value, 1)
        self.assertIsInstance(raw_right.right, ast.Constant)
        self.assertEqual(raw_right.right.value, 2)

        for opt in [opt1, opt2]:
            opt_right = opt.value.right  # expect Constant(3)
            self.assertIsInstance(opt_right, ast.Constant)
            self.assertEqual(opt_right.value, 3)

    def test_delattr(self):
        # [FUNCTION DOC] Check delattr removes an attribute and raises TypeError for missing
        # arguments and non-string names. Side effect: temporarily sets then deletes an attribute
        # on the sys module.
        sys.spam = 1
        delattr(sys, 'spam')
        self.assertRaises(TypeError, delattr)
        self.assertRaises(TypeError, delattr, sys)
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, delattr, sys, 1)

    def test_dir(self):
        # [FUNCTION DOC] Check dir: argument errors, local scope, modules (including an invalid
        # dict attribute), types, instances with and without a dict or slots, custom dir hooks
        # returning lists, tuples, sets (result sorted list) or non-sequences (TypeError), and
        # traceback objects.
        # dir(wrong number of arguments)
        self.assertRaises(TypeError, dir, 42, 42)

        # dir() - local scope
        local_var = 1
        self.assertIn('local_var', dir())

        # dir(module)
        self.assertIn('exit', dir(sys))

        # dir(module_with_invalid__dict__)
        class Foo(types.ModuleType):
            # [CLASS DOC] HELPER: Module subclass whose dict attribute is an invalid integer, to
            # make dir fail.
            __dict__ = 8
        f = Foo("foo")
        self.assertRaises(TypeError, dir, f)

        # dir(type)
        self.assertIn("strip", dir(str))
        self.assertNotIn("__mro__", dir(str))

        # dir(obj)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture with three instance attributes set in the constructor.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Set attributes x, y and z.
                self.x = 7
                self.y = 8
                self.z = 9
        f = Foo()
        self.assertIn("y", dir(f))

        # dir(obj_no__dict__)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture with empty slots, so instances have no dict.
            __slots__ = []
        f = Foo()
        self.assertIn("__repr__", dir(f))

        # dir(obj_no__class__with__dict__)
        # (an ugly trick to cause getattr(f, "__class__") to fail)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture whose slots hide the class attribute and provide a dict,
            # making class lookup fail.
            __slots__ = ["__class__", "__dict__"]
            def __init__(self):
                # [FUNCTION DOC] HELPER: Set the attribute bar.
                self.bar = "wow"
        f = Foo()
        self.assertNotIn("__repr__", dir(f))
        self.assertIn("bar", dir(f))

        # dir(obj_using __dir__)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture with a dir hook returning an unsorted list.
            def __dir__(self):
                # [FUNCTION DOC] HELPER: Return the list of names.
                return ["kan", "ga", "roo"]
        f = Foo()
        self.assertTrue(dir(f) == ["ga", "kan", "roo"])

        # dir(obj__dir__tuple)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture with a dir hook returning a tuple.
            def __dir__(self):
                # [FUNCTION DOC] HELPER: Return a tuple of names.
                return ("b", "c", "a")
        res = dir(Foo())
        self.assertIsInstance(res, list)
        self.assertTrue(res == ["a", "b", "c"])

        # dir(obj__dir__iterable)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture with a dir hook returning a set.
            def __dir__(self):
                # [FUNCTION DOC] HELPER: Return a set of names.
                return {"b", "c", "a"}
        res = dir(Foo())
        self.assertIsInstance(res, list)
        self.assertEqual(sorted(res), ["a", "b", "c"])

        # dir(obj__dir__not_sequence)
        class Foo(object):
            # [CLASS DOC] HELPER: Fixture with a dir hook returning a non-sequence.
            def __dir__(self):
                # [FUNCTION DOC] HELPER: Return an integer, which dir must reject.
                return 7
        f = Foo()
        self.assertRaises(TypeError, dir, f)

        # dir(traceback)
        try:
            raise IndexError
        except IndexError as e:
            self.assertEqual(len(dir(e.__traceback__)), 4)

        # test that object has a __dir__()
        self.assertEqual(sorted([].__dir__()), dir([]))

    def test___ne__(self):
        # [FUNCTION DOC] Check the not-equal hook of None: false against None, not-implemented
        # against other values.
        self.assertFalse(None.__ne__(None))
        self.assertIs(None.__ne__(0), NotImplemented)
        self.assertIs(None.__ne__("abc"), NotImplemented)

    def test_divmod(self):
        # [FUNCTION DOC] Check divmod for ints with all sign combinations, minimum-int divided by
        # minus one, floats, and TypeError without arguments.
        self.assertEqual(divmod(12, 7), (1, 5))
        self.assertEqual(divmod(-12, 7), (-2, 2))
        self.assertEqual(divmod(12, -7), (-2, -2))
        self.assertEqual(divmod(-12, -7), (1, -5))

        self.assertEqual(divmod(-sys.maxsize-1, -1), (sys.maxsize+1, 0))

        for num, denom, exp_result in [ (3.25, 1.0, (3.0, 0.25)),
                                        (-3.25, 1.0, (-4.0, 0.75)),
                                        (3.25, -1.0, (-4.0, -0.75)),
                                        (-3.25, -1.0, (3.0, -0.25))]:
            result = divmod(num, denom)
            self.assertAlmostEqual(result[0], exp_result[0])
            self.assertAlmostEqual(result[1], exp_result[1])

        self.assertRaises(TypeError, divmod)

    @torch._dynamo.error_on_graph_break(False)
    def test_eval(self):
        # [FUNCTION DOC] Check eval of text and bytes (with BOM) with optional globals and locals
        # mappings, TypeError for bad arguments, SyntaxError for a truncated BOM, and errors from a
        # locals object propagate.
        self.assertEqual(eval('1+1'), 2)
        self.assertEqual(eval(' 1+1\n'), 2)
        globals = {'a': 1, 'b': 2}
        locals = {'b': 200, 'c': 300}
        self.assertEqual(eval('a', globals) , 1)
        self.assertEqual(eval('a', globals, locals), 1)
        self.assertEqual(eval('b', globals, locals), 200)
        self.assertEqual(eval('c', globals, locals), 300)
        globals = {'a': 1, 'b': 2}
        locals = {'b': 200, 'c': 300}
        bom = b'\xef\xbb\xbf'
        self.assertEqual(eval(bom + b'a', globals, locals), 1)
        self.assertEqual(eval('"\xe5"', globals), "\xe5")
        self.assertRaises(TypeError, eval)
        self.assertRaises(TypeError, eval, ())
        self.assertRaises(SyntaxError, eval, bom[:2] + b'a')

        class X:
            # [CLASS DOC] HELPER: Locals fixture whose item lookup raises ValueError.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, eval, "foo", {}, X())

    @torch._dynamo.error_on_graph_break(False)
    def test_eval_kwargs(self):
        # [FUNCTION DOC] Check eval accepts globals and locals as keyword arguments and that the
        # globals mapping is the one visible to the evaluated code.
        data = {"A_GLOBAL_VALUE": 456}
        self.assertEqual(eval("globals()['A_GLOBAL_VALUE']", globals=data), 456)
        self.assertEqual(eval("globals()['A_GLOBAL_VALUE']", locals=data), 123)

    @torch._dynamo.error_on_graph_break(False)
    def test_general_eval(self):
        # [FUNCTION DOC] Check eval with general mapping objects, dict subclasses and a recursive
        # spreadsheet-like mapping as locals; non-mappings raise TypeError, missing names raise
        # NameError, and dir inside eval raises TypeError for non-list keys.
        # Tests that general mappings can be used for the locals argument

        class M:
            # [CLASS DOC] HELPER: Minimal mapping fixture returning twelve for key a, KeyError
            # otherwise, with keys x, y, z.
            "Test mapping interface versus possible calls from eval()."
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return 12 for key a, otherwise raise KeyError.
                if key == 'a':
                    return 12
                raise KeyError
            def keys(self):
                # [FUNCTION DOC] HELPER: Return the list of key names x, y, z.
                return list('xyz')

        m = M()
        g = globals()
        self.assertEqual(eval('a', g, m), 12)
        self.assertRaises(NameError, eval, 'b', g, m)
        self.assertEqual(eval('dir()', g, m), list('xyz'))
        self.assertEqual(eval('globals()', g, m), g)
        self.assertEqual(eval('locals()', g, m), m)
        self.assertRaises(TypeError, eval, 'a', m)
        class A:
            # [CLASS DOC] HELPER: Non-mapping fixture.
            "Non-mapping"
            pass
        m = A()
        self.assertRaises(TypeError, eval, 'a', g, m)

        # Verify that dict subclasses work as well
        class D(dict):
            # [CLASS DOC] HELPER: Dict subclass overriding item lookup for key a and keys.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return 12 for key a, otherwise defer to the dict lookup.
                if key == 'a':
                    return 12
                return dict.__getitem__(self, key)
            def keys(self):
                # [FUNCTION DOC] HELPER: Return the list of key names x, y, z.
                return list('xyz')

        d = D()
        self.assertEqual(eval('a', g, d), 12)
        self.assertRaises(NameError, eval, 'b', g, d)
        self.assertEqual(eval('dir()', g, d), list('xyz'))
        self.assertEqual(eval('globals()', g, d), g)
        self.assertEqual(eval('locals()', g, d), d)

        # Verify locals stores (used by list comps)
        eval('[locals() for i in (2,3)]', g, d)
        eval('[locals() for i in (2,3)]', g, collections.UserDict())

        class SpreadSheet:
            # [CLASS DOC] HELPER: Mapping fixture storing formulas by name in a class-level dict
            # and evaluating them lazily with itself as locals.
            "Sample application showing nested, calculated lookups."
            _cells = {}
            def __setitem__(self, key, formula):
                # [FUNCTION DOC] HELPER: Store a formula under the key.
                self._cells[key] = formula
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Evaluate the stored formula with this object as locals.
                return eval(self._cells[key], globals(), self)

        ss = SpreadSheet()
        ss['a1'] = '5'
        ss['a2'] = 'a1*6'
        ss['a3'] = 'a2*7'
        self.assertEqual(ss['a3'], 210)

        # Verify that dir() catches a non-list returned by eval
        # SF bug #1004669
        class C:
            # [CLASS DOC] HELPER: Fixture whose item lookup raises KeyError and whose keys method
            # returns a non-list.
            def __getitem__(self, item):
                # [FUNCTION DOC] HELPER: Always raises KeyError for the item.
                raise KeyError(item)
            def keys(self):
                # [FUNCTION DOC] HELPER: Return the integer one instead of a key list.
                return 1 # used to be 'a' but that's no longer an error
        self.assertRaises(TypeError, eval, 'dir()', globals(), C())

    @torch._dynamo.error_on_graph_break(False)
    def test_exec(self):
        # [FUNCTION DOC] Check exec with globals only and with globals and locals; global
        # declarations land in globals, other assignments in locals. Removes the injected builtins
        # entry before comparing.
        g = {}
        exec('z = 1', g)
        if '__builtins__' in g:
            del g['__builtins__']
        self.assertEqual(g, {'z': 1})

        exec('z = 1+1', g)
        if '__builtins__' in g:
            del g['__builtins__']
        self.assertEqual(g, {'z': 2})
        g = {}
        l = {}

        with check_warnings():
            warnings.filterwarnings("ignore", "global statement",
                    module="<string>")
            exec('global a; a = 1; b = 2', g, l)
        if '__builtins__' in g:
            del g['__builtins__']
        if '__builtins__' in l:
            del l['__builtins__']
        self.assertEqual((g, l), ({'a': 1}, {'b': 2}))

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_kwargs(self):
        # [FUNCTION DOC] Check exec with keyword arguments: a global assignment reaches the globals
        # mapping, but not when only locals is given.
        g = {}
        exec('global z\nz = 1', globals=g)
        if '__builtins__' in g:
            del g['__builtins__']
        self.assertEqual(g, {'z': 1})

        # if we only set locals, the global assignment will not
        # reach this locals dictionary
        g = {}
        exec('global z\nz = 1', locals=g)
        self.assertEqual(g, {})

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_globals(self):
        # [FUNCTION DOC] Check exec with an empty or invalid builtins entry in globals: NameError
        # for missing print, TypeError when builtins is not a mapping.
        code = compile("print('Hello World!')", "", "exec")
        # no builtin function
        self.assertRaisesRegex(NameError, "name 'print' is not defined",
                               exec, code, {'__builtins__': {}})
        # __builtins__ must be a mapping type
        self.assertRaises(TypeError,
                          exec, code, {'__builtins__': 123})

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_globals_frozen(self):
        # [FUNCTION DOC] Check exec with read-only dict subclasses as builtins and as
        # globals: assignments raise the custom error, class creation needs the build-
        # class helper in builtins.
        class frozendict_error(Exception):
            # [CLASS DOC] HELPER: Exception raised by the read-only dict fixture.
            pass

        class frozendict(dict):
            # [CLASS DOC] HELPER: Dict subclass that refuses item assignment.
            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Always raises frozendict_error.
                raise frozendict_error("frozendict is readonly")

        # read-only builtins
        if isinstance(__builtins__, types.ModuleType):
            frozen_builtins = frozendict(__builtins__.__dict__)
        else:
            frozen_builtins = frozendict(__builtins__)
        code = compile("__builtins__['superglobal']=2; print(superglobal)", "test", "exec")
        self.assertRaises(frozendict_error,
                          exec, code, {'__builtins__': frozen_builtins})

        # no __build_class__ function
        code = compile("class A: pass", "", "exec")
        self.assertRaisesRegex(NameError, "__build_class__ not found",
                               exec, code, {'__builtins__': {}})
        # __build_class__ in a custom __builtins__
        exec(code, {'__builtins__': frozen_builtins})
        self.assertRaisesRegex(NameError, "__build_class__ not found",
                               exec, code, {'__builtins__': frozendict()})

        # read-only globals
        namespace = frozendict({})
        code = compile("x=1", "test", "exec")
        self.assertRaises(frozendict_error,
                          exec, code, namespace)

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_globals_error_on_get(self):
        # [FUNCTION DOC] Check that errors raised from item lookup of custom globals or builtins
        # mappings propagate out of exec.
        # custom `globals` or `builtins` can raise errors on item access
        class setonlyerror(Exception):
            # [CLASS DOC] HELPER: Exception fixture.
            pass

        class setonlydict(dict):
            # [CLASS DOC] HELPER: Dict subclass whose item lookup always raises.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Always raises setonlyerror.
                raise setonlyerror

        # globals' `__getitem__` raises
        code = compile("globalname", "test", "exec")
        self.assertRaises(setonlyerror,
                          exec, code, setonlydict({'globalname': 1}))

        # builtins' `__getitem__` raises
        code = compile("superglobal", "test", "exec")
        self.assertRaises(setonlyerror, exec, code,
                          {'__builtins__': setonlydict({'superglobal': 1})})

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_globals_dict_subclass(self):
        # [FUNCTION DOC] Check exec with a plain dict subclass as builtins: found names work,
        # missing names raise NameError.
        class customdict(dict):  # this one should not do anything fancy
            # [CLASS DOC] HELPER: Plain dict subclass with no overrides.
            pass

        code = compile("superglobal", "test", "exec")
        # works correctly
        exec(code, {'__builtins__': customdict({'superglobal': 1})})
        # custom builtins dict subclass is missing key
        self.assertRaisesRegex(NameError, "name 'superglobal' is not defined",
                               exec, code, {'__builtins__': customdict()})

    @torch._dynamo.error_on_graph_break(False)
    def test_eval_builtins_mapping(self):
        # [FUNCTION DOC] Check eval with a read-only mapping proxy as builtins: found names work,
        # missing names raise NameError.
        code = compile("superglobal", "test", "eval")
        # works correctly
        ns = {'__builtins__': types.MappingProxyType({'superglobal': 1})}
        self.assertEqual(eval(code, ns), 1)
        # custom builtins mapping is missing key
        ns = {'__builtins__': types.MappingProxyType({})}
        self.assertRaisesRegex(NameError, "name 'superglobal' is not defined",
                               eval, code, ns)

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_builtins_mapping_import(self):
        # [FUNCTION DOC] Check that an import statement under exec with a mapping-proxy builtins
        # fails with ImportError when the import hook is missing, and calls a custom hook with the
        # expected arguments otherwise.
        code = compile("import foo.bar", "test", "exec")
        ns = {'__builtins__': types.MappingProxyType({})}
        self.assertRaisesRegex(ImportError, "__import__ not found", exec, code, ns)
        ns = {'__builtins__': types.MappingProxyType({'__import__': lambda *args: args})}
        exec(code, ns)
        self.assertEqual(ns['foo'], ('foo.bar', ns, ns, None, 0))

    @torch._dynamo.error_on_graph_break(False)
    def test_eval_builtins_mapping_reduce(self):
        # [FUNCTION DOC] Check that list iterator reduce looks up the iter builtin in a mapping-
        # proxy builtins: AttributeError when missing, expected tuple when present.
        # list_iterator.__reduce__() calls _PyEval_GetBuiltin("iter")
        code = compile("x.__reduce__()", "test", "eval")
        ns = {'__builtins__': types.MappingProxyType({}), 'x': iter([1, 2])}
        self.assertRaisesRegex(AttributeError, "iter", eval, code, ns)
        ns = {'__builtins__': types.MappingProxyType({'iter': iter}), 'x': iter([1, 2])}
        self.assertEqual(eval(code, ns), (iter, ([1, 2],), 0))

    @torch._dynamo.error_on_graph_break(False)
    def test_exec_redirected(self):
        # [FUNCTION DOC] Check that exec of a failing statement with an unflushable stdout (None)
        # still raises NameError and not SystemError. Side effect: temporarily replaces sys.stdout,
        # restored in a finally block.
        savestdout = sys.stdout
        sys.stdout = None # Whatever that cannot flush()
        try:
            # Used to raise SystemError('error return without exception set')
            exec('a')
        except NameError:
            pass
        finally:
            sys.stdout = savestdout

    def test_exec_closure(self):
        # [FUNCTION DOC] Check exec with the closure argument: works for code with free variables
        # given a tuple of cells (including manually created cells), and raises TypeError for code
        # without free variables, missing closure, wrong length, list instead of tuple, closure
        # with string source, or non-cell entries.
        def function_without_closures():
            # [FUNCTION DOC] HELPER: Function without free variables returning a constant.
            return 3 * 5

        result = 0
        def make_closure_functions():
            # [FUNCTION DOC] HELPER: Build two nested functions sharing cell variables a, b and c,
            # return both.
            a = 2
            b = 3
            c = 5
            def three_freevars():
                # [FUNCTION DOC] HELPER: Free-variable function setting the outer
                # result to a times b.
                nonlocal result
                nonlocal a
                nonlocal b
                result = a*b
            def four_freevars():
                # [FUNCTION DOC] HELPER: Free-variable function setting the outer result to a times
                # b times c.
                nonlocal result
                nonlocal a
                nonlocal b
                nonlocal c
                result = a*b*c
            return three_freevars, four_freevars
        three_freevars, four_freevars = make_closure_functions()

        # "smoke" test
        result = 0
        exec(three_freevars.__code__,
            three_freevars.__globals__,
            closure=three_freevars.__closure__)
        self.assertEqual(result, 6)

        # should also work with a manually created closure
        result = 0
        my_closure = (CellType(35), CellType(72), three_freevars.__closure__[2])
        exec(three_freevars.__code__,
            three_freevars.__globals__,
            closure=my_closure)
        self.assertEqual(result, 2520)

        # should fail: closure isn't allowed
        # for functions without free vars
        self.assertRaises(TypeError,
            exec,
            function_without_closures.__code__,
            function_without_closures.__globals__,
            closure=my_closure)

        # should fail: closure required but wasn't specified
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=None)

        # should fail: closure of wrong length
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=four_freevars.__closure__)

        # should fail: closure using a list instead of a tuple
        my_closure = list(my_closure)
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=my_closure)
        my_closure = tuple(my_closure)

        # should fail: anything passed to closure= isn't allowed
        # when the source is a string
        self.assertRaises(TypeError,
            exec,
            "pass",
            closure=int)

        # should fail: correct closure= argument isn't allowed
        # when the source is a string
        self.assertRaises(TypeError,
            exec,
            "pass",
            closure=my_closure)

        # should fail: closure tuple with one non-cell-var
        my_closure = list(my_closure)
        my_closure[0] = int
        my_closure = tuple(my_closure)
        self.assertRaises(TypeError,
            exec,
            three_freevars.__code__,
            three_freevars.__globals__,
            closure=my_closure)


    def test_filter(self):
        # [FUNCTION DOC] Check filter with predicates, None as predicate, custom sequences, tuples,
        # and errors: TypeError for missing arguments, uncallable predicate or wrong-arity
        # function, errors from the sequence propagate.
        self.assertEqual(list(filter(lambda c: 'a' <= c <= 'z', 'Hello World')), list('elloorld'))
        self.assertEqual(list(filter(None, [1, 'hello', [], [3], '', None, 9, 0])), [1, 'hello', [3], 9])
        self.assertEqual(list(filter(lambda x: x > 0, [1, -3, 9, 0, 2])), [1, 9, 2])
        self.assertEqual(list(filter(None, Squares(10))), [1, 4, 9, 16, 25, 36, 49, 64, 81])
        self.assertEqual(list(filter(lambda x: x%2, Squares(10))), [1, 9, 25, 49, 81])
        def identity(item):
            # [FUNCTION DOC] HELPER: Predicate that always returns one.
            return 1
        filter(identity, Squares(5))
        self.assertRaises(TypeError, filter)
        class BadSeq(object):
            # [CLASS DOC] HELPER: Sequence fixture returning 42 for the first four indexes then
            # raising ValueError.
            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Return 42 for index below four, otherwise
                # raise ValueError.
                if index<4:
                    return 42
                raise ValueError
        self.assertRaises(ValueError, list, filter(lambda x: x, BadSeq()))
        def badfunc():
            # [FUNCTION DOC] HELPER: Zero-argument function used as an invalid predicate.
            pass
        self.assertRaises(TypeError, list, filter(badfunc, range(5)))

        # test bltinmodule.c::filtertuple()
        self.assertEqual(list(filter(None, (1, 2))), [1, 2])
        self.assertEqual(list(filter(lambda x: x>=3, (1, 2, 3, 4))), [3, 4])
        self.assertRaises(TypeError, list, filter(42, (1, 2)))

    def test_filter_pickle(self):
        # [FUNCTION DOC] Check filter objects pickle round trip at every pickle protocol using the
        # shared pickle helper.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            f1 = filter(filter_char, "abcdeabcde")
            f2 = filter(filter_char, "abcdeabcde")
            self.check_iter_pickle(f1, list(f2), proto)

    @skipIfTorchDynamo("gc.collect")
    @support.requires_resource('cpu')
    def test_filter_dealloc(self):
        # [FUNCTION DOC] Check that deeply nested filter objects (one million levels) can be
        # deallocated without recursion problems, then force a collection. Skipped under Dynamo and
        # when the cpu resource is not enabled.
        # Tests recursive deallocation of nested filter objects using the
        # thrashcan mechanism. See gh-102356 for more details.
        max_iters = 1000000
        i = filter(bool, range(max_iters))
        for _ in range(max_iters):
            i = filter(bool, i)
        del i
        gc.collect()

    def test_getattr(self):
        # [FUNCTION DOC] Check getattr: returns attributes, TypeError for missing arguments or non-
        # string names (with or without default), AttributeError for missing or unencodable names.
        self.assertTrue(getattr(sys, 'stdout') is sys.stdout)
        self.assertRaises(TypeError, getattr)
        self.assertRaises(TypeError, getattr, sys)
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, getattr, sys, 1)
        self.assertRaisesRegex(TypeError, msg, getattr, sys, 1, 'spam')
        self.assertRaises(AttributeError, getattr, sys, chr(sys.maxunicode))
        # unicode surrogates are not encodable to the default encoding (utf8)
        self.assertRaises(AttributeError, getattr, 1, "\uDAD1\uD51E")

    def test_hasattr(self):
        # [FUNCTION DOC] Check hasattr: true for existing attributes, false for missing ones,
        # TypeError for bad arguments, and exceptions other than AttributeError raised by the
        # attribute hook propagate.
        self.assertTrue(hasattr(sys, 'stdout'))
        self.assertRaises(TypeError, hasattr)
        self.assertRaises(TypeError, hasattr, sys)
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, hasattr, sys, 1)
        self.assertEqual(False, hasattr(sys, chr(sys.maxunicode)))

        # Check that hasattr propagates all exceptions outside of
        # AttributeError.
        class A:
            # [CLASS DOC] HELPER: Fixture whose attribute hook raises SystemExit.
            def __getattr__(self, what):
                # [FUNCTION DOC] HELPER: Always raises SystemExit.
                raise SystemExit
        self.assertRaises(SystemExit, hasattr, A(), "b")
        class B:
            # [CLASS DOC] HELPER: Fixture whose attribute hook raises ValueError.
            def __getattr__(self, what):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, hasattr, B(), "b")

    def test_hash(self):
        # [FUNCTION DOC] Check hash: equal numbers hash equally, strings and bytes agree,
        # unhashable lists and dicts raise TypeError, and a hash hook returning a huge int is
        # reduced to an int.
        hash(None)
        self.assertEqual(hash(1), hash(1))
        self.assertEqual(hash(1), hash(1.0))
        hash('spam')
        self.assertEqual(hash('spam'), hash(b'spam'))
        hash((0,1,2,3))
        # [FUNCTION DOC] HELPER: Empty function used as a hashable object.
        def f(): pass
        hash(f)
        self.assertRaises(TypeError, hash, [])
        self.assertRaises(TypeError, hash, {})
        # Bug 1536021: Allow hash to return long objects
        class X:
            # [CLASS DOC] HELPER: Fixture whose hash hook returns a huge integer.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Return two to the power one hundred.
                return 2**100
        self.assertEqual(type(hash(X())), int)
        class Z(int):
            # [CLASS DOC] HELPER: Int subclass whose hash hook returns itself.
            def __hash__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself as hash.
                return self
        self.assertEqual(hash(Z(42)), hash(42))

    def test_hex(self):
        # [FUNCTION DOC] Check hex output for positive and negative ints and TypeError
        # for non-integers.
        self.assertEqual(hex(16), '0x10')
        self.assertEqual(hex(-16), '-0x10')
        self.assertRaises(TypeError, hex, {})

    def test_id(self):
        # [FUNCTION DOC] Check id accepts a variety of objects without error.
        id(None)
        id(1)
        id(1.0)
        id('spam')
        id((0,1,2,3))
        id([0,1,2,3])
        id({'spam': 1, 'eggs': 2, 'ham': 3})

    # Test input() later, alphabetized as if it were raw_input

    def test_iter(self):
        # [FUNCTION DOC] Check iter: TypeError for bad arguments, and iteration over tuple, list
        # and string yields items then raises StopIteration.
        self.assertRaises(TypeError, iter)
        self.assertRaises(TypeError, iter, 42, 42)
        lists = [("1", "2"), ["1", "2"], "12"]
        for l in lists:
            i = iter(l)
            self.assertEqual(next(i), '1')
            self.assertEqual(next(i), '2')
            self.assertRaises(StopIteration, next, i)

    def test_isinstance(self):
        # [FUNCTION DOC] Check isinstance with a small class hierarchy and TypeError for invalid
        # class argument or missing arguments.
        class C:
            # [CLASS DOC] HELPER: Base class fixture.
            pass
        class D(C):
            # [CLASS DOC] HELPER: Subclass of C.
            pass
        class E:
            # [CLASS DOC] HELPER: Unrelated class fixture.
            pass
        c = C()
        d = D()
        e = E()
        self.assertTrue(isinstance(c, C))
        self.assertTrue(isinstance(d, C))
        self.assertTrue(not isinstance(e, C))
        self.assertTrue(not isinstance(c, D))
        self.assertTrue(not isinstance('foo', E))
        self.assertRaises(TypeError, isinstance, E, 'foo')
        self.assertRaises(TypeError, isinstance)

    def test_issubclass(self):
        # [FUNCTION DOC] Check issubclass with a small class hierarchy and TypeError for non-class
        # arguments or missing arguments.
        class C:
            # [CLASS DOC] HELPER: Base class fixture.
            pass
        class D(C):
            # [CLASS DOC] HELPER: Subclass of C.
            pass
        class E:
            # [CLASS DOC] HELPER: Unrelated class fixture.
            pass
        c = C()
        d = D()
        e = E()
        self.assertTrue(issubclass(D, C))
        self.assertTrue(issubclass(C, C))
        self.assertTrue(not issubclass(C, D))
        self.assertRaises(TypeError, issubclass, 'foo', E)
        self.assertRaises(TypeError, issubclass, E, 'foo')
        self.assertRaises(TypeError, issubclass)

    def test_len(self):
        # [FUNCTION DOC] Check len on builtins and on length hooks returning errors, None or float
        # (TypeError), negative (ValueError), and too-large values (OverflowError), and objects
        # without a length hook (TypeError).
        self.assertEqual(len('123'), 3)
        self.assertEqual(len(()), 0)
        self.assertEqual(len((1, 2, 3, 4)), 4)
        self.assertEqual(len([1, 2, 3, 4]), 4)
        self.assertEqual(len({}), 0)
        self.assertEqual(len({'a':1, 'b': 2}), 2)
        class BadSeq:
            # [CLASS DOC] HELPER: Fixture whose length hook raises ValueError.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, len, BadSeq())
        class InvalidLen:
            # [CLASS DOC] HELPER: Fixture whose length hook returns None.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return None.
                return None
        self.assertRaises(TypeError, len, InvalidLen())
        class FloatLen:
            # [CLASS DOC] HELPER: Fixture whose length hook returns a float.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return a float.
                return 4.5
        self.assertRaises(TypeError, len, FloatLen())
        class NegativeLen:
            # [CLASS DOC] HELPER: Fixture whose length hook returns a negative number.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return minus ten.
                return -10
        self.assertRaises(ValueError, len, NegativeLen())
        class HugeLen:
            # [CLASS DOC] HELPER: Fixture whose length hook returns a value above the
            # machine maximum.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the maximum plus one.
                return sys.maxsize + 1
        self.assertRaises(OverflowError, len, HugeLen())
        class HugeNegativeLen:
            # [CLASS DOC] HELPER: Fixture whose length hook returns a hugely negative number.
            def __len__(self):
                # [FUNCTION DOC] HELPER: Return a hugely negative integer.
                return -sys.maxsize-10
        self.assertRaises(ValueError, len, HugeNegativeLen())
        # [CLASS DOC] HELPER: Fixture with no length hook.
        class NoLenMethod(object): pass
        self.assertRaises(TypeError, len, NoLenMethod())

    def test_map(self):
        # [FUNCTION DOC] Check map with one or several iterables, functions of various arity,
        # uneven lengths, and errors: TypeError for bad arguments, errors from iteration or from
        # the function propagate.
        self.assertEqual(
            list(map(lambda x: x*x, range(1,4))),
            [1, 4, 9]
        )
        try:
            from math import sqrt
        except ImportError:
            def sqrt(x):
                # [FUNCTION DOC] HELPER: Fallback square root via exponentiation, defined only if
                # importing the math version fails.
                return pow(x, 0.5)
        self.assertEqual(
            list(map(lambda x: list(map(sqrt, x)), [[16, 4], [81, 9]])),
            [[4.0, 2.0], [9.0, 3.0]]
        )
        self.assertEqual(
            list(map(lambda x, y: x+y, [1,3,2], [9,1,4])),
            [10, 4, 6]
        )

        def plus(*v):
            # [FUNCTION DOC] HELPER: Variadic function returning the sum of its arguments.
            accu = 0
            for i in v: accu = accu + i
            return accu
        self.assertEqual(
            list(map(plus, [1, 3, 7])),
            [1, 3, 7]
        )
        self.assertEqual(
            list(map(plus, [1, 3, 7], [4, 9, 2])),
            [1+4, 3+9, 7+2]
        )
        self.assertEqual(
            list(map(plus, [1, 3, 7], [4, 9, 2], [1, 1, 0])),
            [1+4+1, 3+9+1, 7+2+0]
        )
        self.assertEqual(
            list(map(int, Squares(10))),
            [0, 1, 4, 9, 16, 25, 36, 49, 64, 81]
        )
        def Max(a, b):
            # [FUNCTION DOC] HELPER: Return the larger of two values, treating None as missing.
            if a is None:
                return b
            if b is None:
                return a
            return max(a, b)
        self.assertEqual(
            list(map(Max, Squares(3), Squares(2))),
            [0, 1]
        )
        self.assertRaises(TypeError, map)
        self.assertRaises(TypeError, map, lambda x: x, 42)
        class BadSeq:
            # [CLASS DOC] HELPER: Fixture whose iteration hook raises ValueError.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Always raises ValueError (the following yield is
                # unreachable and only makes it a generator).
                raise ValueError
                yield None
        self.assertRaises(ValueError, list, map(lambda x: x, BadSeq()))
        def badfunc(x):
            # [FUNCTION DOC] HELPER: Function that always raises RuntimeError.
            raise RuntimeError
        self.assertRaises(RuntimeError, list, map(badfunc, range(5)))

    def test_map_pickle(self):
        # [FUNCTION DOC] Check map objects pickle round trip at every pickle protocol using the
        # shared pickle helper.
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            m1 = map(map_char, "Is this the real life?")
            m2 = map(map_char, "Is this the real life?")
            self.check_iter_pickle(m1, list(m2), proto)

    def test_max(self):
        # [FUNCTION DOC] Check max: iterable and multi-argument forms, mixed numeric
        # types, default and key keywords, errors for no arguments, empty iterables,
        # non-iterables, invalid keyword combinations and uncallable key; compares
        # against a sorted reference on random data.
        self.assertEqual(max('123123'), '3')
        self.assertEqual(max(1, 2, 3), 3)
        self.assertEqual(max((1, 2, 3, 1, 2, 3)), 3)
        self.assertEqual(max([1, 2, 3, 1, 2, 3]), 3)

        self.assertEqual(max(1, 2, 3.0), 3.0)
        self.assertEqual(max(1, 2.0, 3), 3)
        self.assertEqual(max(1.0, 2, 3), 3)

        with self.assertRaisesRegex(
            TypeError,
            'max expected at least 1 argument, got 0'
        ):
            max()

        self.assertRaises(TypeError, max, 42)
        with self.assertRaisesRegex(
            ValueError,
            r'max\(\) iterable argument is empty'
        ):
            max(())
        class BadSeq:
            # [CLASS DOC] HELPER: Fixture whose item hook raises ValueError.
            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, max, BadSeq())

        for stmt in (
            "max(key=int)",                 # no args
            "max(default=None)",
            "max(1, 2, default=None)",      # require container for default
            "max(default=None, key=int)",
            "max(1, key=int)",              # single arg not iterable
            "max(1, 2, keystone=int)",      # wrong keyword
            "max(1, 2, key=int, abc=int)",  # two many keywords
            "max(1, 2, key=1)",             # keyfunc is not callable
            ):
            try:
                exec(stmt, globals())
            except TypeError:
                pass
            else:
                self.fail(stmt)

        self.assertEqual(max((1,), key=neg), 1)     # one elem iterable
        self.assertEqual(max((1,2), key=neg), 1)    # two elem iterable
        self.assertEqual(max(1, 2, key=neg), 1)     # two elems

        self.assertEqual(max((), default=None), None)    # zero elem iterable
        self.assertEqual(max((1,), default=None), 1)     # one elem iterable
        self.assertEqual(max((1,2), default=None), 2)    # two elem iterable

        self.assertEqual(max((), default=1, key=neg), 1)
        self.assertEqual(max((1, 2), default=3, key=neg), 1)

        self.assertEqual(max((1, 2), key=None), 2)

        data = [random.randrange(200) for i in range(100)]
        keys = dict((elem, random.randrange(50)) for elem in data)
        f = keys.__getitem__
        self.assertEqual(max(data, key=f),
                         sorted(reversed(data), key=f)[-1])

    def test_min(self):
        # [FUNCTION DOC] Check min: iterable and multi-argument forms, mixed numeric
        # types, default and key keywords, errors for no arguments, empty iterables,
        # non-iterables, invalid keyword combinations and uncallable key; compares
        # against a sorted reference on random data.
        self.assertEqual(min('123123'), '1')
        self.assertEqual(min(1, 2, 3), 1)
        self.assertEqual(min((1, 2, 3, 1, 2, 3)), 1)
        self.assertEqual(min([1, 2, 3, 1, 2, 3]), 1)

        self.assertEqual(min(1, 2, 3.0), 1)
        self.assertEqual(min(1, 2.0, 3), 1)
        self.assertEqual(min(1.0, 2, 3), 1.0)

        with self.assertRaisesRegex(
            TypeError,
            'min expected at least 1 argument, got 0'
        ):
            min()

        self.assertRaises(TypeError, min, 42)
        with self.assertRaisesRegex(
            ValueError,
            r'min\(\) iterable argument is empty'
        ):
            min(())
        class BadSeq:
            # [CLASS DOC] HELPER: Fixture whose item hook raises ValueError.
            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, min, BadSeq())

        for stmt in (
            "min(key=int)",                 # no args
            "min(default=None)",
            "min(1, 2, default=None)",      # require container for default
            "min(default=None, key=int)",
            "min(1, key=int)",              # single arg not iterable
            "min(1, 2, keystone=int)",      # wrong keyword
            "min(1, 2, key=int, abc=int)",  # two many keywords
            "min(1, 2, key=1)",             # keyfunc is not callable
            ):
            try:
                exec(stmt, globals())
            except TypeError:
                pass
            else:
                self.fail(stmt)

        self.assertEqual(min((1,), key=neg), 1)     # one elem iterable
        self.assertEqual(min((1,2), key=neg), 2)    # two elem iterable
        self.assertEqual(min(1, 2, key=neg), 2)     # two elems

        self.assertEqual(min((), default=None), None)    # zero elem iterable
        self.assertEqual(min((1,), default=None), 1)     # one elem iterable
        self.assertEqual(min((1,2), default=None), 1)    # two elem iterable

        self.assertEqual(min((), default=1, key=neg), 1)
        self.assertEqual(min((1, 2), default=1, key=neg), 2)

        self.assertEqual(min((1, 2), key=None), 1)

        data = [random.randrange(200) for i in range(100)]
        keys = dict((elem, random.randrange(50)) for elem in data)
        f = keys.__getitem__
        self.assertEqual(min(data, key=f),
                         sorted(data, key=f)[0])

    def test_next(self):
        # [FUNCTION DOC] Check next on iterators, with and without a default, for custom iterators
        # and generators; StopIteration is raised repeatedly once exhausted, and the default is
        # returned instead when given.
        it = iter(range(2))
        self.assertEqual(next(it), 0)
        self.assertEqual(next(it), 1)
        self.assertRaises(StopIteration, next, it)
        self.assertRaises(StopIteration, next, it)
        self.assertEqual(next(it, 42), 42)

        class Iter(object):
            # [CLASS DOC] HELPER: Iterator fixture that is exhausted immediately.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Always raises StopIteration.
                raise StopIteration

        it = iter(Iter())
        self.assertEqual(next(it, 42), 42)
        self.assertRaises(StopIteration, next, it)

        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding one item then returning.
            yield 1
            return

        it = gen()
        self.assertEqual(next(it), 1)
        self.assertRaises(StopIteration, next, it)
        self.assertEqual(next(it, 42), 42)

    def test_oct(self):
        # [FUNCTION DOC] Check oct output for positive and negative ints and TypeError
        # for non-integers.
        self.assertEqual(oct(100), '0o144')
        self.assertEqual(oct(-100), '-0o144')
        self.assertRaises(TypeError, oct, ())

    def write_testfile(self):
        # [FUNCTION DOC] HELPER: Write the shared sample file named by the test file constant
        # (arithmetic line, sentence lines, long runs of X and Y) and register its removal as a
        # cleanup. Side effect: creates a file on disk.
        # NB the first 4 lines are also used to test input, below
        fp = open(TESTFN, 'w', encoding="utf-8")
        self.addCleanup(unlink, TESTFN)
        with fp:
            fp.write('1+1\n')
            fp.write('The quick brown fox jumps over the lazy dog')
            fp.write('.\n')
            fp.write('Dear John\n')
            fp.write('XXX'*100)
            fp.write('YYY'*100)

    @torch._dynamo.error_on_graph_break(False)
    def test_open(self):
        # [FUNCTION DOC] Check open: reading by line length and by size from the sample file, and
        # ValueError for names containing null characters (text or bytes). Side effect: creates and
        # removes the sample file.
        self.write_testfile()
        fp = open(TESTFN, encoding="utf-8")
        with fp:
            self.assertEqual(fp.readline(4), '1+1\n')
            self.assertEqual(fp.readline(), 'The quick brown fox jumps over the lazy dog.\n')
            self.assertEqual(fp.readline(4), 'Dear')
            self.assertEqual(fp.readline(100), ' John\n')
            self.assertEqual(fp.read(300), 'XXX'*100)
            self.assertEqual(fp.read(1000), 'YYY'*100)

        # embedded null bytes and characters
        self.assertRaises(ValueError, open, 'a\x00b')
        self.assertRaises(ValueError, open, b'a\x00b')

    @unittest.skipIf(sys.flags.utf8_mode, "utf-8 mode is enabled")
    @torch._dynamo.error_on_graph_break(False)
    def test_open_default_encoding(self):
        # [FUNCTION DOC] Check that open with no encoding uses the locale encoding,
        # even after removing locale environment variables. Side effect: temporarily
        # modifies the process environment (restored in a finally block) and writes the
        # sample file. Skipped in UTF-8 mode.
        old_environ = dict(os.environ)
        try:
            # try to get a user preferred encoding different than the current
            # locale encoding to check that open() uses the current locale
            # encoding and not the user preferred encoding
            for key in ('LC_ALL', 'LANG', 'LC_CTYPE'):
                if key in os.environ:
                    del os.environ[key]

            self.write_testfile()
            current_locale_encoding = locale.getencoding()
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", EncodingWarning)
                fp = open(TESTFN, 'w')
            with fp:
                self.assertEqual(fp.encoding, current_locale_encoding)
        finally:
            os.environ.clear()
            os.environ.update(old_environ)

    @support.requires_subprocess()
    @torch._dynamo.error_on_graph_break(False)
    def test_open_non_inheritable(self):
        # [FUNCTION DOC] Check that a file opened by open is not inheritable by child processes.
        # Skipped when subprocesses are unavailable.
        fileobj = open(__file__, encoding="utf-8")
        with fileobj:
            self.assertFalse(os.get_inheritable(fileobj.fileno()))

    def test_ord(self):
        # [FUNCTION DOC] Check ord for str and bytes of length one, boundaries of the Unicode
        # range, and TypeError for non-strings.
        self.assertEqual(ord(' '), 32)
        self.assertEqual(ord('A'), 65)
        self.assertEqual(ord('a'), 97)
        self.assertEqual(ord('\x80'), 128)
        self.assertEqual(ord('\xff'), 255)

        self.assertEqual(ord(b' '), 32)
        self.assertEqual(ord(b'A'), 65)
        self.assertEqual(ord(b'a'), 97)
        self.assertEqual(ord(b'\x80'), 128)
        self.assertEqual(ord(b'\xff'), 255)

        self.assertEqual(ord(chr(sys.maxunicode)), sys.maxunicode)
        self.assertRaises(TypeError, ord, 42)

        self.assertEqual(ord(chr(0x10FFFF)), 0x10FFFF)
        self.assertEqual(ord("\U0000FFFF"), 0x0000FFFF)
        self.assertEqual(ord("\U00010000"), 0x00010000)
        self.assertEqual(ord("\U00010001"), 0x00010001)
        self.assertEqual(ord("\U000FFFFE"), 0x000FFFFE)
        self.assertEqual(ord("\U000FFFFF"), 0x000FFFFF)
        self.assertEqual(ord("\U00100000"), 0x00100000)
        self.assertEqual(ord("\U00100001"), 0x00100001)
        self.assertEqual(ord("\U0010FFFE"), 0x0010FFFE)
        self.assertEqual(ord("\U0010FFFF"), 0x0010FFFF)

    def test_pow(self):
        # [FUNCTION DOC] Check pow: int and float cases, sign handling, three-argument modular form
        # (TypeError for floats, ValueError for zero modulus), complex results for negative base
        # with fractional exponent, TypeError without arguments, and keyword and partial-
        # application forms.
        self.assertEqual(pow(0,0), 1)
        self.assertEqual(pow(0,1), 0)
        self.assertEqual(pow(1,0), 1)
        self.assertEqual(pow(1,1), 1)

        self.assertEqual(pow(2,0), 1)
        self.assertEqual(pow(2,10), 1024)
        self.assertEqual(pow(2,20), 1024*1024)
        self.assertEqual(pow(2,30), 1024*1024*1024)

        self.assertEqual(pow(-2,0), 1)
        self.assertEqual(pow(-2,1), -2)
        self.assertEqual(pow(-2,2), 4)
        self.assertEqual(pow(-2,3), -8)

        self.assertAlmostEqual(pow(0.,0), 1.)
        self.assertAlmostEqual(pow(0.,1), 0.)
        self.assertAlmostEqual(pow(1.,0), 1.)
        self.assertAlmostEqual(pow(1.,1), 1.)

        self.assertAlmostEqual(pow(2.,0), 1.)
        self.assertAlmostEqual(pow(2.,10), 1024.)
        self.assertAlmostEqual(pow(2.,20), 1024.*1024.)
        self.assertAlmostEqual(pow(2.,30), 1024.*1024.*1024.)

        self.assertAlmostEqual(pow(-2.,0), 1.)
        self.assertAlmostEqual(pow(-2.,1), -2.)
        self.assertAlmostEqual(pow(-2.,2), 4.)
        self.assertAlmostEqual(pow(-2.,3), -8.)

        for x in 2, 2.0:
            for y in 10, 10.0:
                for z in 1000, 1000.0:
                    if isinstance(x, float) or \
                       isinstance(y, float) or \
                       isinstance(z, float):
                        self.assertRaises(TypeError, pow, x, y, z)
                    else:
                        self.assertAlmostEqual(pow(x, y, z), 24.0)

        self.assertAlmostEqual(pow(-1, 0.5), 1j)
        self.assertAlmostEqual(pow(-1, 1/3), 0.5 + 0.8660254037844386j)

        # See test_pow for additional tests for three-argument pow.
        self.assertEqual(pow(-1, -2, 3), 1)
        self.assertRaises(ValueError, pow, 1, 2, 0)

        self.assertRaises(TypeError, pow)

        # Test passing in arguments as keywords.
        self.assertEqual(pow(0, exp=0), 1)
        self.assertEqual(pow(base=2, exp=4), 16)
        self.assertEqual(pow(base=5, exp=2, mod=14), 11)
        twopow = partial(pow, base=2)
        self.assertEqual(twopow(exp=5), 32)
        fifth_power = partial(pow, exp=5)
        self.assertEqual(fifth_power(2), 32)
        mod10 = partial(pow, mod=10)
        self.assertEqual(mod10(2, 6), 4)
        self.assertEqual(mod10(exp=6, base=2), 4)

    @torch._dynamo.error_on_graph_break(False)
    def test_input(self):
        # [FUNCTION DOC] Check input with files and in-memory streams as stdin: returns lines
        # without newline, ValueError on closed stdin, TypeError for bad arguments, EOFError on
        # empty input, RuntimeError when stdin or stdout is missing. Side effect: swaps sys.stdin
        # and sys.stdout, restored in a finally block; writes the sample file.
        self.write_testfile()
        fp = open(TESTFN, encoding="utf-8")
        savestdin = sys.stdin
        savestdout = sys.stdout # Eats the echo
        try:
            sys.stdin = fp
            sys.stdout = BitBucket()
            self.assertEqual(input(), "1+1")
            self.assertEqual(input(), 'The quick brown fox jumps over the lazy dog.')
            self.assertEqual(input('testing\n'), 'Dear John')

            # SF 1535165: don't segfault on closed stdin
            # sys.stdout must be a regular file for triggering
            sys.stdout = savestdout
            sys.stdin.close()
            self.assertRaises(ValueError, input)

            sys.stdout = BitBucket()
            sys.stdin = io.StringIO("NULL\0")
            self.assertRaises(TypeError, input, 42, 42)
            sys.stdin = io.StringIO("    'whitespace'")
            self.assertEqual(input(), "    'whitespace'")
            sys.stdin = io.StringIO()
            self.assertRaises(EOFError, input)

            del sys.stdout
            self.assertRaises(RuntimeError, input, 'prompt')
            del sys.stdin
            self.assertRaises(RuntimeError, input, 'prompt')
        finally:
            sys.stdin = savestdin
            sys.stdout = savestdout
            fp.close()

    @torch._dynamo.error_on_graph_break(False)
    def test_input_gh130163(self):
        # [FUNCTION DOC] Regression test for a crash when the stream attribute lookup replaces the
        # standard streams during input; the call must not crash. Side effect: temporarily replaces
        # stdin, stdout and stderr and forces garbage collection.
        class X(io.StringIO):
            # [CLASS DOC] HELPER: StringIO subclass whose first attribute access (when armed)
            # replaces the standard streams with fresh instances and collects garbage.
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: On the first access while armed, disarm and replace
                # stdout, stderr and stdin, collect garbage, then defer to normal attribute lookup.
                nonlocal patch
                if patch:
                    patch = False
                    sys.stdout = X()
                    sys.stderr = X()
                    sys.stdin = X('input\n')
                    support.gc_collect()
                return io.StringIO.__getattribute__(self, name)

        with (support.swap_attr(sys, 'stdout', None),
              support.swap_attr(sys, 'stderr', None),
              support.swap_attr(sys, 'stdin', None)):
            patch = False
            # the only references:
            sys.stdout = X()
            sys.stderr = X()
            sys.stdin = X('input\n')
            patch = True
            input()  # should not crash

    # test_int(): see test_int.py for tests of built-in function int().

    def test_repr(self):
        # [FUNCTION DOC] Check repr on basic values and self-referencing containers (rendered with
        # an ellipsis marker).
        self.assertEqual(repr(''), '\'\'')
        self.assertEqual(repr(0), '0')
        self.assertEqual(repr(()), '()')
        self.assertEqual(repr([]), '[]')
        self.assertEqual(repr({}), '{}')
        a = []
        a.append(a)
        self.assertEqual(repr(a), '[[...]]')
        a = {}
        a[0] = a
        self.assertEqual(repr(a), '{0: {...}}')

    def test_round(self):
        # [FUNCTION DOC] Check round on floats and ints: result types, half-to-even behavior,
        # negative digits, keyword arguments, TypeError for bad argument counts, and delegation to
        # a rounding hook (TypeError when absent or only set on the instance).
        self.assertEqual(round(0.0), 0.0)
        self.assertEqual(type(round(0.0)), int)
        self.assertEqual(round(1.0), 1.0)
        self.assertEqual(round(10.0), 10.0)
        self.assertEqual(round(1000000000.0), 1000000000.0)
        self.assertEqual(round(1e20), 1e20)

        self.assertEqual(round(-1.0), -1.0)
        self.assertEqual(round(-10.0), -10.0)
        self.assertEqual(round(-1000000000.0), -1000000000.0)
        self.assertEqual(round(-1e20), -1e20)

        self.assertEqual(round(0.1), 0.0)
        self.assertEqual(round(1.1), 1.0)
        self.assertEqual(round(10.1), 10.0)
        self.assertEqual(round(1000000000.1), 1000000000.0)

        self.assertEqual(round(-1.1), -1.0)
        self.assertEqual(round(-10.1), -10.0)
        self.assertEqual(round(-1000000000.1), -1000000000.0)

        self.assertEqual(round(0.9), 1.0)
        self.assertEqual(round(9.9), 10.0)
        self.assertEqual(round(999999999.9), 1000000000.0)

        self.assertEqual(round(-0.9), -1.0)
        self.assertEqual(round(-9.9), -10.0)
        self.assertEqual(round(-999999999.9), -1000000000.0)

        self.assertEqual(round(-8.0, -1), -10.0)
        self.assertEqual(type(round(-8.0, -1)), float)

        self.assertEqual(type(round(-8.0, 0)), float)
        self.assertEqual(type(round(-8.0, 1)), float)

        # Check even / odd rounding behaviour
        self.assertEqual(round(5.5), 6)
        self.assertEqual(round(6.5), 6)
        self.assertEqual(round(-5.5), -6)
        self.assertEqual(round(-6.5), -6)

        # Check behavior on ints
        self.assertEqual(round(0), 0)
        self.assertEqual(round(8), 8)
        self.assertEqual(round(-8), -8)
        self.assertEqual(type(round(0)), int)
        self.assertEqual(type(round(-8, -1)), int)
        self.assertEqual(type(round(-8, 0)), int)
        self.assertEqual(type(round(-8, 1)), int)

        # test new kwargs
        self.assertEqual(round(number=-8.0, ndigits=-1), -10.0)

        self.assertRaises(TypeError, round)

        # test generic rounding delegation for reals
        class TestRound:
            # [CLASS DOC] HELPER: Fixture with a rounding hook returning 23.
            def __round__(self):
                # [FUNCTION DOC] HELPER: Return 23.
                return 23

        class TestNoRound:
            # [CLASS DOC] HELPER: Fixture without a rounding hook.
            pass

        self.assertEqual(round(TestRound()), 23)

        self.assertRaises(TypeError, round, 1, 2, 3)
        self.assertRaises(TypeError, round, TestNoRound())

        t = TestNoRound()
        t.__round__ = lambda *args: args
        self.assertRaises(TypeError, round, t)
        self.assertRaises(TypeError, round, t, 0)

    # Some versions of glibc for alpha have a bug that affects
    # float -> integer rounding (floor, ceil, rint, round) for
    # values in the range [2**52, 2**53).  See:
    #
    # We skip this test on Linux/alpha if it would fail.
    linux_alpha = (platform.system().startswith('Linux') and
                   platform.machine().startswith('alpha'))
    system_round_bug = round(5e15+1) != 5e15+1
    @unittest.skipIf(linux_alpha and system_round_bug,
                     "test will fail;  failure is probably due to a "
                     "buggy system round function")
    def test_round_large(self):
        # [FUNCTION DOC] Check round leaves large integral floats (around 5e15) unchanged. Skipped
        # on Linux/alpha when the system rounding bug is present.
        # Issue #1869: integral floats should remain unchanged
        self.assertEqual(round(5e15-1), 5e15-1)
        self.assertEqual(round(5e15), 5e15)
        self.assertEqual(round(5e15+1), 5e15+1)
        self.assertEqual(round(5e15+2), 5e15+2)
        self.assertEqual(round(5e15+3), 5e15+3)

    def test_bug_27936(self):
        # [FUNCTION DOC] Check that passing None as the digits argument of round behaves like
        # omitting it, for ints, floats, decimals and fractions, including the result type.
        # Verify that ndigits=None means the same as passing in no argument
        for x in [1234,
                  1234.56,
                  decimal.Decimal('1234.56'),
                  fractions.Fraction(123456, 100)]:
            self.assertEqual(round(x, None), round(x))
            self.assertEqual(type(round(x, None)), type(round(x)))

    def test_setattr(self):
        # [FUNCTION DOC] Check setattr sets an attribute and raises TypeError for missing arguments
        # and non-string names. Side effect: sets an attribute on the sys module and does not
        # remove it.
        # KNOWN ISSUE: the attribute spam set on the sys module is never deleted (test_delattr also
        # sets and deletes it), leaking global state.
        setattr(sys, 'spam', 1)
        self.assertEqual(sys.spam, 1)
        self.assertRaises(TypeError, setattr)
        self.assertRaises(TypeError, setattr, sys)
        self.assertRaises(TypeError, setattr, sys, 'spam')
        msg = r"^attribute name must be string, not 'int'$"
        self.assertRaisesRegex(TypeError, msg, setattr, sys, 1, 'spam')

    # test_str(): see test_str.py and test_bytes.py for str() tests.

    def test_sum(self):
        # [FUNCTION DOC] Check sum: ints, floats, bools, lists with a list start value, start
        # keyword, large starts, negative zero and infinities, and TypeError for strings, bytes,
        # non-iterables and other invalid inputs; the start list is not mutated.
        self.assertEqual(sum([]), 0)
        self.assertEqual(sum(list(range(2,8))), 27)
        self.assertEqual(sum(iter(list(range(2,8)))), 27)
        self.assertEqual(sum(Squares(10)), 285)
        self.assertEqual(sum(iter(Squares(10))), 285)
        self.assertEqual(sum([[1], [2], [3]], []), [1, 2, 3])

        self.assertEqual(sum(range(10), 1000), 1045)
        self.assertEqual(sum(range(10), start=1000), 1045)
        self.assertEqual(sum(range(10), 2**31-5), 2**31+40)
        self.assertEqual(sum(range(10), 2**63-5), 2**63+40)

        self.assertEqual(sum(i % 2 != 0 for i in range(10)), 5)
        self.assertEqual(sum((i % 2 != 0 for i in range(10)), 2**31-3),
                         2**31+2)
        self.assertEqual(sum((i % 2 != 0 for i in range(10)), 2**63-3),
                         2**63+2)
        self.assertIs(sum([], False), False)

        self.assertEqual(sum(i / 2 for i in range(10)), 22.5)
        self.assertEqual(sum((i / 2 for i in range(10)), 1000), 1022.5)
        self.assertEqual(sum((i / 2 for i in range(10)), 1000.25), 1022.75)
        self.assertEqual(sum([0.5, 1]), 1.5)
        self.assertEqual(sum([1, 0.5]), 1.5)
        self.assertEqual(repr(sum([-0.0])), '0.0')
        self.assertEqual(repr(sum([-0.0], -0.0)), '-0.0')
        self.assertEqual(repr(sum([], -0.0)), '-0.0')
        self.assertTrue(math.isinf(sum([float("inf"), float("inf")])))
        self.assertTrue(math.isinf(sum([1e308, 1e308])))

        self.assertRaises(TypeError, sum)
        self.assertRaises(TypeError, sum, 42)
        self.assertRaises(TypeError, sum, ['a', 'b', 'c'])
        self.assertRaises(TypeError, sum, ['a', 'b', 'c'], '')
        self.assertRaises(TypeError, sum, [b'a', b'c'], b'')
        values = [bytearray(b'a'), bytearray(b'b')]
        self.assertRaises(TypeError, sum, values, bytearray(b''))
        self.assertRaises(TypeError, sum, [[1], [2], [3]])
        self.assertRaises(TypeError, sum, [{2:3}])
        self.assertRaises(TypeError, sum, [{2:3}]*2, {2:3})
        self.assertRaises(TypeError, sum, [], '')
        self.assertRaises(TypeError, sum, [], b'')
        self.assertRaises(TypeError, sum, [], bytearray())

        class BadSeq:
            # [CLASS DOC] HELPER: Fixture whose item hook raises ValueError.
            def __getitem__(self, index):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError
        self.assertRaises(ValueError, sum, BadSeq())

        empty = []
        sum(([x] for x in range(10)), empty)
        self.assertEqual(empty, [])

    @requires_IEEE_754
    @unittest.skipIf(HAVE_DOUBLE_ROUNDING,
                         "sum accuracy not guaranteed on machines with double rounding")
    @support.cpython_only    # Other implementations may choose a different algorithm
    def test_sum_accuracy(self):
        # [FUNCTION DOC] Check sum gives exactly rounded float results (compensated summation).
        # Requires IEEE 754, no double rounding, and the reference interpreter.
        self.assertEqual(sum([0.1] * 10), 1.0)
        self.assertEqual(sum([1.0, 10E100, 1.0, -10E100]), 2.0)

    def test_type(self):
        # [FUNCTION DOC] Check type of a string matches type of another string and differs from
        # type of a tuple.
        self.assertEqual(type(''),  type('123'))
        self.assertNotEqual(type(''), type(()))

    # We don't want self in vars(), so these are static methods

    @staticmethod
    def get_vars_f0():
        # [FUNCTION DOC] HELPER: Static method returning the local variable mapping of its own
        # frame (empty).
        return vars()

    @staticmethod
    def get_vars_f2():
        # [FUNCTION DOC] HELPER: Static method calling the zero-variable helper, then returning its
        # own local variables (a and b).
        BuiltinTest.get_vars_f0()
        a = 1
        b = 2
        return vars()

    class C_get_vars(object):
        # [CLASS DOC] HELPER: Fixture whose dict attribute is a property returning a fixed mapping.
        def getDict(self):
            # [FUNCTION DOC] HELPER: Return a mapping with key a set to two.
            return {'a':2}
        __dict__ = property(fget=getDict)

    def test_vars(self):
        # [FUNCTION DOC] Check vars: no-argument form equals dir, module form, static-method
        # helpers, TypeError for objects without a dict or too many arguments, and objects whose
        # dict is a property.
        self.assertEqual(set(vars()), set(dir()))
        self.assertEqual(set(vars(sys)), set(dir(sys)))
        self.assertEqual(self.get_vars_f0(), {})
        self.assertEqual(self.get_vars_f2(), {'a': 1, 'b': 2})
        self.assertRaises(TypeError, vars, 42, 42)
        self.assertRaises(TypeError, vars, 42)
        self.assertEqual(vars(self.C_get_vars()), {'a':2})

    def iter_error(self, iterable, error):
        # [FUNCTION DOC] HELPER: Collect items from an iterable into a list while asserting that
        # iteration raises the given exception; return the items collected before the error.
        """Collect `iterable` into a list, catching an expected `error`."""
        items = []
        with self.assertRaises(error):
            for item in iterable:
                items.append(item)
        return items

    def test_zip(self):
        # [FUNCTION DOC] Check zip: pairs from sequences of equal and unequal length, custom
        # sequences, no arguments, TypeError for non-iterables, error propagation, no pre-
        # allocation from unbounded sequences.
        a = (1, 2, 3)
        b = (4, 5, 6)
        t = [(1, 4), (2, 5), (3, 6)]
        self.assertEqual(list(zip(a, b)), t)
        b = [4, 5, 6]
        self.assertEqual(list(zip(a, b)), t)
        b = (4, 5, 6, 7)
        self.assertEqual(list(zip(a, b)), t)
        class I:
            # [CLASS DOC] HELPER: Sequence fixture supporting indexes zero to two with values four
            # to six.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return index plus four for indexes zero to two, otherwise
                # raise IndexError.
                if i < 0 or i > 2: raise IndexError
                return i + 4
        self.assertEqual(list(zip(a, I())), t)
        self.assertEqual(list(zip()), [])
        self.assertEqual(list(zip(*[])), [])
        self.assertRaises(TypeError, zip, None)
        class G:
            # [CLASS DOC] HELPER: Fixture that is not iterable.
            pass
        self.assertRaises(TypeError, zip, a, G())
        self.assertRaises(RuntimeError, zip, a, TestFailingIter())

        # Make sure zip doesn't try to allocate a billion elements for the
        # result list when one of its arguments doesn't say how long it is.
        # A MemoryError is the most likely failure mode.
        class SequenceWithoutALength:
            # [CLASS DOC] HELPER: Sequence fixture without a length yielding indexes below five.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the index below five, otherwise raise IndexError.
                if i == 5:
                    raise IndexError
                else:
                    return i
        self.assertEqual(
            list(zip(SequenceWithoutALength(), range(2**30))),
            list(enumerate(range(5)))
        )

        class BadSeq:
            # [CLASS DOC] HELPER: Sequence fixture raising ValueError at index five.
            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the index, but raise ValueError at five.
                if i == 5:
                    raise ValueError
                else:
                    return i
        self.assertRaises(ValueError, list, zip(BadSeq(), BadSeq()))

    def test_zip_pickle(self):
        # [FUNCTION DOC] Check zip objects pickle round trip at every pickle protocol using the
        # shared pickle helper.
        a = (1, 2, 3)
        b = (4, 5, 6)
        t = [(1, 4), (2, 5), (3, 6)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z1 = zip(a, b)
            self.check_iter_pickle(z1, t, proto)

    def test_zip_pickle_strict(self):
        # [FUNCTION DOC] Check strict-mode zip objects pickle round trip at every pickle protocol
        # using the shared pickle helper.
        a = (1, 2, 3)
        b = (4, 5, 6)
        t = [(1, 4), (2, 5), (3, 6)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z1 = zip(a, b, strict=True)
            self.check_iter_pickle(z1, t, proto)

    def test_zip_pickle_strict_fail(self):
        # [FUNCTION DOC] Check strict-mode zip with unequal lengths raises ValueError after
        # yielding the common prefix, both before and after a pickle round trip.
        a = (1, 2, 3)
        b = (4, 5, 6, 7)
        t = [(1, 4), (2, 5), (3, 6)]
        for proto in range(pickle.HIGHEST_PROTOCOL + 1):
            z1 = zip(a, b, strict=True)
            z2 = pickle.loads(pickle.dumps(z1, proto))
            self.assertEqual(self.iter_error(z1, ValueError), t)
            self.assertEqual(self.iter_error(z2, ValueError), t)

    def test_zip_bad_iterable(self):
        # [FUNCTION DOC] Check that the exact exception raised by an iterable iteration hook
        # propagates out of zip creation.
        exception = TypeError()

        class BadIterable:
            # [CLASS DOC] HELPER: Fixture whose iteration hook raises the shared exception object.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Always raises the captured TypeError instance.
                raise exception

        with self.assertRaises(TypeError) as cm:
            zip(BadIterable())

        self.assertIs(cm.exception, exception)

    def test_zip_strict(self):
        # [FUNCTION DOC] Check strict-mode zip yields tuples for equal lengths and raises
        # ValueError when any iterable is shorter or longer.
        self.assertEqual(tuple(zip((1, 2, 3), 'abc', strict=True)),
                         ((1, 'a'), (2, 'b'), (3, 'c')))
        self.assertRaises(ValueError, tuple,
                          zip((1, 2, 3, 4), 'abc', strict=True))
        self.assertRaises(ValueError, tuple,
                          zip((1, 2), 'abc', strict=True))
        self.assertRaises(ValueError, tuple,
                          zip((1, 2), (1, 2), 'abc', strict=True))

    def test_zip_strict_iterators(self):
        # [FUNCTION DOC] Check strict-mode zip consumes the shorter iterators in a predictable way
        # when lengths mismatch (leftover iterators advanced as expected).
        x = iter(range(5))
        y = [0]
        z = iter(range(5))
        self.assertRaises(ValueError, list,
                          (zip(x, y, z, strict=True)))
        self.assertEqual(next(x), 2)
        self.assertEqual(next(z), 1)

    def test_zip_strict_error_handling(self):
        # [FUNCTION DOC] Check strict-mode zip error ordering: custom exceptions from iterators
        # take priority or ValueError for length mismatch, depending on which iterator ends first.
        class Error(Exception):
            # [CLASS DOC] HELPER: Custom exception fixture.
            pass

        class Iter:
            # [CLASS DOC] HELPER: Iterator fixture yielding size-1 down to 0 then raising Error.
            # Attribute: size.
            def __init__(self, size):
                # [FUNCTION DOC] HELPER: Store the number of items to yield.
                self.size = size
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Decrement the counter; raise Error when it goes negative,
                # otherwise return the counter.
                self.size -= 1
                if self.size < 0:
                    raise Error
                return self.size

        l1 = self.iter_error(zip("AB", Iter(1), strict=True), Error)
        self.assertEqual(l1, [("A", 0)])
        l2 = self.iter_error(zip("AB", Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l2, [("A", 1, "A")])
        l3 = self.iter_error(zip("AB", Iter(2), "ABC", strict=True), Error)
        self.assertEqual(l3, [("A", 1, "A"), ("B", 0, "B")])
        l4 = self.iter_error(zip("AB", Iter(3), strict=True), ValueError)
        self.assertEqual(l4, [("A", 2), ("B", 1)])
        l5 = self.iter_error(zip(Iter(1), "AB", strict=True), Error)
        self.assertEqual(l5, [(0, "A")])
        l6 = self.iter_error(zip(Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l6, [(1, "A")])
        l7 = self.iter_error(zip(Iter(2), "ABC", strict=True), Error)
        self.assertEqual(l7, [(1, "A"), (0, "B")])
        l8 = self.iter_error(zip(Iter(3), "AB", strict=True), ValueError)
        self.assertEqual(l8, [(2, "A"), (1, "B")])

    def test_zip_strict_error_handling_stopiteration(self):
        # [FUNCTION DOC] Check strict-mode zip when an iterator ends with StopIteration: length
        # mismatches always produce ValueError.
        class Iter:
            # [CLASS DOC] HELPER: Iterator fixture yielding size-1 down to 0 then raising
            # StopIteration. Attribute: size.
            def __init__(self, size):
                # [FUNCTION DOC] HELPER: Store the number of items to yield.
                self.size = size
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self
            def __next__(self):
                # [FUNCTION DOC] HELPER: Decrement the counter; raise StopIteration when it goes
                # negative, otherwise return the counter.
                self.size -= 1
                if self.size < 0:
                    raise StopIteration
                return self.size

        l1 = self.iter_error(zip("AB", Iter(1), strict=True), ValueError)
        self.assertEqual(l1, [("A", 0)])
        l2 = self.iter_error(zip("AB", Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l2, [("A", 1, "A")])
        l3 = self.iter_error(zip("AB", Iter(2), "ABC", strict=True), ValueError)
        self.assertEqual(l3, [("A", 1, "A"), ("B", 0, "B")])
        l4 = self.iter_error(zip("AB", Iter(3), strict=True), ValueError)
        self.assertEqual(l4, [("A", 2), ("B", 1)])
        l5 = self.iter_error(zip(Iter(1), "AB", strict=True), ValueError)
        self.assertEqual(l5, [(0, "A")])
        l6 = self.iter_error(zip(Iter(2), "A", strict=True), ValueError)
        self.assertEqual(l6, [(1, "A")])
        l7 = self.iter_error(zip(Iter(2), "ABC", strict=True), ValueError)
        self.assertEqual(l7, [(1, "A"), (0, "B")])
        l8 = self.iter_error(zip(Iter(3), "AB", strict=True), ValueError)
        self.assertEqual(l8, [(2, "A"), (1, "B")])

    @support.cpython_only
    def test_zip_result_gc(self):
        # [FUNCTION DOC] Regression test: the recycled result tuple of zip is tracked by the
        # garbage collector after being mutated and returned. Requires the reference interpreter.
        # bpo-42536: zip's tuple-reuse speed trick breaks the GC's assumptions
        # about what can be untracked. Make sure we re-track result tuples
        # whenever we reuse them.
        it = zip([[]])
        gc.collect()
        # That GC collection probably untracked the recycled internal result
        # tuple, which is initialized to (None,). Make sure it's re-tracked when
        # it's mutated and returned from __next__:
        self.assertTrue(gc.is_tracked(next(it)))

    def test_format(self):
        # [FUNCTION DOC] Check format: basic call, user format hooks on classes and subclasses,
        # empty spec equals str for builtin types, TypeError for hooks returning non-strings or
        # non-string specs, object format with a non-empty spec raises TypeError mentioning the
        # class name, and str subclasses as spec.
        # Test the basic machinery of the format() builtin.  Don't test
        #  the specifics of the various formatters
        self.assertEqual(format(3, ''), '3')

        # Returns some classes to use for various tests.  There's
        #  an old-style version, and a new-style version
        def classes_new():
            # [FUNCTION DOC] HELPER: Define and return four fixture classes with format hooks (a
            # base, a subclass, a hook-less base, and subclasses of it).
            class A(object):
                # [CLASS DOC] HELPER: Class with a value and a format hook appending the spec to
                # its string form.
                def __init__(self, x):
                    # [FUNCTION DOC] HELPER: Store the value.
                    self.x = x
                def __format__(self, format_spec):
                    # [FUNCTION DOC] HELPER: Return the value as text followed by the spec.
                    return str(self.x) + format_spec
            class DerivedFromA(A):
                # [CLASS DOC] HELPER: Subclass of A without changes.
                pass

            # [CLASS DOC] HELPER: Class without a format hook.
            class Simple(object): pass
            class DerivedFromSimple(Simple):
                # [CLASS DOC] HELPER: Subclass with a value and a format hook appending the spec.
                def __init__(self, x):
                    # [FUNCTION DOC] HELPER: Store the value.
                    self.x = x
                def __format__(self, format_spec):
                    # [FUNCTION DOC] HELPER: Return the value as text followed by the spec.
                    return str(self.x) + format_spec
            # [CLASS DOC] HELPER: Further subclass without changes.
            class DerivedFromSimple2(DerivedFromSimple): pass
            return A, DerivedFromA, DerivedFromSimple, DerivedFromSimple2

        def class_test(A, DerivedFromA, DerivedFromSimple, DerivedFromSimple2):
            # [FUNCTION DOC] HELPER: Assert format output for instances of the four
            # fixture classes.
            self.assertEqual(format(A(3), 'spec'), '3spec')
            self.assertEqual(format(DerivedFromA(4), 'spec'), '4spec')
            self.assertEqual(format(DerivedFromSimple(5), 'abc'), '5abc')
            self.assertEqual(format(DerivedFromSimple2(10), 'abcdef'),
                             '10abcdef')

        class_test(*classes_new())

        def empty_format_spec(value):
            # [FUNCTION DOC] HELPER: Assert that format with an empty or missing spec equals the
            # string form.
            # test that:
            #  format(x, '') == str(x)
            #  format(x) == str(x)
            self.assertEqual(format(value, ""), str(value))
            self.assertEqual(format(value), str(value))

        # for builtin types, format(x, "") == str(x)
        empty_format_spec(17**13)
        empty_format_spec(1.0)
        empty_format_spec(3.1415e104)
        empty_format_spec(-3.1415e104)
        empty_format_spec(3.1415e-104)
        empty_format_spec(-3.1415e-104)
        empty_format_spec(object)
        empty_format_spec(None)

        # TypeError because self.__format__ returns the wrong type
        class BadFormatResult:
            # [CLASS DOC] HELPER: Fixture whose format hook returns a float.
            def __format__(self, format_spec):
                # [FUNCTION DOC] HELPER: Return a float, which format must reject with TypeError.
                return 1.0
        self.assertRaises(TypeError, format, BadFormatResult(), "")

        # TypeError because format_spec is not unicode or str
        self.assertRaises(TypeError, format, object(), 4)
        self.assertRaises(TypeError, format, object(), object())

        # tests for object.__format__ really belong elsewhere, but
        #  there's no good place to put them
        x = object().__format__('')
        self.assertTrue(x.startswith('<object object at'))

        # first argument to object.__format__ must be string
        self.assertRaises(TypeError, object().__format__, 3)
        self.assertRaises(TypeError, object().__format__, object())
        self.assertRaises(TypeError, object().__format__, None)

        # --------------------------------------------------------------------
        # Issue #7994: object.__format__ with a non-empty format string is
        # disallowed
        class A:
            # [CLASS DOC] HELPER: Fixture whose format hook formats an empty string with the spec.
            def __format__(self, fmt_str):
                # [FUNCTION DOC] HELPER: Return an empty string formatted with the spec.
                return format('', fmt_str)

        self.assertEqual(format(A()), '')
        self.assertEqual(format(A(), ''), '')
        self.assertEqual(format(A(), 's'), '')

        class B:
            # [CLASS DOC] HELPER: Fixture class with no format hook.
            pass

        class C(object):
            # [CLASS DOC] HELPER: Fixture class with no format hook.
            pass

        for cls in [object, B, C]:
            obj = cls()
            self.assertEqual(format(obj), str(obj))
            self.assertEqual(format(obj, ''), str(obj))
            with self.assertRaisesRegex(TypeError,
                                        r'\b%s\b' % re.escape(cls.__name__)):
                format(obj, 's')
        # --------------------------------------------------------------------

        # make sure we can take a subclass of str as a format spec
        # [CLASS DOC] HELPER: str subclass used as a format spec.
        class DerivedFromStr(str): pass
        self.assertEqual(format(0, DerivedFromStr('10')), '         0')

    def test_bin(self):
        # [FUNCTION DOC] Check bin output for zero, small and very large positive and
        # negative ints.
        self.assertEqual(bin(0), '0b0')
        self.assertEqual(bin(1), '0b1')
        self.assertEqual(bin(-1), '-0b1')
        self.assertEqual(bin(2**65), '0b1' + '0' * 65)
        self.assertEqual(bin(2**65-1), '0b' + '1' * 65)
        self.assertEqual(bin(-(2**65)), '-0b1' + '0' * 65)
        self.assertEqual(bin(-(2**65-1)), '-0b' + '1' * 65)

    def test_bytearray_translate(self):
        # [FUNCTION DOC] Check bytearray translate raises ValueError for a short table with
        # deletions argument and TypeError for a table of the wrong element kind.
        x = bytearray(b"abc")
        self.assertRaises(ValueError, x.translate, b"1", 1)
        self.assertRaises(TypeError, x.translate, b"1"*256, 1)

    def test_bytearray_extend_error(self):
        # [FUNCTION DOC] Check bytearray extend propagates ValueError from an iterator producing
        # bad values.
        array = bytearray()
        bad_iter = map(int, "X")
        self.assertRaises(ValueError, array.extend, bad_iter)

    def test_bytearray_join_with_misbehaving_iterator(self):
        # [FUNCTION DOC] Regression test: join on a bytearray whose iterator clears the bytearray
        # raises BufferError.
        # Issue #112625
        array = bytearray(b',')
        def iterator():
            # [FUNCTION DOC] HELPER: Generator that clears the outer bytearray then
            # yields two byte strings.
            array.clear()
            yield b'A'
            yield b'B'
        self.assertRaises(BufferError, array.join, iterator())

    def test_bytearray_join_with_custom_iterator(self):
        # [FUNCTION DOC] Check bytearray join works with a custom iterator.
        # Issue #112625
        array = bytearray(b',')
        def iterator():
            # [FUNCTION DOC] HELPER: Generator yielding two byte strings.
            yield b'A'
            yield b'B'
        self.assertEqual(bytearray(b'A,B'), array.join(iterator()))

    def test_construct_singletons(self):
        # [FUNCTION DOC] Check None, Ellipsis and NotImplemented types return the singleton when
        # called with no arguments and raise TypeError with arguments.
        for const in None, Ellipsis, NotImplemented:
            tp = type(const)
            self.assertIs(tp(), const)
            self.assertRaises(TypeError, tp, 1, 2)
            self.assertRaises(TypeError, tp, a=1, b=2)

    def test_warning_notimplemented(self):
        # [FUNCTION DOC] Check that using NotImplemented in a boolean context emits a
        # DeprecationWarning while still being true.
        # Issue #35712: NotImplemented is a sentinel value that should never
        # be evaluated in a boolean context (virtually all such use cases
        # are a result of accidental misuse implementing rich comparison
        # operations in terms of one another).
        # For the time being, it will continue to evaluate as a true value, but
        # issue a deprecation warning (with the eventual intent to make it
        # a TypeError).
        self.assertWarns(DeprecationWarning, bool, NotImplemented)
        with self.assertWarns(DeprecationWarning):
            self.assertTrue(NotImplemented)
        with self.assertWarns(DeprecationWarning):
            self.assertFalse(not NotImplemented)

    def test_singleton_attribute_access(self):
        # [FUNCTION DOC] Check NotImplemented and Ellipsis singletons reject setting and reading
        # missing attributes on instances (AttributeError) and on their types (TypeError when
        # setting, AttributeError when reading).
        for singleton in (NotImplemented, Ellipsis):
            with self.subTest(singleton):
                self.assertIs(type(singleton), singleton.__class__)
                self.assertIs(type(singleton).__class__, type)

                # Missing instance attributes:
                with self.assertRaises(AttributeError):
                    singleton.prop = 1
                with self.assertRaises(AttributeError):
                    singleton.prop

                # Missing class attributes:
                with self.assertRaises(TypeError):
                    type(singleton).prop = 1
                with self.assertRaises(AttributeError):
                    type(singleton).prop


class TestBreakpoint(CPythonTestCase):
    # [CLASS DOC] Test case for the breakpoint builtin, the breakpoint hook on sys and the
    # breakpoint environment variable. State: resources (an ExitStack of cleanups) and env
    # (environment variable guard).
    def setUp(self):
        # [FUNCTION DOC] Reset state: call the base setup, create the cleanup stack, guard the
        # environment, remove the breakpoint environment variable, and restore the default hook on
        # sys. Everything is undone by cleanups.
        # These tests require a clean slate environment.  For example, if the
        # test suite is run with $PYTHONBREAKPOINT set to something else, it
        # will mess up these tests.  Similarly for sys.breakpointhook.
        # Cleaning the slate here means you can't use breakpoint() to debug
        # these tests, but I think that's okay.  Just use pdb.set_trace() if
        # you must.
        super().setUp()
        self.resources = ExitStack()
        self.addCleanup(self.resources.close)
        self.env = self.resources.enter_context(EnvironmentVarGuard())
        del self.env['PYTHONBREAKPOINT']
        self.resources.enter_context(
            swap_attr(sys, 'breakpointhook', sys.__breakpointhook__))

    @torch._dynamo.error_on_graph_break(False)
    def test_breakpoint(self):
        # [FUNCTION DOC] Check breakpoint calls the debugger set-trace function by default.
        with patch('pdb.set_trace') as mock:
            breakpoint()
        mock.assert_called_once()

    def test_breakpoint_with_breakpointhook_set(self):
        # [FUNCTION DOC] Check breakpoint calls a replaced hook on sys exactly once
        # with no arguments.
        my_breakpointhook = MagicMock()
        sys.breakpointhook = my_breakpointhook
        breakpoint()
        my_breakpointhook.assert_called_once_with()

    def test_breakpoint_with_breakpointhook_reset(self):
        # [FUNCTION DOC] Check that after resetting the hook to the default, breakpoint calls the
        # debugger again and not the old hook.
        my_breakpointhook = MagicMock()
        sys.breakpointhook = my_breakpointhook
        breakpoint()
        my_breakpointhook.assert_called_once_with()
        # Reset the hook and it will not be called again.
        sys.breakpointhook = sys.__breakpointhook__
        with patch('pdb.set_trace') as mock:
            breakpoint()
            mock.assert_called_once_with()
        my_breakpointhook.assert_called_once_with()

    def test_breakpoint_with_args_and_keywords(self):
        # [FUNCTION DOC] Check breakpoint forwards positional and keyword arguments to the hook.
        my_breakpointhook = MagicMock()
        sys.breakpointhook = my_breakpointhook
        breakpoint(1, 2, 3, four=4, five=5)
        my_breakpointhook.assert_called_once_with(1, 2, 3, four=4, five=5)

    def test_breakpoint_with_passthru_error(self):
        # [FUNCTION DOC] Check that arguments pass through to a hook that does not accept them,
        # producing TypeError.
        def my_breakpointhook():
            # [FUNCTION DOC] HELPER: Hook taking no arguments.
            pass
        sys.breakpointhook = my_breakpointhook
        self.assertRaises(TypeError, breakpoint, 1, 2, 3, four=4, five=5)

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    def test_envar_good_path_builtin(self):
        # [FUNCTION DOC] Check the environment variable naming a builtin makes breakpoint call that
        # builtin with the arguments. Skipped when the environment is ignored.
        self.env['PYTHONBREAKPOINT'] = 'int'
        with patch('builtins.int') as mock:
            breakpoint('7')
            mock.assert_called_once_with('7')

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    @torch._dynamo.error_on_graph_break(False)
    def test_envar_good_path_other(self):
        # [FUNCTION DOC] Check the environment variable naming a module attribute makes breakpoint
        # call it. Skipped when the environment is ignored.
        self.env['PYTHONBREAKPOINT'] = 'sys.exit'
        with patch('sys.exit') as mock:
            breakpoint()
            mock.assert_called_once_with()

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    @torch._dynamo.error_on_graph_break(False)
    def test_envar_good_path_noop_0(self):
        # [FUNCTION DOC] Check the environment variable set to zero disables breakpoint. Skipped
        # when the environment is ignored.
        self.env['PYTHONBREAKPOINT'] = '0'
        with patch('pdb.set_trace') as mock:
            breakpoint()
            mock.assert_not_called()

    @torch._dynamo.error_on_graph_break(False)
    def test_envar_good_path_empty_string(self):
        # [FUNCTION DOC] Check an empty environment variable behaves as if unset.
        # PYTHONBREAKPOINT='' is the same as it not being set.
        self.env['PYTHONBREAKPOINT'] = ''
        with patch('pdb.set_trace') as mock:
            breakpoint()
            mock.assert_called_once_with()

    @unittest.skipIf(sys.flags.ignore_environment, '-E was given')
    @torch._dynamo.error_on_graph_break(False)
    def test_envar_unimportable(self):
        # [FUNCTION DOC] Check unimportable values of the environment variable produce a
        # RuntimeWarning with a specific message and no debugger call. Skipped when the environment
        # is ignored.
        for envar in (
                '.', '..', '.foo', 'foo.', '.int', 'int.',
                '.foo.bar', '..foo.bar', '/./',
                'nosuchbuiltin',
                'nosuchmodule.nosuchcallable',
                ):
            with self.subTest(envar=envar):
                self.env['PYTHONBREAKPOINT'] = envar
                mock = self.resources.enter_context(patch('pdb.set_trace'))
                w = self.resources.enter_context(check_warnings(quiet=True))
                breakpoint()
                self.assertEqual(
                    str(w.message),
                    f'Ignoring unimportable $PYTHONBREAKPOINT: "{envar}"')
                self.assertEqual(w.category, RuntimeWarning)
                mock.assert_not_called()

    @torch._dynamo.error_on_graph_break(False)
    def test_envar_ignored_when_hook_is_set(self):
        # [FUNCTION DOC] Check that the environment variable is ignored when the sys
        # hook has been replaced.
        self.env['PYTHONBREAKPOINT'] = 'sys.exit'
        with patch('sys.exit') as mock:
            sys.breakpointhook = int
            breakpoint()
            mock.assert_not_called()

    @torch._dynamo.error_on_graph_break(False)
    def test_runtime_error_when_hook_is_lost(self):
        # [FUNCTION DOC] Check breakpoint raises RuntimeError when the sys hook has been deleted.
        del sys.breakpointhook
        with self.assertRaises(RuntimeError):
            breakpoint()


@unittest.skipUnless(pty, "the pty and signal modules must be available")
class PtyTests(CPythonTestCase):
    # [CLASS DOC] Test case running input in a child process under a pseudo terminal so stdin and
    # stdout are real terminals. Skipped when the pty and signal modules are unavailable.
    """Tests that use a pseudo terminal to guarantee stdin and stdout are
    terminals in the test environment"""

    @staticmethod
    def handle_sighup(signum, frame):
        # [FUNCTION DOC] HELPER: Signal handler that ignores the hangup signal (which may be raised
        # when closing the pty).
        # bpo-40140: if the process is the session leader, os.close(fd)
        # of "pid, fd = pty.fork()" can raise SIGHUP signal:
        # just ignore the signal.
        pass

    def run_child(self, child, terminal_input):
        # [FUNCTION DOC] HELPER: Run the child function with a temporary hangup handler, restoring
        # the old handler afterwards; return the lines produced.
        old_sighup = signal.signal(signal.SIGHUP, self.handle_sighup)
        try:
            return self._run_child(child, terminal_input)
        finally:
            signal.signal(signal.SIGHUP, old_sighup)

    def _run_child(self, child, terminal_input):
        # [FUNCTION DOC] HELPER: Fork a child under a pty and a result pipe, send it the terminal
        # input, and read result lines from the pipe until the child closes it. If anything other
        # than two lines arrives, fail with the child's output. Closes the pty, waits for exit code
        # zero and returns the lines. Skips the test if forking fails.
        r, w = os.pipe()  # Pipe test results from child back to parent
        try:
            pid, fd = pty.fork()
        except (OSError, AttributeError) as e:
            os.close(r)
            os.close(w)
            self.skipTest("pty.fork() raised {}".format(e))
            raise

        if pid == 0:
            # Child
            try:
                os.close(r)
                with open(w, "w") as wpipe:
                    child(wpipe)
            except:
                traceback.print_exc()
            finally:
                # We don't want to return to unittest...
                os._exit(0)

        # Parent
        os.close(w)
        os.write(fd, terminal_input)

        # Get results from the pipe
        with open(r, encoding="utf-8") as rpipe:
            lines = []
            while True:
                line = rpipe.readline().strip()
                if line == "":
                    # The other end was closed => the child exited
                    break
                lines.append(line)

        # Check the result was got and corresponds to the user's terminal input
        if len(lines) != 2:
            # Something went wrong, try to get at stderr
            # Beware of Linux raising EIO when the slave is closed
            child_output = bytearray()
            while True:
                try:
                    chunk = os.read(fd, 3000)
                except OSError:  # Assume EIO
                    break
                if not chunk:
                    break
                child_output.extend(chunk)
            os.close(fd)
            child_output = child_output.decode("ascii", "ignore")
            self.fail("got %d lines in pipe but expected 2, child output was:\n%s"
                      % (len(lines), child_output))

        # bpo-40155: Close the PTY before waiting for the child process
        # completion, otherwise the child process hangs on AIX.
        os.close(fd)

        support.wait_process(pid, exitcode=0)

        return lines

    def check_input_tty(self, prompt, terminal_input, stdio_encoding=None, *,
                        expected=None,
                        stdin_errors='surrogateescape',
                        stdout_errors='replace'):
        # [FUNCTION DOC] HELPER: Run input in a child with the given prompt and terminal input
        # (optionally rewrapping stdio with an encoding and error handlers) and compare the echoed
        # result to the expected value, derived from the input if not given. Skips when the process
        # lacks ttys or the child does not see a tty.
        if not sys.stdin.isatty() or not sys.stdout.isatty():
            self.skipTest("stdin and stdout must be ttys")
        def child(wpipe):
            # [FUNCTION DOC] HELPER: Runs in the child process: optionally rewrap stdin and stdout
            # with the encoding, report whether both are ttys, then report the result of input (or
            # the exception) through the pipe.
            # Check the error handlers are accounted for
            if stdio_encoding:
                sys.stdin = io.TextIOWrapper(sys.stdin.detach(),
                                             encoding=stdio_encoding,
                                             errors=stdin_errors)
                sys.stdout = io.TextIOWrapper(sys.stdout.detach(),
                                              encoding=stdio_encoding,
                                              errors=stdout_errors)
            print("tty =", sys.stdin.isatty() and sys.stdout.isatty(), file=wpipe)
            try:
                print(ascii(input(prompt)), file=wpipe)
            except BaseException as e:
                print(ascii(f'{e.__class__.__name__}: {e!s}'), file=wpipe)
        with self.detach_readline():
            lines = self.run_child(child, terminal_input + b"\r\n")
        # Check we did exercise the GNU readline path
        self.assertIn(lines[0], {'tty = True', 'tty = False'})
        if lines[0] != 'tty = True':
            self.skipTest("standard IO in should have been a tty")
        input_result = eval(lines[1])   # ascii() -> eval() roundtrip
        if expected is None:
            if stdio_encoding:
                expected = terminal_input.decode(stdio_encoding, 'surrogateescape')
            else:
                expected = terminal_input.decode(sys.stdin.encoding)  # what else?
        self.assertEqual(input_result, expected)

    @contextlib.contextmanager
    def detach_readline(self):
        # [FUNCTION DOC] HELPER: Context manager that temporarily detaches the readline function
        # pointer from the interpreter, if readline is loaded, so input does not use the readline
        # implementation; restores it on exit.
        # bpo-13886: When the readline module is loaded, PyOS_Readline() uses
        # the readline implementation. In some cases, the Python readline
        # callback rlhandler() is called by readline with a string without
        # non-ASCII characters.
        # Unlink readline temporarily from PyOS_Readline() for those tests,
        # since test_builtin is not intended to test
        # the readline module, but the builtins module.
        if "readline" in sys.modules:
            c = import_module("ctypes")
            fp_api = "PyOS_ReadlineFunctionPointer"
            prev_value = c.c_void_p.in_dll(c.pythonapi, fp_api).value
            c.c_void_p.in_dll(c.pythonapi, fp_api).value = None
            try:
                yield
            finally:
                c.c_void_p.in_dll(c.pythonapi, fp_api).value = prev_value
        else:
            yield

    def test_input_tty(self):
        # [FUNCTION DOC] Check input via a tty with a plain ASCII prompt.
        # Test input() functionality when wired to a tty
        self.check_input_tty("prompt", b"quux")

    def test_input_tty_non_ascii(self):
        # [FUNCTION DOC] Check input via a tty uses the stdio encoding for non-ASCII
        # prompt and input.
        # Check stdin/stdout encoding is used when invoking PyOS_Readline()
        self.check_input_tty("prompté", b"quux\xc3\xa9", "utf-8")

    def test_input_tty_non_ascii_unicode_errors(self):
        # [FUNCTION DOC] Check input via a tty with an ASCII stdio encoding uses the configured
        # error handlers for non-ASCII data.
        # Check stdin/stdout error handler is used when invoking PyOS_Readline()
        self.check_input_tty("prompté", b"quux\xe9", "ascii")

    def test_input_tty_null_in_prompt(self):
        # [FUNCTION DOC] Check a prompt containing a null character raises ValueError.
        self.check_input_tty("prompt\0", b"",
                expected='ValueError: input: prompt string cannot contain '
                         'null characters')

    def test_input_tty_nonencodable_prompt(self):
        # [FUNCTION DOC] Check an unencodable prompt with strict output errors raises
        # UnicodeEncodeError.
        self.check_input_tty("prompté", b"quux", "ascii", stdout_errors='strict',
                expected="UnicodeEncodeError: 'ascii' codec can't encode "
                         "character '\\xe9' in position 6: ordinal not in "
                         "range(128)")

    def test_input_tty_nondecodable_input(self):
        # [FUNCTION DOC] Check undecodable input with strict input errors raises
        # UnicodeDecodeError.
        self.check_input_tty("prompt", b"quux\xe9", "ascii", stdin_errors='strict',
                expected="UnicodeDecodeError: 'ascii' codec can't decode "
                         "byte 0xe9 in position 4: ordinal not in "
                         "range(128)")

    def test_input_no_stdout_fileno(self):
        # [FUNCTION DOC] Check that when stdout has no usable file number, input does not use the
        # original terminal output and writes the prompt to the replacement stream.
        # Issue #24402: If stdin is the original terminal but stdout.fileno()
        # fails, do not use the original stdout file descriptor
        def child(wpipe):
            # [FUNCTION DOC] HELPER: Runs in the child: report whether stdin is a tty, swap stdout
            # for an in-memory stream, call input, then report the captured output.
            print("stdin.isatty():", sys.stdin.isatty(), file=wpipe)
            sys.stdout = io.StringIO()  # Does not support fileno()
            input("prompt")
            print("captured:", ascii(sys.stdout.getvalue()), file=wpipe)
        lines = self.run_child(child, b"quux\r")
        expected = (
            "stdin.isatty(): True",
            "captured: 'prompt'",
        )
        self.assertSequenceEqual(lines, expected)

class TestSorted(CPythonTestCase):
    # [CLASS DOC] Test case for the sorted builtin.

    def test_basic(self):
        # [FUNCTION DOC] Check sorted on shuffled data with default order, key function and reverse
        # flag, and that the input is not changed.
        data = list(range(100))
        copy = data[:]
        random.shuffle(copy)
        self.assertEqual(data, sorted(copy))
        self.assertNotEqual(data, copy)

        data.reverse()
        random.shuffle(copy)
        self.assertEqual(data, sorted(copy, key=lambda x: -x))
        self.assertNotEqual(data, copy)
        random.shuffle(copy)
        self.assertEqual(data, sorted(copy, reverse=True))
        self.assertNotEqual(data, copy)

    def test_bad_arguments(self):
        # [FUNCTION DOC] Check sorted takes its iterable positionally only and other arguments by
        # keyword only.
        # Issue #29327: The first argument is positional-only.
        sorted([])
        with self.assertRaises(TypeError):
            sorted(iterable=[])
        # Other arguments are keyword-only
        sorted([], key=None)
        with self.assertRaises(TypeError):
            sorted([], None)

    def test_inputtypes(self):
        # [FUNCTION DOC] Check sorted gives the same result for strings, lists, tuples, sets and
        # similar input types.
        s = 'abracadabra'
        types = [list, tuple, str]
        for T in types:
            self.assertEqual(sorted(s), sorted(T(s)))

        s = ''.join(set(s))  # unique letters only
        types = [str, set, frozenset, list, tuple, dict.fromkeys]
        for T in types:
            self.assertEqual(sorted(s), sorted(T(s)))

    def test_baddecorator(self):
        # [FUNCTION DOC] Check sorted raises TypeError when given positional arguments
        # beyond the iterable.
        data = 'The quick Brown fox Jumped over The lazy Dog'.split()
        self.assertRaises(TypeError, sorted, data, None, lambda x,y: 0)


class ShutdownTest(CPythonTestCase):
    # [CLASS DOC] Test case for interpreter shutdown behavior of builtins.

    def test_cleanup(self):
        # [FUNCTION DOC] Check builtins remain usable during object cleanup at interpreter
        # shutdown, by running a child interpreter with ASCII output encoding and comparing its
        # output lines.
        # Issue #19255: builtins are still available at shutdown
        code = """if 1:
            import builtins
            import sys

            class C:
                def __del__(self):
                    print("before")
                    # Check that builtins still exist
                    len(())
                    print("after")

            c = C()
            # Make this module survive until builtins and sys are cleaned
            builtins.here = sys.modules[__name__]
            sys.here = sys.modules[__name__]
            # Create a reference loop so that this module needs to go
            # through a GC phase.
            here = sys.modules[__name__]
            """
        # Issue #20599: Force ASCII encoding to get a codec implemented in C,
        # otherwise the codec may be unloaded before C.__del__() is called, and
        # so print("before") fails because the codec cannot be used to encode
        # "before" to sys.stdout.encoding. For example, on Windows,
        # sys.stdout.encoding is the OEM code page and these code pages are
        # implemented in Python
        rc, out, err = assert_python_ok("-c", code,
                                        PYTHONIOENCODING="ascii")
        self.assertEqual(["before", "after"], out.decode().splitlines())


@cpython_only
class ImmortalTests(CPythonTestCase):
    # [CLASS DOC] Test case checking that small shared objects (None, booleans, Ellipsis,
    # NotImplemented, small ints) keep the immortal reference count. Reference-interpreter only.
    # Class constants: IMMORTAL_REFCOUNT (depends on pointer size) and IMMORTALS.

    if sys.maxsize < (1 << 32):
        IMMORTAL_REFCOUNT = (1 << 30) - 1
    else:
        IMMORTAL_REFCOUNT = (1 << 32) - 1

    IMMORTALS = (None, True, False, Ellipsis, NotImplemented, *range(-5, 257))

    def assert_immortal(self, immortal):
        # [FUNCTION DOC] HELPER: Assert the object has the immortal reference count.
        with self.subTest(immortal):
            self.assertEqual(sys.getrefcount(immortal), self.IMMORTAL_REFCOUNT)

    def test_immortals(self):
        # [FUNCTION DOC] Check all listed immortal objects have the immortal reference count.
        for immortal in self.IMMORTALS:
            self.assert_immortal(immortal)

    def test_list_repeat_respect_immortality(self):
        # [FUNCTION DOC] Check repeating a list of immortal objects keeps them immortal.
        refs = list(self.IMMORTALS) * 42
        for immortal in self.IMMORTALS:
            self.assert_immortal(immortal)

    def test_tuple_repeat_respect_immortality(self):
        # [FUNCTION DOC] Check repeating a tuple of immortal objects keeps them immortal.
        refs = tuple(self.IMMORTALS) * 42
        for immortal in self.IMMORTALS:
            self.assert_immortal(immortal)


class TestType(CPythonTestCase):
    # [CLASS DOC] Test case for the three-argument and keyword behavior of the type builtin when
    # creating classes.
    def test_new_type(self):
        # [FUNCTION DOC] Check creating classes with type: name, qualified name, module, bases,
        # namespace contents and instance behavior with multiple bases.
        A = type('A', (), {})
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__qualname__, 'A')
        self.assertEqual(A.__module__, __name__)
        self.assertEqual(A.__bases__, (object,))
        self.assertIs(A.__base__, object)
        self.assertNotIn('__firstlineno__', A.__dict__)
        x = A()
        self.assertIs(type(x), A)
        self.assertIs(x.__class__, A)

        class B:
            # [CLASS DOC] HELPER: Base class fixture with a method.
            def ham(self):
                # [FUNCTION DOC] HELPER: Return a string built from the instance number.
                return 'ham%d' % self
        C = type('C', (B, int), {'spam': lambda self: 'spam%s' % self})
        self.assertEqual(C.__name__, 'C')
        self.assertEqual(C.__qualname__, 'C')
        self.assertEqual(C.__module__, __name__)
        self.assertEqual(C.__bases__, (B, int))
        self.assertIs(C.__base__, int)
        self.assertIn('spam', C.__dict__)
        self.assertNotIn('ham', C.__dict__)
        x = C(42)
        self.assertEqual(x, 42)
        self.assertIs(type(x), C)
        self.assertIs(x.__class__, C)
        self.assertEqual(x.ham(), 'ham42')
        self.assertEqual(x.spam(), 'spam42')
        self.assertEqual(x.to_bytes(2, 'little'), b'\x2a\x00')

    def test_type_nokwargs(self):
        # [FUNCTION DOC] Check type rejects keyword arguments.
        with self.assertRaises(TypeError):
            type('a', (), {}, x=5)
        with self.assertRaises(TypeError):
            type('a', (), dict={})

    def test_type_name(self):
        # [FUNCTION DOC] Check class naming: arbitrary names are accepted, null characters raise
        # ValueError, unencodable names raise UnicodeEncodeError, bytes raise TypeError; renaming
        # changes the name only and failed renames leave it unchanged.
        for name in 'A', '\xc4', '\U0001f40d', 'B.A', '42', '':
            with self.subTest(name=name):
                A = type(name, (), {})
                self.assertEqual(A.__name__, name)
                self.assertEqual(A.__qualname__, name)
                self.assertEqual(A.__module__, __name__)
        with self.assertRaises(ValueError):
            type('A\x00B', (), {})
        with self.assertRaises(UnicodeEncodeError):
            type('A\udcdcB', (), {})
        with self.assertRaises(TypeError):
            type(b'A', (), {})

        C = type('C', (), {})
        for name in 'A', '\xc4', '\U0001f40d', 'B.A', '42', '':
            with self.subTest(name=name):
                C.__name__ = name
                self.assertEqual(C.__name__, name)
                self.assertEqual(C.__qualname__, 'C')
                self.assertEqual(C.__module__, __name__)

        A = type('C', (), {})
        with self.assertRaises(ValueError):
            A.__name__ = 'A\x00B'
        self.assertEqual(A.__name__, 'C')
        with self.assertRaises(UnicodeEncodeError):
            A.__name__ = 'A\udcdcB'
        self.assertEqual(A.__name__, 'C')
        with self.assertRaises(TypeError):
            A.__name__ = b'A'
        self.assertEqual(A.__name__, 'C')

    def test_type_qualname(self):
        # [FUNCTION DOC] Check qualified name handling in the namespace and by assignment; non-
        # string values raise TypeError and leave the old value.
        A = type('A', (), {'__qualname__': 'B.C'})
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__qualname__, 'B.C')
        self.assertEqual(A.__module__, __name__)
        with self.assertRaises(TypeError):
            type('A', (), {'__qualname__': b'B'})
        self.assertEqual(A.__qualname__, 'B.C')

        A.__qualname__ = 'D.E'
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__qualname__, 'D.E')
        with self.assertRaises(TypeError):
            A.__qualname__ = b'B'
        self.assertEqual(A.__qualname__, 'D.E')

    def test_type_firstlineno(self):
        # [FUNCTION DOC] Check the first-line-number entry: kept when passed in the namespace,
        # removed when the module is reassigned, and settable afterwards.
        A = type('A', (), {'__firstlineno__': 42})
        self.assertEqual(A.__name__, 'A')
        self.assertEqual(A.__module__, __name__)
        self.assertEqual(A.__dict__['__firstlineno__'], 42)
        A.__module__ = 'testmodule'
        self.assertEqual(A.__module__, 'testmodule')
        self.assertNotIn('__firstlineno__', A.__dict__)
        A.__firstlineno__ = 43
        self.assertEqual(A.__dict__['__firstlineno__'], 43)

    def test_type_typeparams(self):
        # [FUNCTION DOC] Check a class with a type parameter exposes the type parameters; they can
        # be replaced but not deleted (TypeError).
        class A[T]:
            # [CLASS DOC] HELPER: Generic class fixture with one type parameter.
            pass
        T, = A.__type_params__
        self.assertIsInstance(T, typing.TypeVar)
        A.__type_params__ = "whatever"
        self.assertEqual(A.__type_params__, "whatever")
        with self.assertRaises(TypeError):
            del A.__type_params__
        self.assertEqual(A.__type_params__, "whatever")

    def test_type_doc(self):
        # [FUNCTION DOC] Check the doc attribute accepts arbitrary values at class creation and
        # assignment, but unencodable strings at creation raise UnicodeEncodeError.
        for doc in 'x', '\xc4', '\U0001f40d', 'x\x00y', b'x', 42, None:
            A = type('A', (), {'__doc__': doc})
            self.assertEqual(A.__doc__, doc)
        with self.assertRaises(UnicodeEncodeError):
            type('A', (), {'__doc__': 'x\udcdcy'})

        A = type('A', (), {})
        self.assertEqual(A.__doc__, None)
        for doc in 'x', '\xc4', '\U0001f40d', 'x\x00y', 'x\udcdcy', b'x', 42, None:
            A.__doc__ = doc
            self.assertEqual(A.__doc__, doc)

    def test_bad_args(self):
        # [FUNCTION DOC] Check type raises TypeError for wrong numbers or kinds of arguments, non-
        # dict namespaces, invalid bases and conflicting bases.
        with self.assertRaises(TypeError):
            type()
        with self.assertRaises(TypeError):
            type('A', ())
        with self.assertRaises(TypeError):
            type('A', (), {}, ())
        with self.assertRaises(TypeError):
            type('A', (), dict={})
        with self.assertRaises(TypeError):
            type('A', [], {})
        with self.assertRaises(TypeError):
            type('A', (), types.MappingProxyType({}))
        with self.assertRaises(TypeError):
            type('A', (None,), {})
        with self.assertRaises(TypeError):
            type('A', (bool,), {})
        with self.assertRaises(TypeError):
            type('A', (int, str), {})

    def test_bad_slots(self):
        # [FUNCTION DOC] Check type raises TypeError or ValueError for invalid slot declarations
        # (bad kinds, empty names, duplicates, conflict with class attributes, dict or weakref
        # slots that the base already provides).
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': b'x'})
        with self.assertRaises(TypeError):
            type('A', (int,), {'__slots__': 'x'})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': ''})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': '42'})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': 'x\x00y'})
        with self.assertRaises(ValueError):
            type('A', (), {'__slots__': 'x', 'x': 0})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': ('__dict__', '__dict__')})
        with self.assertRaises(TypeError):
            type('A', (), {'__slots__': ('__weakref__', '__weakref__')})

        class B:
            # [CLASS DOC] HELPER: Base class fixture.
            pass
        with self.assertRaises(TypeError):
            type('A', (B,), {'__slots__': '__dict__'})
        with self.assertRaises(TypeError):
            type('A', (B,), {'__slots__': '__weakref__'})

    def test_namespace_order(self):
        # [FUNCTION DOC] Check that the namespace order of an ordered dictionary is preserved when
        # creating a class.
        # bpo-34320: namespace should preserve order
        od = collections.OrderedDict([('a', 1), ('b', 2)])
        od.move_to_end('a')
        expected = list(od.items())

        C = type('C', (), od)
        self.assertEqual(list(C.__dict__.items())[:2], [('b', 2), ('a', 1)])


if __name__ == "__main__":
    run_tests()
