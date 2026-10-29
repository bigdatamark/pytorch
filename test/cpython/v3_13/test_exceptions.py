# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 test_exceptions tests (built-in
# exception types and semantics). Covers raising/catching every builtin exception, SyntaxError
# messages and error offsets, exception attributes and pickling, __notes__, tracebacks,
# chaining (__cause__/__context__), cleanup of the "except ... as" name and exception state
# (reference leaks checked through weakrefs), generator interaction with sys.exception,
# recursion errors, MemoryError handling, NameError/AttributeError/ImportError details,
# AssertionError locations, SyntaxError constructors, and PEP 626 line numbers. Many tests
# rely on CPython internals (test.support, _testcapi, subprocesses). Import-time side effects:
# installs RedirectImportFinder on sys.meta_path and tries to import _testcapi (falls back to
# a default INT_MAX if unavailable). Running the file as a script calls run_tests.
# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# Test copied from
# https://raw.githubusercontent.com/python/cpython/refs/tags/v3.13.5/Lib/test/test_exceptions.py

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch._dynamo.test_case import CPythonTestCase
from torch.testing._internal.common_utils import (
    run_tests,
    skipIfTorchDynamo,
    xfailIfTorchDynamo,
)

__TestCase = CPythonTestCase

# redirect import statements
import sys
import importlib.abc

redirect_imports = (
    "test.mapping_tests",
    "test.typinganndata",
    "test.test_grammar",
    "test.test_math",
    "test.test_iter",
    "test.typinganndata.ann_module",
)

class RedirectImportFinder(importlib.abc.MetaPathFinder):
    # [CLASS DOC] HELPER: Import hook (meta path finder) that maps selected "test.<name>"
    # module names listed in redirect_imports to the top-level module "<name>" so the copied
    # CPython tests can import their helpers. Stateless; one instance is placed on
    # sys.meta_path.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a module name in redirect_imports, strips the "test."
        # prefix, imports that top-level module, registers it in sys.modules under the
        # original name, and returns its spec. Returns None for other names or if the import
        # fails (ImportError). Side effect: mutates sys.modules and triggers an import.
        # Check if the import is the problematic one
        if fullname in redirect_imports:
            try:
                # Attempt to import the standalone module
                name = fullname.removeprefix("test.")
                r = importlib.import_module(name)
                # Redirect the module in sys.modules
                sys.modules[fullname] = r
                # Return a module spec from the found module
                return importlib.util.find_spec(name)
            except ImportError:
                return None
        return None

# Add the custom finder to sys.meta_path
sys.meta_path.insert(0, RedirectImportFinder())


# ======= END DYNAMO PATCH =======

# Python test set -- part 5, built-in exceptions

import copy
import os
import sys
import unittest
import pickle
import weakref
import errno
from codecs import BOM_UTF8
from itertools import product
from textwrap import dedent

from test.support import (captured_stderr, check_impl_detail,
                          cpython_only, gc_collect,
                          no_tracing, script_helper,
                          SuppressCrashReport,
                          force_not_colorized)
from test.support.import_helper import import_module
from test.support.os_helper import TESTFN, unlink
from test.support.warnings_helper import check_warnings
from test import support

try:
    import _testcapi
    from _testcapi import INT_MAX
except ImportError:
    _testcapi = None
    INT_MAX = 2**31 - 1


class NaiveException(Exception):
    # [CLASS DOC] HELPER: Exception subclass whose constructor takes one argument and stores
    # it as attribute x without calling the base constructor; used to check attribute
    # handling/pickling.
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the single argument x on the instance as attribute x.
        self.x = x

class SlottedNaiveException(Exception):
    # [CLASS DOC] HELPER: Like NaiveException but declares __slots__ with only x, so the
    # instance keeps the attribute in a slot; used for the same attribute/pickle checks.
    __slots__ = ('x',)
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the single argument x in the x slot.
        self.x = x

class BrokenStrException(Exception):
    # [CLASS DOC] HELPER: Exception whose string conversion always fails by raising a plain
    # Exception; used to test error handling paths that stringify exceptions.
    def __str__(self):
        # [FUNCTION DOC] HELPER: Always raises Exception with the message "str() is broken".
        raise Exception("str() is broken")

# XXX This is not really enough, each *operation* should be tested!


class ExceptionTests(__TestCase):
    # [CLASS DOC] Main test case for builtin exception behavior (raising, attributes,
    # pickling, notes, tracebacks, chaining, cleanup, generator exception state, recursion and
    # MemoryError handling, unraisable/unhandled output). Mostly stateless; some tests use the
    # TESTFN file or subprocesses. Several tests are marked cpython_only or resource-gated.

    def raise_catch(self, exc, excname):
        # [FUNCTION DOC] HELPER: Raises and catches the exception class exc twice with message
        # "spam" and checks both string forms are equal and that exc.__name__ equals excname.
        # Runs inside a subTest. Asserts via the test case; returns nothing.
        with self.subTest(exc=exc, excname=excname):
            try:
                raise exc("spam")
            except exc as err:
                buf1 = str(err)
            try:
                raise exc("spam")
            except exc as err:
                buf2 = str(err)
            self.assertEqual(buf1, buf2)
            self.assertEqual(exc.__name__, excname)

    @torch._dynamo.error_on_graph_break(False)
    def testRaising(self):
        # [FUNCTION DOC] Walks through the builtin exception classes with raise_catch and
        # exercises several concrete triggers: getattr on a missing attribute, EOFError from
        # marshal.loads of empty bytes, OSError opening a missing file, ImportError,
        # IndexError, KeyError, NameError, OverflowError (repeated doubling of an int),
        # SyntaxError via exec, TabError via compile, SystemExit via sys.exit, TypeError,
        # ValueError, ZeroDivisionError. Side effects: creates and removes the TESTFN file and
        # temporarily replaces sys.stdin. Dynamo graph breaks allowed.
        self.raise_catch(AttributeError, "AttributeError")
        self.assertRaises(AttributeError, getattr, sys, "undefined_attribute")

        self.raise_catch(EOFError, "EOFError")
        fp = open(TESTFN, 'w', encoding="utf-8")
        fp.close()
        fp = open(TESTFN, 'r', encoding="utf-8")
        savestdin = sys.stdin
        try:
            try:
                import marshal
                marshal.loads(b'')
            except EOFError:
                pass
        finally:
            sys.stdin = savestdin
            fp.close()
            unlink(TESTFN)

        self.raise_catch(OSError, "OSError")
        self.assertRaises(OSError, open, 'this file does not exist', 'r')

        self.raise_catch(ImportError, "ImportError")
        self.assertRaises(ImportError, __import__, "undefined_module")

        self.raise_catch(IndexError, "IndexError")
        x = []
        self.assertRaises(IndexError, x.__getitem__, 10)

        self.raise_catch(KeyError, "KeyError")
        x = {}
        self.assertRaises(KeyError, x.__getitem__, 'key')

        self.raise_catch(KeyboardInterrupt, "KeyboardInterrupt")

        self.raise_catch(MemoryError, "MemoryError")

        self.raise_catch(NameError, "NameError")
        try: x = undefined_variable
        except NameError: pass

        self.raise_catch(OverflowError, "OverflowError")
        x = 1
        for dummy in range(128):
            x += x  # this simply shouldn't blow up

        self.raise_catch(RuntimeError, "RuntimeError")
        self.raise_catch(RecursionError, "RecursionError")

        self.raise_catch(SyntaxError, "SyntaxError")
        try: exec('/\n')
        except SyntaxError: pass

        self.raise_catch(IndentationError, "IndentationError")

        self.raise_catch(TabError, "TabError")
        try: compile("try:\n\t1/0\n    \t1/0\nfinally:\n pass\n",
                     '<string>', 'exec')
        except TabError: pass
        else: self.fail("TabError not raised")

        self.raise_catch(SystemError, "SystemError")

        self.raise_catch(SystemExit, "SystemExit")
        self.assertRaises(SystemExit, sys.exit, 0)

        self.raise_catch(TypeError, "TypeError")
        try: [] + ()
        except TypeError: pass

        self.raise_catch(ValueError, "ValueError")
        self.assertRaises(ValueError, chr, 17<<16)

        self.raise_catch(ZeroDivisionError, "ZeroDivisionError")
        try: x = 1/0
        except ZeroDivisionError: pass

        self.raise_catch(Exception, "Exception")
        try: x = 1/0
        except Exception as e: pass

        self.raise_catch(StopAsyncIteration, "StopAsyncIteration")

    @torch._dynamo.error_on_graph_break(False)
    def testSyntaxErrorMessage(self):
        # [FUNCTION DOC] Checks that compiling certain fragments (continue outside a loop, a
        # bad f-string expression) raises SyntaxError with exactly the expected message text.
        # make sure the right exception message is raised for each of
        # these code fragments

        def ckmsg(src, msg):
            # [FUNCTION DOC] HELPER: Compiles source src and fails the test unless it raises
            # SyntaxError whose msg equals msg (also fails if no error is raised). Runs in a
            # subTest.
            with self.subTest(src=src, msg=msg):
                try:
                    compile(src, '<fragment>', 'exec')
                except SyntaxError as e:
                    if e.msg != msg:
                        self.fail("expected %s, got %s" % (msg, e.msg))
                else:
                    self.fail("failed to get expected SyntaxError")

        s = '''if 1:
        try:
            continue
        except:
            pass'''

        ckmsg(s, "'continue' not properly in loop")
        ckmsg("continue\n", "'continue' not properly in loop")
        ckmsg("f'{6 0}'", "invalid syntax. Perhaps you forgot a comma?")

    @torch._dynamo.error_on_graph_break(False)
    def testSyntaxErrorMissingParens(self):
        # [FUNCTION DOC] Checks the "Missing parentheses in call to print/exec" messages for
        # old style statements, that similar-looking invalid syntax gets a plain "invalid
        # syntax" message, and that IndentationError and TabError subclasses keep their own
        # messages.
        def ckmsg(src, msg, exception=SyntaxError):
            # [FUNCTION DOC] HELPER: Compiles src and asserts it raises the given exception
            # class (default SyntaxError) with message msg; fails the test if nothing is
            # raised.
            try:
                compile(src, '<fragment>', 'exec')
            except exception as e:
                if e.msg != msg:
                    self.fail("expected %s, got %s" % (msg, e.msg))
            else:
                self.fail("failed to get expected SyntaxError")

        s = '''print "old style"'''
        ckmsg(s, "Missing parentheses in call to 'print'. Did you mean print(...)?")

        s = '''print "old style",'''
        ckmsg(s, "Missing parentheses in call to 'print'. Did you mean print(...)?")

        s = 'print f(a+b,c)'
        ckmsg(s, "Missing parentheses in call to 'print'. Did you mean print(...)?")

        s = '''exec "old style"'''
        ckmsg(s, "Missing parentheses in call to 'exec'. Did you mean exec(...)?")

        s = 'exec f(a+b,c)'
        ckmsg(s, "Missing parentheses in call to 'exec'. Did you mean exec(...)?")

        # Check that we don't incorrectly identify '(...)' as an expression to the right
        # of 'print'

        s = 'print (a+b,c) $ 42'
        ckmsg(s, "invalid syntax")

        s = 'exec (a+b,c) $ 42'
        ckmsg(s, "invalid syntax")

        # should not apply to subclasses, see issue #31161
        s = '''if True:\nprint "No indent"'''
        ckmsg(s, "expected an indented block after 'if' statement on line 1", IndentationError)

        s = '''if True:\n        print()\n\texec "mixed tabs and spaces"'''
        ckmsg(s, "inconsistent use of tabs and spaces in indentation", TabError)

    def check(self, src, lineno, offset, end_lineno=None, end_offset=None, encoding='utf-8'):
        # [FUNCTION DOC] HELPER: Compiles source src (str or bytes) and asserts a SyntaxError
        # whose lineno and offset equal the given values; end_lineno/end_offset are compared
        # only when given. If the error has text, the matching source line (decoded with
        # encoding for bytes) must be contained in it.
        with self.subTest(source=src, lineno=lineno, offset=offset):
            with self.assertRaises(SyntaxError) as cm:
                compile(src, '<fragment>', 'exec')
            self.assertEqual(cm.exception.lineno, lineno)
            self.assertEqual(cm.exception.offset, offset)
            if end_lineno is not None:
                self.assertEqual(cm.exception.end_lineno, end_lineno)
            if end_offset is not None:
                self.assertEqual(cm.exception.end_offset, end_offset)

            if cm.exception.text is not None:
                if not isinstance(src, str):
                    src = src.decode(encoding, 'replace')
                line = src.split('\n')[lineno-1]
                self.assertIn(line, cm.exception.text)

    def test_error_offset_continuation_characters(self):
        # [FUNCTION DOC] Checks the error position reported for a source ending in a line
        # continuation character.
        check = self.check
        check('"\\\n"(1 for c in I,\\\n\\', 2, 2)

    def testSyntaxErrorOffset(self):
        # [FUNCTION DOC] Table of source fragments with expected SyntaxError line/offset (and
        # end positions in some cases), grouped by the component that reports them: parser,
        # compile, tokenizer, symtable and future handling. Uses the check helper for each
        # case.
        check = self.check
        check('def fact(x):\n\treturn x!\n', 2, 10)
        check('1 +\n', 1, 4)
        check('def spam():\n  print(1)\n print(2)', 3, 10)
        check('Python = "Python" +', 1, 20)
        check('Python = "\u1e54\xfd\u0163\u0125\xf2\xf1" +', 1, 20)
        check(b'# -*- coding: cp1251 -*-\nPython = "\xcf\xb3\xf2\xee\xed" +',
              2, 19, encoding='cp1251')
        check(b'Python = "\xcf\xb3\xf2\xee\xed" +', 1, 10)
        check('x = "a', 1, 5)
        check('lambda x: x = 2', 1, 1)
        check('f{a + b + c}', 1, 2)
        check('[file for str(file) in []\n]', 1, 11)
        check('a = « hello » « world »', 1, 5)
        check('[\nfile\nfor str(file)\nin\n[]\n]', 3, 5)
        check('[file for\n str(file) in []]', 2, 2)
        check("ages = {'Alice'=22, 'Bob'=23}", 1, 9)
        check('match ...:\n    case {**rest, "key": value}:\n        ...', 2, 19)
        check("[a b c d e f]", 1, 2)
        check("for x yfff:", 1, 7)
        check("f(a for a in b, c)", 1, 3, 1, 15)
        check("f(a for a in b if a, c)", 1, 3, 1, 20)
        check("f(a, b for b in c)", 1, 6, 1, 18)
        check("f(a, b for b in c, d)", 1, 6, 1, 18)

        # Errors thrown by compile.c
        check('class foo:return 1', 1, 11)
        check('def f():\n  continue', 2, 3)
        check('def f():\n  break', 2, 3)
        check('try:\n  pass\nexcept:\n  pass\nexcept ValueError:\n  pass', 3, 1)
        check('try:\n  pass\nexcept*:\n  pass', 3, 8)
        check('try:\n  pass\nexcept*:\n  pass\nexcept* ValueError:\n  pass', 3, 8)

        # Errors thrown by the tokenizer
        check('(0x+1)', 1, 3)
        check('x = 0xI', 1, 6)
        check('0010 + 2', 1, 1)
        check('x = 32e-+4', 1, 8)
        check('x = 0o9', 1, 7)
        check('\u03b1 = 0xI', 1, 6)
        check(b'\xce\xb1 = 0xI', 1, 6)
        check(b'# -*- coding: iso8859-7 -*-\n\xe1 = 0xI', 2, 6,
              encoding='iso8859-7')
        check(b"""if 1:
            def foo():
                '''

            def bar():
                pass

            def baz():
                '''quux'''
            """, 9, 24)
        check("pass\npass\npass\n(1+)\npass\npass\npass", 4, 4)
        check("(1+)", 1, 4)
        check("[interesting\nfoo()\n", 1, 1)
        check(b"\xef\xbb\xbf#coding: utf8\nprint('\xe6\x88\x91')\n", 0, -1)
        check("""f'''
            {
            (123_a)
            }'''""", 3, 17)
        check("""f'''
            {
            f\"\"\"
            {
            (123_a)
            }
            \"\"\"
            }'''""", 5, 17)
        check('''f"""


            {
            6
            0="""''', 5, 13)
        check('b"fooжжж"'.encode(), 1, 1, 1, 10)

        # Errors thrown by symtable.c
        check('x = [(yield i) for i in range(3)]', 1, 7)
        check('def f():\n  from _ import *', 2, 17)
        check('def f(x, x):\n  pass', 1, 10)
        check('{i for i in range(5) if (j := 0) for j in range(5)}', 1, 38)
        check('def f(x):\n  nonlocal x', 2, 3)
        check('def f(x):\n  x = 1\n  global x', 3, 3)
        check('nonlocal x', 1, 1)
        check('def f():\n  global x\n  nonlocal x', 2, 3)

        # Errors thrown by future.c
        check('from __future__ import doesnt_exist', 1, 24)
        check('from __future__ import braces', 1, 24)
        check('x=1\nfrom __future__ import division', 2, 1)
        check('foo(1=2)', 1, 5)
        check('def f():\n  x, y: int', 2, 3)
        check('[*x for x in xs]', 1, 2)
        check('foo(x for x in range(10), 100)', 1, 5)
        check('for 1 in []: pass', 1, 5)
        check('(yield i) = 2', 1, 2)
        check('def f(*):\n  pass', 1, 7)

    @unittest.skipIf(INT_MAX >= sys.maxsize, "Downcasting to int is safe for col_offset")
    @support.requires_resource('cpu')
    @support.bigmemtest(INT_MAX, memuse=2, dry_run=False)
    def testMemoryErrorBigSource(self, size):
        # [FUNCTION DOC] Builds a source whose second line is padded to size bytes and checks
        # that compiling it raises OverflowError with "Parser column offset overflow". Skipped
        # when int is as large as the platform's maximum size; needs the cpu resource and
        # large memory.
        src = b"if True:\n%*s" % (size, b"pass")
        with self.assertRaisesRegex(OverflowError, "Parser column offset overflow"):
            compile(src, '<fragment>', 'exec')

    @cpython_only
    def testSettingException(self):
        # [FUNCTION DOC] Checks C-level exception raising (via _testcapi.raise_exception) when
        # the exception class cannot be instantiated: a failing constructor yields
        # RuntimeError or TypeError with a traceback pointing at the expected frames, and a
        # non-exception class yields SystemError. Skipped parts need _testcapi.
        # test that setting an exception at the C level works even if the
        # exception object can't be constructed.

        class BadException(Exception):
            # [CLASS DOC] HELPER: Exception class whose constructor always raises
            # RuntimeError.
            def __init__(self_):
                # [FUNCTION DOC] HELPER: Raises RuntimeError("can't instantiate
                # BadException").
                raise RuntimeError("can't instantiate BadException")

        class InvalidException:
            # [CLASS DOC] HELPER: Plain class that is not an exception; used to provoke
            # SystemError.
            pass

        @unittest.skipIf(_testcapi is None, "requires _testcapi")
        def test_capi1():
            # [FUNCTION DOC] HELPER: Raising BadException with flag 1 must give TypeError
            # whose traceback frame is this function in test_exceptions.py; otherwise fails
            # the test.
            try:
                _testcapi.raise_exception(BadException, 1)
            except TypeError as err:
                co = err.__traceback__.tb_frame.f_code
                self.assertEqual(co.co_name, "test_capi1")
                self.assertTrue(co.co_filename.endswith('test_exceptions.py'))
            else:
                self.fail("Expected exception")

        @unittest.skipIf(_testcapi is None, "requires _testcapi")
        def test_capi2():
            # [FUNCTION DOC] HELPER: Raising BadException with flag 0 must give RuntimeError
            # whose traceback points to BadException.__init__ called from this function.
            try:
                _testcapi.raise_exception(BadException, 0)
            except RuntimeError as err:
                tb = err.__traceback__.tb_next
                co = tb.tb_frame.f_code
                self.assertEqual(co.co_name, "__init__")
                self.assertTrue(co.co_filename.endswith('test_exceptions.py'))
                co2 = tb.tb_frame.f_back.f_code
                self.assertEqual(co2.co_name, "test_capi2")
            else:
                self.fail("Expected exception")

        @unittest.skipIf(_testcapi is None, "requires _testcapi")
        def test_capi3():
            # [FUNCTION DOC] HELPER: Asserts that raising a non-exception class from C raises
            # SystemError.
            self.assertRaises(SystemError, _testcapi.raise_exception,
                              InvalidException, 1)

        test_capi1()
        test_capi2()
        test_capi3()

    def test_WindowsError(self):
        # [FUNCTION DOC] On platforms where WindowsError exists, checks it is an alias of
        # OSError and the mapping of Windows error codes to errno, winerror, strerror,
        # filename and str output; does nothing (passes) when WindowsError is undefined.
        try:
            WindowsError
        except NameError:
            pass
        else:
            self.assertIs(WindowsError, OSError)
            self.assertEqual(str(OSError(1001)), "1001")
            self.assertEqual(str(OSError(1001, "message")),
                             "[Errno 1001] message")
            # POSIX errno (9 aka EBADF) is untranslated
            w = OSError(9, 'foo', 'bar')
            self.assertEqual(w.errno, 9)
            self.assertEqual(w.winerror, None)
            self.assertEqual(str(w), "[Errno 9] foo: 'bar'")
            # ERROR_PATH_NOT_FOUND (win error 3) becomes ENOENT (2)
            w = OSError(0, 'foo', 'bar', 3)
            self.assertEqual(w.errno, 2)
            self.assertEqual(w.winerror, 3)
            self.assertEqual(w.strerror, 'foo')
            self.assertEqual(w.filename, 'bar')
            self.assertEqual(w.filename2, None)
            self.assertEqual(str(w), "[WinError 3] foo: 'bar'")
            # Unknown win error becomes EINVAL (22)
            w = OSError(0, 'foo', None, 1001)
            self.assertEqual(w.errno, 22)
            self.assertEqual(w.winerror, 1001)
            self.assertEqual(w.strerror, 'foo')
            self.assertEqual(w.filename, None)
            self.assertEqual(w.filename2, None)
            self.assertEqual(str(w), "[WinError 1001] foo")
            # Non-numeric "errno"
            w = OSError('bar', 'foo')
            self.assertEqual(w.errno, 'bar')
            self.assertEqual(w.winerror, None)
            self.assertEqual(w.strerror, 'foo')
            self.assertEqual(w.filename, None)
            self.assertEqual(w.filename2, None)

    @unittest.skipUnless(sys.platform == 'win32',
                         'test specific to Windows')
    def test_windows_message(self):
        # [FUNCTION DOC] Windows only: an unknown Windows error code set through ctypes must
        # produce an OSError message formatted as hexadecimal "Windows Error 0x...".
        """Should fill in unknown error code in Windows error message"""
        ctypes = import_module('ctypes')
        # this error code has no message, Python formats it as hexadecimal
        code = 3765269347
        with self.assertRaisesRegex(OSError, 'Windows Error 0x%x' % code):
            ctypes.pythonapi.PyErr_SetFromWindowsErr(code)

    def testAttributes(self):
        # [FUNCTION DOC] Table driven check of exception attributes: for each (class, args,
        # kwargs, expected attributes) tuple, builds the exception and compares repr of each
        # attribute, that the class module is builtins (except the Naive classes), and that
        # the attribute values survive a pickle round trip for every protocol (AttributeError
        # obj is not pickled and is expected to be None). Failed construction prints the case
        # to stderr and continues.
        # KNOWN ISSUE: a construction failure only prints to stderr (the re-raise is commented
        #   out), so that case is silently skipped instead of failing the test.
        # test that exception attributes are happy

        exceptionList = [
            (BaseException, (), {}, {'args' : ()}),
            (BaseException, (1, ), {}, {'args' : (1,)}),
            (BaseException, ('foo',), {},
                {'args' : ('foo',)}),
            (BaseException, ('foo', 1), {},
                {'args' : ('foo', 1)}),
            (SystemExit, ('foo',), {},
                {'args' : ('foo',), 'code' : 'foo'}),
            (OSError, ('foo',), {},
                {'args' : ('foo',), 'filename' : None, 'filename2' : None,
                 'errno' : None, 'strerror' : None}),
            (OSError, ('foo', 'bar'), {},
                {'args' : ('foo', 'bar'),
                 'filename' : None, 'filename2' : None,
                 'errno' : 'foo', 'strerror' : 'bar'}),
            (OSError, ('foo', 'bar', 'baz'), {},
                {'args' : ('foo', 'bar'),
                 'filename' : 'baz', 'filename2' : None,
                 'errno' : 'foo', 'strerror' : 'bar'}),
            (OSError, ('foo', 'bar', 'baz', None, 'quux'), {},
                {'args' : ('foo', 'bar'), 'filename' : 'baz', 'filename2': 'quux'}),
            (OSError, ('errnoStr', 'strErrorStr', 'filenameStr'), {},
                {'args' : ('errnoStr', 'strErrorStr'),
                 'strerror' : 'strErrorStr', 'errno' : 'errnoStr',
                 'filename' : 'filenameStr'}),
            (OSError, (1, 'strErrorStr', 'filenameStr'), {},
                {'args' : (1, 'strErrorStr'), 'errno' : 1,
                 'strerror' : 'strErrorStr',
                 'filename' : 'filenameStr', 'filename2' : None}),
            (SyntaxError, (), {}, {'msg' : None, 'text' : None,
                'filename' : None, 'lineno' : None, 'offset' : None,
                'end_offset': None, 'print_file_and_line' : None}),
            (SyntaxError, ('msgStr',), {},
                {'args' : ('msgStr',), 'text' : None,
                 'print_file_and_line' : None, 'msg' : 'msgStr',
                 'filename' : None, 'lineno' : None, 'offset' : None,
                 'end_offset': None}),
            (SyntaxError, ('msgStr', ('filenameStr', 'linenoStr', 'offsetStr',
                           'textStr', 'endLinenoStr', 'endOffsetStr')), {},
                {'offset' : 'offsetStr', 'text' : 'textStr',
                 'args' : ('msgStr', ('filenameStr', 'linenoStr',
                                      'offsetStr', 'textStr',
                                      'endLinenoStr', 'endOffsetStr')),
                 'print_file_and_line' : None, 'msg' : 'msgStr',
                 'filename' : 'filenameStr', 'lineno' : 'linenoStr',
                 'end_lineno': 'endLinenoStr', 'end_offset': 'endOffsetStr'}),
            (SyntaxError, ('msgStr', 'filenameStr', 'linenoStr', 'offsetStr',
                           'textStr', 'endLinenoStr', 'endOffsetStr',
                           'print_file_and_lineStr'), {},
                {'text' : None,
                 'args' : ('msgStr', 'filenameStr', 'linenoStr', 'offsetStr',
                           'textStr', 'endLinenoStr', 'endOffsetStr',
                           'print_file_and_lineStr'),
                 'print_file_and_line' : None, 'msg' : 'msgStr',
                 'filename' : None, 'lineno' : None, 'offset' : None,
                 'end_lineno': None, 'end_offset': None}),
            (UnicodeError, (), {}, {'args' : (),}),
            (UnicodeEncodeError, ('ascii', 'a', 0, 1,
                                  'ordinal not in range'), {},
                {'args' : ('ascii', 'a', 0, 1,
                                           'ordinal not in range'),
                 'encoding' : 'ascii', 'object' : 'a',
                 'start' : 0, 'reason' : 'ordinal not in range'}),
            (UnicodeDecodeError, ('ascii', bytearray(b'\xff'), 0, 1,
                                  'ordinal not in range'), {},
                {'args' : ('ascii', bytearray(b'\xff'), 0, 1,
                                           'ordinal not in range'),
                 'encoding' : 'ascii', 'object' : b'\xff',
                 'start' : 0, 'reason' : 'ordinal not in range'}),
            (UnicodeDecodeError, ('ascii', b'\xff', 0, 1,
                                  'ordinal not in range'), {},
                {'args' : ('ascii', b'\xff', 0, 1,
                                           'ordinal not in range'),
                 'encoding' : 'ascii', 'object' : b'\xff',
                 'start' : 0, 'reason' : 'ordinal not in range'}),
            (UnicodeTranslateError, ("\u3042", 0, 1, "ouch"), {},
                {'args' : ('\u3042', 0, 1, 'ouch'),
                 'object' : '\u3042', 'reason' : 'ouch',
                 'start' : 0, 'end' : 1}),
            (NaiveException, ('foo',), {},
                {'args': ('foo',), 'x': 'foo'}),
            (SlottedNaiveException, ('foo',), {},
                {'args': ('foo',), 'x': 'foo'}),
            (AttributeError, ('foo',), dict(name='name', obj='obj'),
                dict(args=('foo',), name='name', obj='obj')),
        ]
        try:
            # More tests are in test_WindowsError
            exceptionList.append(
                (WindowsError, (1, 'strErrorStr', 'filenameStr'), {},
                    {'args' : (1, 'strErrorStr'),
                     'strerror' : 'strErrorStr', 'winerror' : None,
                     'errno' : 1,
                     'filename' : 'filenameStr', 'filename2' : None})
            )
        except NameError:
            pass

        for exc, args, kwargs, expected in exceptionList:
            try:
                e = exc(*args, **kwargs)
            except:
                print(f"\nexc={exc!r}, args={args!r}", file=sys.stderr)
                # raise
            else:
                # Verify module name
                if not type(e).__name__.endswith('NaiveException'):
                    self.assertEqual(type(e).__module__, 'builtins')
                # Verify no ref leaks in Exc_str()
                s = str(e)
                for checkArgName in expected:
                    value = getattr(e, checkArgName)
                    self.assertEqual(repr(value),
                                     repr(expected[checkArgName]),
                                     '%r.%s == %r, expected %r' % (
                                     e, checkArgName,
                                     value, expected[checkArgName]))

                # test for pickling support
                for p in [pickle]:
                    for protocol in range(p.HIGHEST_PROTOCOL + 1):
                        s = p.dumps(e, protocol)
                        new = p.loads(s)
                        for checkArgName in expected:
                            got = repr(getattr(new, checkArgName))
                            if exc == AttributeError and checkArgName == 'obj':
                                # See GH-103352, we're not pickling
                                # obj at this point. So verify it's None.
                                want = repr(None)
                            else:
                                want = repr(expected[checkArgName])
                            self.assertEqual(got, want,
                                             'pickled "%r", attribute "%s' %
                                             (e, checkArgName))

    def test_setstate(self):
        # [FUNCTION DOC] Checks Exception.__setstate__: it merges dict entries into the
        # instance attributes, can reset args, and leaves other attributes untouched.
        e = Exception(42)
        e.blah = 53
        self.assertEqual(e.args, (42,))
        self.assertEqual(e.blah, 53)
        self.assertRaises(AttributeError, getattr, e, 'a')
        self.assertRaises(AttributeError, getattr, e, 'b')
        e.__setstate__({'a': 1 , 'b': 2})
        self.assertEqual(e.args, (42,))
        self.assertEqual(e.blah, 53)
        self.assertEqual(e.a, 1)
        self.assertEqual(e.b, 2)
        e.__setstate__({'a': 11, 'args': (1,2,3), 'blah': 35})
        self.assertEqual(e.args, (1,2,3))
        self.assertEqual(e.blah, 35)
        self.assertEqual(e.a, 11)
        self.assertEqual(e.b, 2)

    def test_invalid_setstate(self):
        # [FUNCTION DOC] __setstate__ with a non-dict argument must raise TypeError "state is
        # not a dictionary".
        e = Exception(42)
        with self.assertRaisesRegex(TypeError, "state is not a dictionary"):
            e.__setstate__(42)

    def test_notes(self):
        # [FUNCTION DOC] Checks add_note and __notes__ for BaseException, Exception and
        # ValueError: notes accumulate in a list, a non-str note raises TypeError, deleting
        # __notes__ resets it, and add_note raises TypeError if __notes__ was replaced with a
        # non-list value.
        for e in [BaseException(1), Exception(2), ValueError(3)]:
            with self.subTest(e=e):
                self.assertFalse(hasattr(e, '__notes__'))
                e.add_note("My Note")
                self.assertEqual(e.__notes__, ["My Note"])

                with self.assertRaises(TypeError):
                    e.add_note(42)
                self.assertEqual(e.__notes__, ["My Note"])

                e.add_note("Your Note")
                self.assertEqual(e.__notes__, ["My Note", "Your Note"])

                del e.__notes__
                self.assertFalse(hasattr(e, '__notes__'))

                e.add_note("Our Note")
                self.assertEqual(e.__notes__, ["Our Note"])

                e.__notes__ = 42
                self.assertEqual(e.__notes__, 42)

                with self.assertRaises(TypeError):
                    e.add_note("will not work")
                self.assertEqual(e.__notes__, 42)

    def testWithTraceback(self):
        # [FUNCTION DOC] with_traceback returns the same exception object with __traceback__
        # set, for BaseException, a builtin subclass and a user subclass.
        try:
            raise IndexError(4)
        except Exception as e:
            tb = e.__traceback__

        e = BaseException().with_traceback(tb)
        self.assertIsInstance(e, BaseException)
        self.assertEqual(e.__traceback__, tb)

        e = IndexError(5).with_traceback(tb)
        self.assertIsInstance(e, IndexError)
        self.assertEqual(e.__traceback__, tb)

        class MyException(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass used to check with_traceback on
            # user classes.
            pass

        e = MyException().with_traceback(tb)
        self.assertIsInstance(e, MyException)
        self.assertEqual(e.__traceback__, tb)

    def testInvalidTraceback(self):
        # [FUNCTION DOC] Assigning a non-traceback to __traceback__ must raise TypeError
        # mentioning "__traceback__ must be a traceback".
        try:
            Exception().__traceback__ = 5
        except TypeError as e:
            self.assertIn("__traceback__ must be a traceback", str(e))
        else:
            self.fail("No exception raised")

    def test_invalid_setattr(self):
        # [FUNCTION DOC] Invalid values for args, __traceback__, __cause__ and __context__
        # must raise TypeError with the specific messages.
        TE = TypeError
        exc = Exception()
        msg = "'int' object is not iterable"
        self.assertRaisesRegex(TE, msg, setattr, exc, 'args', 1)
        msg = "__traceback__ must be a traceback or None"
        self.assertRaisesRegex(TE, msg, setattr, exc, '__traceback__', 1)
        msg = "exception cause must be None or derive from BaseException"
        self.assertRaisesRegex(TE, msg, setattr, exc, '__cause__', 1)
        msg = "exception context must be None or derive from BaseException"
        self.assertRaisesRegex(TE, msg, setattr, exc, '__context__', 1)

    def test_invalid_delattr(self):
        # [FUNCTION DOC] Deleting args, __traceback__, __cause__ or __context__ must raise
        # TypeError "may not be deleted".
        TE = TypeError
        try:
            raise IndexError(4)
        except Exception as e:
            exc = e

        msg = "may not be deleted"
        self.assertRaisesRegex(TE, msg, delattr, exc, 'args')
        self.assertRaisesRegex(TE, msg, delattr, exc, '__traceback__')
        self.assertRaisesRegex(TE, msg, delattr, exc, '__cause__')
        self.assertRaisesRegex(TE, msg, delattr, exc, '__context__')

    def testNoneClearsTracebackAttr(self):
        # [FUNCTION DOC] Setting __traceback__ to None clears a previously set traceback.
        try:
            raise IndexError(4)
        except Exception as e:
            tb = e.__traceback__

        e = Exception()
        e.__traceback__ = tb
        e.__traceback__ = None
        self.assertEqual(e.__traceback__, None)

    def testChainingAttrs(self):
        # [FUNCTION DOC] Fresh exceptions (builtin and user-defined subclass) have __context__
        # and __cause__ equal to None.
        e = Exception()
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)

        e = TypeError()
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)

        class MyException(OSError):
            # [CLASS DOC] HELPER: Empty OSError subclass used to check default chaining
            # attributes.
            pass

        e = MyException()
        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)

    def testChainingDescriptors(self):
        # [FUNCTION DOC] Checks __suppress_context__ semantics: it starts false, becomes true
        # when __cause__ or __context__ is assigned, and can be assigned explicitly.
        try:
            raise Exception()
        except Exception as exc:
            e = exc

        self.assertIsNone(e.__context__)
        self.assertIsNone(e.__cause__)
        self.assertFalse(e.__suppress_context__)

        e.__context__ = NameError()
        e.__cause__ = None
        self.assertIsInstance(e.__context__, NameError)
        self.assertIsNone(e.__cause__)
        self.assertTrue(e.__suppress_context__)
        e.__suppress_context__ = False
        self.assertFalse(e.__suppress_context__)

    def testKeywordArgs(self):
        # [FUNCTION DOC] Builtin exceptions reject keyword arguments with TypeError, while a
        # user subclass with its own constructor may accept them.
        # test that builtin exception don't take keyword args,
        # but user-defined subclasses can if they want
        self.assertRaises(TypeError, BaseException, a=1)

        class DerivedException(BaseException):
            # [CLASS DOC] HELPER: BaseException subclass taking a fancy_arg keyword in its
            # constructor.
            def __init__(self, fancy_arg):
                # [FUNCTION DOC] HELPER: Calls the base constructor with no args and stores
                # fancy_arg.
                BaseException.__init__(self)
                self.fancy_arg = fancy_arg

        x = DerivedException(fancy_arg=42)
        self.assertEqual(x.fancy_arg, 42)

    @no_tracing
    def testInfiniteRecursion(self):
        # [FUNCTION DOC] Unbounded recursion (also through a try/except that does not match)
        # must raise RecursionError. Runs with tracing disabled.
        def f():
            # [FUNCTION DOC] HELPER: Recurses forever by calling itself.
            return f()
        self.assertRaises(RecursionError, f)

        def g():
            # [FUNCTION DOC] HELPER: Recurses forever inside a try that catches only
            # ValueError.
            try:
                return g()
            except ValueError:
                return -1
        self.assertRaises(RecursionError, g)

    def test_str(self):
        # [FUNCTION DOC] Both exception classes and instances (with one or two args) have a
        # non-empty string form.
        # Make sure both instances and classes have a str representation.
        self.assertTrue(str(Exception))
        self.assertTrue(str(Exception('a')))
        self.assertTrue(str(Exception('a', 'b')))

    def test_exception_cleanup_names(self):
        # [FUNCTION DOC] The name bound by "except ... as e" is removed after the handler, so
        # e is not in locals and reading it raises UnboundLocalError.
        # Make sure the local variable bound to the exception instance by
        # an "except" statement is only visible inside the except block.
        try:
            raise Exception()
        except Exception as e:
            self.assertIsInstance(e, Exception)
        self.assertNotIn('e', locals())
        with self.assertRaises(UnboundLocalError):
            e

    def test_exception_cleanup_names2(self):
        # [FUNCTION DOC] Same cleanup as above but the name is explicitly deleted inside the
        # handler, which must not break the implicit cleanup.
        # Make sure the cleanup doesn't break if the variable is explicitly deleted.
        try:
            raise Exception()
        except Exception as e:
            self.assertIsInstance(e, Exception)
            del e
        self.assertNotIn('e', locals())
        with self.assertRaises(UnboundLocalError):
            e

    def testExceptionCleanupState(self):
        # [FUNCTION DOC] Verifies that references held by an exception (value and traceback)
        # are released when the handler ends, by checking via weakref that the referenced
        # object dies. Covers except with and without "as", bare except, break out of handler,
        # handler raising another exception, a re-raise inside finally, and an exception-
        # swallowing context manager.
        # Make sure exception state is cleaned up as soon as the except
        # block is left. See #2507

        class MyException(Exception):
            # [CLASS DOC] HELPER: Exception that keeps a reference to obj in attribute obj.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: Stores obj on the instance.
                self.obj = obj
        class MyObj:
            # [CLASS DOC] HELPER: Empty class; instances are tracked with weak references.
            pass

        def inner_raising_func():
            # [FUNCTION DOC] HELPER: Reads the enclosing variable obj into a local and raises
            # MyException carrying obj, so both the exception and traceback reference it.
            # Create some references in exception value and traceback
            local_ref = obj
            raise MyException(obj)

        # Qualified "except" with "as"
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except MyException as e:
            pass
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # Qualified "except" without "as"
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except MyException:
            pass
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # Bare "except"
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except:
            pass
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # "except" with premature block leave
        obj = MyObj()
        wr = weakref.ref(obj)
        for i in [0]:
            try:
                inner_raising_func()
            except:
                break
        obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

        # "except" block raising another exception
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            try:
                inner_raising_func()
            except:
                raise KeyError
        except KeyError as e:
            # We want to test that the except block above got rid of
            # the exception raised in inner_raising_func(), but it
            # also ends up in the __context__ of the KeyError, so we
            # must clear the latter manually for our test to succeed.
            e.__context__ = None
            obj = None
            gc_collect()  # For PyPy or other GCs.
            obj = wr()
            # guarantee no ref cycles on CPython (don't gc_collect)
            if check_impl_detail(cpython=False):
                gc_collect()
            self.assertIsNone(obj)

        # Some complicated construct
        obj = MyObj()
        wr = weakref.ref(obj)
        try:
            inner_raising_func()
        except MyException:
            try:
                try:
                    raise
                finally:
                    raise
            except MyException:
                pass
        obj = None
        if check_impl_detail(cpython=False):
            gc_collect()
        obj = wr()
        self.assertIsNone(obj)

        # Inside an exception-silencing "with" block
        class Context:
            # [CLASS DOC] HELPER: Context manager whose exit swallows every exception.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __exit__ (self, exc_type, exc_value, exc_tb):
                # [FUNCTION DOC] HELPER: Returns True, suppressing any exception.
                return True
        obj = MyObj()
        wr = weakref.ref(obj)
        with Context():
            inner_raising_func()
        obj = None
        if check_impl_detail(cpython=False):
            gc_collect()
        obj = wr()
        self.assertIsNone(obj)

    def test_exception_target_in_nested_scope(self):
        # [FUNCTION DOC] Regression test (issue 4617): the exception target name used in a
        # nested function must compile and run; the NameError for an undefined name is caught.
        # issue 4617: This used to raise a SyntaxError
        # "can not delete variable 'e' referenced in nested scope"
        def print_error():
            # [FUNCTION DOC] HELPER: Reads the enclosing variable e (no result is used).
            e
        try:
            something
        except Exception as e:
            print_error()
            # implicit "del e" here

    def test_generator_leaking(self):
        # [FUNCTION DOC] A generator's handled exception must not leak into the caller's
        # sys.exception, both outside and inside an exception handler of the caller.
        # Test that generator exception state doesn't leak into the calling
        # frame
        def yield_raise():
            # [FUNCTION DOC] HELPER: Generator that yields sys.exception inside a handler
            # twice, then once more after the handler ends (None).
            try:
                raise KeyError("caught")
            except KeyError:
                yield sys.exception()
                yield sys.exception()
            yield sys.exception()
        g = yield_raise()
        self.assertIsInstance(next(g), KeyError)
        self.assertIsNone(sys.exception())
        self.assertIsInstance(next(g), KeyError)
        self.assertIsNone(sys.exception())
        self.assertIsNone(next(g))

        # Same test, but inside an exception handler
        try:
            raise TypeError("foo")
        except TypeError:
            g = yield_raise()
            self.assertIsInstance(next(g), KeyError)
            self.assertIsInstance(sys.exception(), TypeError)
            self.assertIsInstance(next(g), KeyError)
            self.assertIsInstance(sys.exception(), TypeError)
            self.assertIsInstance(next(g), TypeError)
            del g
            self.assertIsInstance(sys.exception(), TypeError)

    def test_generator_leaking2(self):
        # [FUNCTION DOC] Issue 12475: finishing a generator started inside a handler must
        # leave the caller's sys.exception as None.
        # See issue 12475.
        def g():
            # [FUNCTION DOC] HELPER: Generator that yields once.
            yield
        try:
            raise RuntimeError
        except RuntimeError:
            it = g()
            next(it)
        try:
            next(it)
        except StopIteration:
            pass
        self.assertIsNone(sys.exception())

    def test_generator_leaking3(self):
        # [FUNCTION DOC] Issue 23353: throwing into a generator saves and restores the
        # caller's exception state.
        # See issue #23353.  When gen.throw() is called, the caller's
        # exception state should be save and restored.
        def g():
            # [FUNCTION DOC] HELPER: Generator that yields sys.exception after catching a
            # thrown ZeroDivisionError.
            try:
                yield
            except ZeroDivisionError:
                yield sys.exception()
        it = g()
        next(it)
        try:
            1/0
        except ZeroDivisionError as e:
            self.assertIs(sys.exception(), e)
            gen_exc = it.throw(e)
            self.assertIs(sys.exception(), e)
            self.assertIs(gen_exc, e)
        self.assertIsNone(sys.exception())

    def test_generator_leaking4(self):
        # [FUNCTION DOC] Issue 23353: when a generator raises, the caller's exception state is
        # still restored properly (not left as the caller's earlier TypeError).
        # See issue #23353.  When an exception is raised by a generator,
        # the caller's exception state should still be restored.
        def g():
            # [FUNCTION DOC] HELPER: Generator that catches ZeroDivisionError, yields
            # sys.exception, then re-raises.
            try:
                1/0
            except ZeroDivisionError:
                yield sys.exception()
                raise
        it = g()
        try:
            raise TypeError
        except TypeError:
            # The caller's exception state (TypeError) is temporarily
            # saved in the generator.
            tp = type(next(it))
        self.assertIs(tp, ZeroDivisionError)
        try:
            next(it)
            # We can't check it immediately, but while next() returns
            # with an exception, it shouldn't have restored the old
            # exception state (TypeError).
        except ZeroDivisionError as e:
            self.assertIs(sys.exception(), e)
        # We used to find TypeError here.
        self.assertIsNone(sys.exception())

    def test_generator_doesnt_retain_old_exc(self):
        # [FUNCTION DOC] Generator must not retain the exception that was active when it was
        # created.
        def g():
            # [FUNCTION DOC] HELPER: Generator that asserts the active exception is
            # RuntimeError when resumed in the first step and None afterwards.
            self.assertIsInstance(sys.exception(), RuntimeError)
            yield
            self.assertIsNone(sys.exception())
        it = g()
        try:
            raise RuntimeError
        except RuntimeError:
            next(it)
        self.assertRaises(StopIteration, next, it)

    def test_generator_finalizing_and_sys_exception(self):
        # [FUNCTION DOC] Issue 7173: finalizing a generator after it ran inside an exception
        # handler must not leave sys.exception set.
        # See #7173
        def simple_gen():
            # [FUNCTION DOC] HELPER: Generator yielding the single value 1.
            yield 1
        def run_gen():
            # [FUNCTION DOC] HELPER: Starts simple_gen inside a handler of a raised
            # RuntimeError and returns its first value.
            gen = simple_gen()
            try:
                raise RuntimeError
            except RuntimeError:
                return next(gen)
        run_gen()
        gc_collect()
        self.assertIsNone(sys.exception())

    def _check_generator_cleanup_exc_state(self, testfunc):
        # [FUNCTION DOC] HELPER: Issue 12791 check. Starts a generator that holds an exception
        # referencing a tracked object, applies testfunc to it (throw, close, delete, next or
        # send), then drops all references and asserts through a weakref that the object was
        # freed.
        # Issue #12791: exception state is cleaned up as soon as a generator
        # is closed (reference cycles are broken).
        class MyException(Exception):
            # [CLASS DOC] HELPER: Exception that keeps a reference to obj.
            def __init__(self, obj):
                # [FUNCTION DOC] HELPER: Stores obj on the instance.
                self.obj = obj
        class MyObj:
            # [CLASS DOC] HELPER: Empty class whose instance is tracked by a weak reference.
            pass

        def raising_gen():
            # [FUNCTION DOC] HELPER: Generator that raises and catches MyException(obj) and
            # yields once while the handler is active.
            try:
                raise MyException(obj)
            except MyException:
                yield

        obj = MyObj()
        wr = weakref.ref(obj)
        g = raising_gen()
        next(g)
        testfunc(g)
        g = obj = None
        gc_collect()  # For PyPy or other GCs.
        obj = wr()
        self.assertIsNone(obj)

    def test_generator_throw_cleanup_exc_state(self):
        # [FUNCTION DOC] Cleanup check where the generator is finished by throw().
        def do_throw(g):
            # [FUNCTION DOC] HELPER: Throws RuntimeError into g and swallows it.
            try:
                g.throw(RuntimeError())
            except RuntimeError:
                pass
        self._check_generator_cleanup_exc_state(do_throw)

    def test_generator_close_cleanup_exc_state(self):
        # [FUNCTION DOC] Cleanup check where the generator is finished by close().
        def do_close(g):
            # [FUNCTION DOC] HELPER: Closes generator g.
            g.close()
        self._check_generator_cleanup_exc_state(do_close)

    def test_generator_del_cleanup_exc_state(self):
        # [FUNCTION DOC] Cleanup check where the generator is finished by dropping the last
        # reference.
        def do_del(g):
            # [FUNCTION DOC] HELPER: Rebinds the local name only; the caller releases the
            # generator.
            g = None
        self._check_generator_cleanup_exc_state(do_del)

    def test_generator_next_cleanup_exc_state(self):
        # [FUNCTION DOC] Cleanup check where the generator is finished by next().
        def do_next(g):
            # [FUNCTION DOC] HELPER: Calls next on g and expects StopIteration, failing the
            # test otherwise.
            try:
                next(g)
            except StopIteration:
                pass
            else:
                self.fail("should have raised StopIteration")
        self._check_generator_cleanup_exc_state(do_next)

    def test_generator_send_cleanup_exc_state(self):
        # [FUNCTION DOC] Cleanup check where the generator is finished by send(None).
        def do_send(g):
            # [FUNCTION DOC] HELPER: Sends None to g and expects StopIteration, failing
            # otherwise.
            try:
                g.send(None)
            except StopIteration:
                pass
            else:
                self.fail("should have raised StopIteration")
        self._check_generator_cleanup_exc_state(do_send)

    def test_3114(self):
        # [FUNCTION DOC] Bug 3114: an object whose finalizer reads sys.exception must see no
        # dangling exception; after the exception holding it is dropped the finalizer must
        # have recorded None.
        # Bug #3114: in its destructor, MyObject retrieves a pointer to
        # obsolete and/or deallocated objects.
        class MyObject:
            # [CLASS DOC] HELPER: Object whose finalizer stores the current sys.exception into
            # the enclosing variable e.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Sets the enclosing variable e to the current handled
                # exception.
                nonlocal e
                e = sys.exception()
        e = ()
        try:
            raise Exception(MyObject())
        except:
            pass
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(e)

    def test_raise_does_not_create_context_chain_cycle(self):
        # [FUNCTION DOC] Raising an exception whose context chain already contains the active
        # one must not create a cycle; builds a context chain of three exceptions, re-raises
        # the oldest and checks the resulting chain is reordered without a loop.
        class A(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass A.
            pass
        class B(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass B.
            pass
        class C(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass C.
            pass

        # Create a context chain:
        # C -> B -> A
        # Then raise A in context of C.
        try:
            try:
                raise A
            except A as a_:
                a = a_
                try:
                    raise B
                except B as b_:
                    b = b_
                    try:
                        raise C
                    except C as c_:
                        c = c_
                        self.assertIsInstance(a, A)
                        self.assertIsInstance(b, B)
                        self.assertIsInstance(c, C)
                        self.assertIsNone(a.__context__)
                        self.assertIs(b.__context__, a)
                        self.assertIs(c.__context__, b)
                        raise a
        except A as e:
            exc = e

        # Expect A -> C -> B, without cycle
        self.assertIs(exc, a)
        self.assertIs(a.__context__, c)
        self.assertIs(c.__context__, b)
        self.assertIsNone(b.__context__)

    def test_no_hang_on_context_chain_cycle1(self):
        # [FUNCTION DOC] Issue 25782: an exception whose __context__ is itself must not make
        # raising hang; the new exception's context is the self-referencing one.
        # See issue 25782. Cycle in context chain.

        def cycle():
            # [FUNCTION DOC] HELPER: Raises ValueError, makes it its own context, then raises
            # TypeError.
            try:
                raise ValueError(1)
            except ValueError as ex:
                ex.__context__ = ex
                raise TypeError(2)

        try:
            cycle()
        except Exception as e:
            exc = e

        self.assertIsInstance(exc, TypeError)
        self.assertIsInstance(exc.__context__, ValueError)
        self.assertIs(exc.__context__.__context__, exc.__context__)

    def test_no_hang_on_context_chain_cycle2(self):
        # [FUNCTION DOC] Issue 25782: a context chain with a cycle at its head must not hang
        # when the last exception is raised; the cyclic links are preserved.
        # See issue 25782. Cycle at head of context chain.

        class A(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass A.
            pass
        class B(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass B.
            pass
        class C(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass C.
            pass

        # Context cycle:
        # +-----------+
        # V           |
        # C --> B --> A
        with self.assertRaises(C) as cm:
            try:
                raise A()
            except A as _a:
                a = _a
                try:
                    raise B()
                except B as _b:
                    b = _b
                    try:
                        raise C()
                    except C as _c:
                        c = _c
                        a.__context__ = c
                        raise c

        self.assertIs(cm.exception, c)
        # Verify the expected context chain cycle
        self.assertIs(c.__context__, b)
        self.assertIs(b.__context__, a)
        self.assertIs(a.__context__, c)

    def test_no_hang_on_context_chain_cycle3(self):
        # [FUNCTION DOC] Issue 25782: a longer context chain (five exceptions) containing a
        # cycle must not hang; the links are checked afterwards.
        # See issue 25782. Longer context chain with cycle.

        class A(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass A.
            pass
        class B(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass B.
            pass
        class C(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass C.
            pass
        class D(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass D.
            pass
        class E(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass E.
            pass

        # Context cycle:
        #             +-----------+
        #             V           |
        # E --> D --> C --> B --> A
        with self.assertRaises(E) as cm:
            try:
                raise A()
            except A as _a:
                a = _a
                try:
                    raise B()
                except B as _b:
                    b = _b
                    try:
                        raise C()
                    except C as _c:
                        c = _c
                        a.__context__ = c
                        try:
                            raise D()
                        except D as _d:
                            d = _d
                            e = E()
                            raise e

        self.assertIs(cm.exception, e)
        # Verify the expected context chain cycle
        self.assertIs(e.__context__, d)
        self.assertIs(d.__context__, c)
        self.assertIs(c.__context__, b)
        self.assertIs(b.__context__, a)
        self.assertIs(a.__context__, c)

    def test_context_of_exception_in_try_and_finally(self):
        # [FUNCTION DOC] An exception raised in a finally block gets the pending exception
        # from the try body as its __context__.
        try:
            try:
                te = TypeError(1)
                raise te
            finally:
                ve = ValueError(2)
                raise ve
        except Exception as e:
            exc = e

        self.assertIs(exc, ve)
        self.assertIs(exc.__context__, te)

    def test_context_of_exception_in_except_and_finally(self):
        # [FUNCTION DOC] Exception raised in finally after one raised in an except block: the
        # chain is finally exception -> except exception -> original.
        try:
            try:
                te = TypeError(1)
                raise te
            except:
                ve = ValueError(2)
                raise ve
            finally:
                oe = OSError(3)
                raise oe
        except Exception as e:
            exc = e

        self.assertIs(exc, oe)
        self.assertIs(exc.__context__, ve)
        self.assertIs(exc.__context__.__context__, te)

    def test_context_of_exception_in_else_and_finally(self):
        # [FUNCTION DOC] Exception raised in finally after one raised in the else block:
        # finally exception has the else exception as context.
        try:
            try:
                pass
            except:
                pass
            else:
                ve = ValueError(1)
                raise ve
            finally:
                oe = OSError(2)
                raise oe
        except Exception as e:
            exc = e

        self.assertIs(exc, oe)
        self.assertIs(exc.__context__, ve)

    def test_unicode_change_attributes(self):
        # [FUNCTION DOC] Issue 7309 regression: mutating start, end, reason and encoding on
        # UnicodeEncodeError, UnicodeDecodeError and UnicodeTranslateError (including a huge
        # int reason and a start beyond end) must not crash, and str output follows the new
        # attributes.
        # See issue 7309. This was a crasher.

        u = UnicodeEncodeError('baz', 'xxxxx', 1, 5, 'foo')
        self.assertEqual(str(u), "'baz' codec can't encode characters in position 1-4: foo")
        u.end = 2
        self.assertEqual(str(u), "'baz' codec can't encode character '\\x78' in position 1: foo")
        u.end = 5
        u.reason = 0x345345345345345345
        self.assertEqual(str(u), "'baz' codec can't encode characters in position 1-4: 965230951443685724997")
        u.encoding = 4000
        self.assertEqual(str(u), "'4000' codec can't encode characters in position 1-4: 965230951443685724997")
        u.start = 1000
        self.assertEqual(str(u), "'4000' codec can't encode characters in position 1000-4: 965230951443685724997")

        u = UnicodeDecodeError('baz', b'xxxxx', 1, 5, 'foo')
        self.assertEqual(str(u), "'baz' codec can't decode bytes in position 1-4: foo")
        u.end = 2
        self.assertEqual(str(u), "'baz' codec can't decode byte 0x78 in position 1: foo")
        u.end = 5
        u.reason = 0x345345345345345345
        self.assertEqual(str(u), "'baz' codec can't decode bytes in position 1-4: 965230951443685724997")
        u.encoding = 4000
        self.assertEqual(str(u), "'4000' codec can't decode bytes in position 1-4: 965230951443685724997")
        u.start = 1000
        self.assertEqual(str(u), "'4000' codec can't decode bytes in position 1000-4: 965230951443685724997")

        u = UnicodeTranslateError('xxxx', 1, 5, 'foo')
        self.assertEqual(str(u), "can't translate characters in position 1-4: foo")
        u.end = 2
        self.assertEqual(str(u), "can't translate character '\\x78' in position 1: foo")
        u.end = 5
        u.reason = 0x345345345345345345
        self.assertEqual(str(u), "can't translate characters in position 1-4: 965230951443685724997")
        u.start = 1000
        self.assertEqual(str(u), "can't translate characters in position 1000-4: 965230951443685724997")

    def test_unicode_errors_no_object(self):
        # [FUNCTION DOC] Issue 21134: Unicode error instances created through __new__ without
        # initialization stringify to an empty string.
        # See issue #21134.
        klasses = UnicodeEncodeError, UnicodeDecodeError, UnicodeTranslateError
        for klass in klasses:
            self.assertEqual(str(klass.__new__(klass)), "")

    @skipIfTorchDynamo("slow test")
    def test_unicode_error_str_does_not_crash(self):
        # [FUNCTION DOC] Loops over many start/end/object-length combinations and checks that
        # converting Unicode encode, decode and translate errors to str never crashes and
        # always returns a string. Skipped under Dynamo as slow.
        # Test that str(UnicodeError(...)) does not crash.
        # See https://github.com/python/cpython/issues/123378.

        for start, end, objlen in product(
            range(-5, 5),
            range(-5, 5),
            range(7),
        ):
            obj = 'a' * objlen
            with self.subTest('encode', objlen=objlen, start=start, end=end):
                exc = UnicodeEncodeError('utf-8', obj, start, end, '')
                self.assertIsInstance(str(exc), str)

            with self.subTest('translate', objlen=objlen, start=start, end=end):
                exc = UnicodeTranslateError(obj, start, end, '')
                self.assertIsInstance(str(exc), str)

            encoded = obj.encode()
            with self.subTest('decode', objlen=objlen, start=start, end=end):
                exc = UnicodeDecodeError('utf-8', encoded, start, end, '')
                self.assertIsInstance(str(exc), str)

    @no_tracing
    def test_badisinstance(self):
        # [FUNCTION DOC] Bug 2542: when the subclass check on an except clause class raises,
        # that error is ignored and the real KeyError handler still runs. Also checks
        # unbounded recursion produces a RecursionError mentioning "maximum recursion depth
        # exceeded". Tracing is disabled and stderr is captured.
        # Bug #2542: if issubclass(e, MyException) raises an exception,
        # it should be ignored
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose subclass check always raises ValueError.
            def __subclasscheck__(cls, subclass):
                # [FUNCTION DOC] HELPER: Always raises ValueError.
                raise ValueError()
        class MyException(Exception, metaclass=Meta):
            # [CLASS DOC] HELPER: Exception class using the failing metaclass Meta.
            pass

        with captured_stderr() as stderr:
            try:
                raise KeyError()
            except MyException as e:
                self.fail("exception should not be a MyException")
            except KeyError:
                pass
            except:
                self.fail("Should have raised KeyError")
            else:
                self.fail("Should have raised KeyError")

        def g():
            # [FUNCTION DOC] HELPER: Recurses infinitely and returns the RecursionError it
            # catches.
            try:
                return g()
            except RecursionError as e:
                return e
        exc = g()
        self.assertIsInstance(exc, RecursionError, type(exc))
        self.assertIn("maximum recursion depth exceeded", str(exc))


    @cpython_only
    @support.requires_resource('cpu')
    def test_trashcan_recursion(self):
        # [FUNCTION DOC] Bpo-33930: builds a chain of one million bound method objects and
        # frees it to exercise deep destruction (trashcan) without crashing. CPython only; CPU
        # resource.
        # See bpo-33930

        def foo():
            # [FUNCTION DOC] HELPER: Builds the long chain of method objects, then drops it on
            # return.
            o = object()
            for x in range(1_000_000):
                # Create a big chain of method objects that will trigger
                # a deep chain of calls when they need to be destructed.
                o = o.__dir__

        foo()
        support.gc_collect()

    @cpython_only
    def test_recursion_normalizing_exception(self):
        # [FUNCTION DOC] Issue 22898: runs a subprocess script that throws into a generator at
        # the recursion limit; expects exit code 1, RecursionError and ResourceWarning on
        # stderr and "Done." on stdout (no abort). CPython only; needs _testinternalcapi.
        import_module("_testinternalcapi")
        # Issue #22898.
        # Test that a RecursionError is raised when tstate->recursion_depth is
        # equal to recursion_limit in PyErr_NormalizeException() and check
        # that a ResourceWarning is printed.
        # Prior to #22898, the recursivity of PyErr_NormalizeException() was
        # controlled by tstate->recursion_depth and a PyExc_RecursionErrorInst
        # singleton was being used in that case, that held traceback data and
        # locals indefinitely and would cause a segfault in _PyExc_Fini() upon
        # finalization of these locals.
        code = """if 1:
            import sys
            from _testinternalcapi import get_recursion_depth
            from test import support

            class MyException(Exception): pass

            def setrecursionlimit(depth):
                while 1:
                    try:
                        sys.setrecursionlimit(depth)
                        return depth
                    except RecursionError:
                        # sys.setrecursionlimit() raises a RecursionError if
                        # the new recursion limit is too low (issue #25274).
                        depth += 1

            def recurse(cnt):
                cnt -= 1
                if cnt:
                    recurse(cnt)
                else:
                    generator.throw(MyException)

            def gen():
                f = open(%a, mode='rb', buffering=0)
                yield

            generator = gen()
            next(generator)
            recursionlimit = sys.getrecursionlimit()
            try:
                recurse(support.exceeds_recursion_limit())
            finally:
                sys.setrecursionlimit(recursionlimit)
                print('Done.')
        """ % __file__
        rc, out, err = script_helper.assert_python_failure("-Wd", "-c", code)
        # Check that the program does not fail with SIGABRT.
        self.assertEqual(rc, 1)
        self.assertIn(b'RecursionError', err)
        self.assertIn(b'ResourceWarning', err)
        self.assertIn(b'Done.', out)

    @cpython_only
    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    @force_not_colorized
    def test_recursion_normalizing_infinite_exception(self):
        # [FUNCTION DOC] Issue 30697: runs a subprocess raising a C-level infinitely recursing
        # exception; expects exit code 1, the RecursionError message on stderr and "Done." on
        # stdout. Needs _testcapi; CPython only; colorized output disabled.
        # Issue #30697. Test that a RecursionError is raised when
        # maximum recursion depth has been exceeded when creating
        # an exception
        code = """if 1:
            import _testcapi
            try:
                raise _testcapi.RecursingInfinitelyError
            finally:
                print('Done.')
        """
        rc, out, err = script_helper.assert_python_failure("-c", code)
        self.assertEqual(rc, 1)
        expected = b'RecursionError: maximum recursion depth exceeded'
        self.assertTrue(expected in err, msg=f"{expected!r} not found in {err[:3_000]!r}... (truncated)")
        self.assertIn(b'Done.', out)


    def test_recursion_in_except_handler(self):
        # [FUNCTION DOC] Checks that deep recursion from within, after, or around an except
        # handler raises RecursionError (with a recursion limit set 10 above the current
        # depth) and that the original recursion limit is always restored.

        def set_relative_recursion_limit(n):
            # [FUNCTION DOC] HELPER: Finds the smallest accepted recursion limit by raising it
            # until sys.setrecursionlimit stops failing, then sets the limit n above that.
            depth = 1
            while True:
                try:
                    sys.setrecursionlimit(depth)
                except RecursionError:
                    depth += 1
                else:
                    break
            sys.setrecursionlimit(depth+n)

        def recurse_in_except():
            # [FUNCTION DOC] HELPER: Recurses infinitely from inside an except handler.
            try:
                1/0
            except:
                recurse_in_except()

        def recurse_after_except():
            # [FUNCTION DOC] HELPER: Handles an error, then recurses infinitely after the
            # handler.
            try:
                1/0
            except:
                pass
            recurse_after_except()

        def recurse_in_body_and_except():
            # [FUNCTION DOC] HELPER: Recurses infinitely both in the try body and in its
            # handler.
            try:
                recurse_in_body_and_except()
            except:
                recurse_in_body_and_except()

        recursionlimit = sys.getrecursionlimit()
        for func in (recurse_in_except, recurse_after_except, recurse_in_body_and_except):
            with self.subTest(func=func):
                try:
                    set_relative_recursion_limit(10)
                    try:
                        func()
                    except RecursionError:
                        pass
                    else:
                        self.fail("Should have raised a RecursionError")
                finally:
                    sys.setrecursionlimit(recursionlimit)


    @cpython_only
    # Python built with Py_TRACE_REFS fail with a fatal error in
    # _PyRefchain_Trace() on memory allocation error.
    @unittest.skipIf(support.Py_TRACE_REFS, 'cannot test Py_TRACE_REFS build')
    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    def test_recursion_normalizing_with_no_memory(self):
        # [FUNCTION DOC] Issue 30697: subprocess run with simulated memory exhaustion at deep
        # recursion; expects failure with MemoryError mentioned in stderr. CPython only;
        # skipped on reference-tracing builds or without _testcapi.
        # Issue #30697. Test that in the abort that occurs when there is no
        # memory left and the size of the Python frames stack is greater than
        # the size of the list of preallocated MemoryError instances, the
        # Fatal Python error message mentions MemoryError.
        code = """if 1:
            import _testcapi
            class C(): pass
            def recurse(cnt):
                cnt -= 1
                if cnt:
                    recurse(cnt)
                else:
                    _testcapi.set_nomemory(0)
                    C()
            recurse(16)
        """
        with SuppressCrashReport():
            rc, out, err = script_helper.assert_python_failure("-c", code)
            self.assertIn(b'MemoryError', err)

    @cpython_only
    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    def test_MemoryError(self):
        # [FUNCTION DOC] The shared preallocated MemoryError raised by C code must not
        # accumulate duplicate traceback entries: two raises give identical formatted
        # tracebacks.
        # PyErr_NoMemory always raises the same exception instance.
        # Check that the traceback is not doubled.
        import traceback
        from _testcapi import raise_memoryerror
        def raiseMemError():
            # [FUNCTION DOC] HELPER: Triggers a C-level MemoryError and returns the formatted
            # traceback.
            try:
                raise_memoryerror()
            except MemoryError as e:
                tb = e.__traceback__
            else:
                self.fail("Should have raised a MemoryError")
            return traceback.format_tb(tb)

        tb1 = raiseMemError()
        tb2 = raiseMemError()
        self.assertEqual(tb1, tb2)

    @cpython_only
    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    def test_exception_with_doc(self):
        # [FUNCTION DOC] Checks _testcapi.make_exception_with_doc: an undotted name raises
        # SystemError; the result is a type subclassing Exception; doc, base class (single or
        # tuple) and attribute dictionary arguments are honored.
        doc2 = "This is a test docstring."
        doc4 = "This is another test docstring."

        self.assertRaises(SystemError, _testcapi.make_exception_with_doc,
                          "error1")

        # test basic usage of PyErr_NewException
        error1 = _testcapi.make_exception_with_doc("_testcapi.error1")
        self.assertIs(type(error1), type)
        self.assertTrue(issubclass(error1, Exception))
        self.assertIsNone(error1.__doc__)

        # test with given docstring
        error2 = _testcapi.make_exception_with_doc("_testcapi.error2", doc2)
        self.assertEqual(error2.__doc__, doc2)

        # test with explicit base (without docstring)
        error3 = _testcapi.make_exception_with_doc("_testcapi.error3",
                                                   base=error2)
        self.assertTrue(issubclass(error3, error2))

        # test with explicit base tuple
        class C(object):
            # [CLASS DOC] HELPER: Plain object subclass used as an extra base class.
            pass
        error4 = _testcapi.make_exception_with_doc("_testcapi.error4", doc4,
                                                   (error3, C))
        self.assertTrue(issubclass(error4, error3))
        self.assertTrue(issubclass(error4, C))
        self.assertEqual(error4.__doc__, doc4)

        # test with explicit dictionary
        error5 = _testcapi.make_exception_with_doc("_testcapi.error5", "",
                                                   error4, {'a': 1})
        self.assertTrue(issubclass(error5, error4))
        self.assertEqual(error5.a, 1)
        self.assertEqual(error5.__doc__, "")

    @cpython_only
    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    def test_memory_error_cleanup(self):
        # [FUNCTION DOC] Issue 5437: the preallocated MemoryError must not keep traceback
        # objects alive; a weakly referenced local in the failing function is freed after the
        # handler ends.
        # Issue #5437: preallocated MemoryError instances should not keep
        # traceback objects alive.
        from _testcapi import raise_memoryerror
        class C:
            # [CLASS DOC] HELPER: Empty class whose instance is tracked by a weak reference.
            pass
        wr = None
        def inner():
            # [FUNCTION DOC] HELPER: Creates a tracked object, stores a weak reference to it
            # in the enclosing variable wr, then triggers a MemoryError from C.
            nonlocal wr
            c = C()
            wr = weakref.ref(c)
            raise_memoryerror()
        # We cannot use assertRaises since it manually deletes the traceback
        try:
            inner()
        except MemoryError as e:
            self.assertNotEqual(wr(), None)
        else:
            self.fail("MemoryError not raised")
        gc_collect()  # For PyPy or other GCs.
        self.assertEqual(wr(), None)

    @no_tracing
    def test_recursion_error_cleanup(self):
        # [FUNCTION DOC] Same cleanup check as for MemoryError but with RecursionError from
        # unbounded recursion; tracing disabled.
        # Same test as above, but with "recursion exceeded" errors
        class C:
            # [CLASS DOC] HELPER: Empty class whose instance is tracked by a weak reference.
            pass
        wr = None
        def inner():
            # [FUNCTION DOC] HELPER: Records a weak reference to a new object in wr and
            # recurses forever.
            nonlocal wr
            c = C()
            wr = weakref.ref(c)
            inner()
        # We cannot use assertRaises since it manually deletes the traceback
        try:
            inner()
        except RecursionError as e:
            self.assertNotEqual(wr(), None)
        else:
            self.fail("RecursionError not raised")
        gc_collect()  # For PyPy or other GCs.
        self.assertEqual(wr(), None)

    def test_errno_ENOTDIR(self):
        # [FUNCTION DOC] Issue 12802: listing a regular file as a directory raises OSError
        # with errno ENOTDIR.
        # Issue #12802: "not a directory" errors are ENOTDIR even on Windows
        with self.assertRaises(OSError) as cm:
            os.listdir(__file__)
        self.assertEqual(cm.exception.errno, errno.ENOTDIR, cm.exception)

    def test_unraisable(self):
        # [FUNCTION DOC] Issue 22836: an exception raised in __del__ is reported as unraisable
        # with the __del__ function as the object and a traceback attached.
        # Issue #22836: PyErr_WriteUnraisable() should give sensible reports
        class BrokenDel:
            # [CLASS DOC] HELPER: Class whose finalizer always raises ValueError.
            def __del__(self):
                # [FUNCTION DOC] HELPER: Raises ValueError("del is broken").
                exc = ValueError("del is broken")
                # The following line is included in the traceback report:
                raise exc

        obj = BrokenDel()
        with support.catch_unraisable_exception() as cm:
            del obj

            gc_collect()  # For PyPy or other GCs.
            self.assertEqual(cm.unraisable.object, BrokenDel.__del__)
            self.assertIsNotNone(cm.unraisable.exc_traceback)

    def test_unhandled(self):
        # [FUNCTION DOC] Checks the report from the default excepthook for ValueError and for
        # an exception whose str fails: it contains file name, source line, type name, message
        # (or the "<exception str() failed>" marker) and ends with a newline. stderr is
        # captured.
        # Check for sensible reporting of unhandled exceptions
        for exc_type in (ValueError, BrokenStrException):
            with self.subTest(exc_type):
                try:
                    exc = exc_type("test message")
                    # The following line is included in the traceback report:
                    raise exc
                except exc_type:
                    with captured_stderr() as stderr:
                        sys.__excepthook__(*sys.exc_info())
                report = stderr.getvalue()
                self.assertIn("test_exceptions.py", report)
                self.assertIn("raise exc", report)
                self.assertIn(exc_type.__name__, report)
                if exc_type is BrokenStrException:
                    self.assertIn("<exception str() failed>", report)
                else:
                    self.assertIn("test message", report)
                self.assertTrue(report.endswith("\n"))

    @cpython_only
    # Python built with Py_TRACE_REFS fail with a fatal error in
    # _PyRefchain_Trace() on memory allocation error.
    @unittest.skipIf(support.Py_TRACE_REFS, 'cannot test Py_TRACE_REFS build')
    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    def test_memory_error_in_PyErr_PrintEx(self):
        # [FUNCTION DOC] Issue 30817: for allocation-failure points 1 to 19 runs a subprocess
        # that simulates memory exhaustion and expects exit code 1 or 120 and MemoryError in
        # stderr. CPython only; needs _testcapi; skipped on reference-tracing builds.
        code = """if 1:
            import _testcapi
            class C(): pass
            _testcapi.set_nomemory(0, %d)
            C()
        """

        # Issue #30817: Abort in PyErr_PrintEx() when no memory.
        # Span a large range of tests as the CPython code always evolves with
        # changes that add or remove memory allocations.
        for i in range(1, 20):
            rc, out, err = script_helper.assert_python_failure("-c", code % i)
            self.assertIn(rc, (1, 120))
            self.assertIn(b'MemoryError', err)

    def test_yield_in_nested_try_excepts(self):
        # [FUNCTION DOC] Issue 25612: a coroutine yielding inside a nested handler, then
        # thrown an unrelated exception, must re-raise the original MainError via a bare
        # raise.
        #Issue #25612
        class MainError(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass raised first.
            pass

        class SubError(Exception):
            # [CLASS DOC] HELPER: Empty Exception subclass thrown into the generator.
            pass

        def main():
            # [FUNCTION DOC] HELPER: Generator that raises MainError, yields inside the
            # handler within an inner try catching SubError, then re-raises with a bare raise.
            try:
                raise MainError()
            except MainError:
                try:
                    yield
                except SubError:
                    pass
                raise

        coro = main()
        coro.send(None)
        with self.assertRaises(MainError):
            coro.throw(SubError())

    def test_generator_doesnt_retain_old_exc2(self):
        # [FUNCTION DOC] Issue 28884: a generator must not retain the caller's active
        # exception when resumed after yielding from its own handler (sys.exception is None).
        #Issue 28884#msg282532
        def g():
            # [FUNCTION DOC] HELPER: Generator yielding from within a handler, then asserting
            # no active exception and yielding again.
            try:
                raise ValueError
            except ValueError:
                yield 1
            self.assertIsNone(sys.exception())
            yield 2

        gen = g()

        try:
            raise IndexError
        except IndexError:
            self.assertEqual(next(gen), 1)
        self.assertEqual(next(gen), 2)

    def test_raise_in_generator(self):
        # [FUNCTION DOC] Issue 25612: a bare raise inside a generator resumed in the caller's
        # handler re-raises the caller's ZeroDivisionError.
        #Issue 25612#msg304117
        def g():
            # [FUNCTION DOC] HELPER: Generator that yields 1 then does a bare raise.
            yield 1
            raise
            yield 2

        with self.assertRaises(ZeroDivisionError):
            i = g()
            try:
                1/0
            except:
                next(i)
                next(i)

    @unittest.skipUnless(__debug__, "Won't work if __debug__ is False")
    def test_assert_shadowing(self):
        # [FUNCTION DOC] Rebinding the module global AssertionError to TypeError must not
        # change which exception an assert statement raises; the global is deleted afterwards.
        # Skipped when asserts are disabled. Side effect: temporarily creates and deletes a
        # module global.
        # Shadowing AssertionError would cause the assert statement to
        # misbehave.
        global AssertionError
        AssertionError = TypeError
        try:
            assert False, 'hello'
        except BaseException as e:
            del AssertionError
            self.assertIsInstance(e, AssertionError)
            self.assertEqual(str(e), 'hello')
        else:
            del AssertionError
            self.fail('Expected exception')

    def test_memory_error_subclasses(self):
        # [FUNCTION DOC] Bpo-41654: creating and discarding a MemoryError subclass instance
        # must not corrupt the MemoryError freelist; raises plain MemoryError repeatedly
        # afterwards with garbage collection.
        # bpo-41654: MemoryError instances use a freelist of objects that are
        # linked using the 'dict' attribute when they are inactive/dead.
        # Subclasses of MemoryError should not participate in the freelist
        # schema. This test creates a MemoryError object and keeps it alive
        # (therefore advancing the freelist) and then it creates and destroys a
        # subclass object. Finally, it checks that creating a new MemoryError
        # succeeds, proving that the freelist is not corrupted.

        class TestException(MemoryError):
            # [CLASS DOC] HELPER: Empty MemoryError subclass.
            pass

        try:
            raise MemoryError
        except MemoryError as exc:
            inst = exc

        try:
            raise TestException
        except Exception:
            pass

        for _ in range(10):
            try:
                raise MemoryError
            except MemoryError as exc:
                pass

            gc_collect()

    @unittest.skipIf(_testcapi is None, "requires _testcapi")
    def test_memory_error_in_subinterp(self):
        # [FUNCTION DOC] Gh-109894: a subprocess runs an allocation too large for a
        # subinterpreter; expects a clean exit with MemoryError reported on stderr. Needs
        # _testcapi.
        # gh-109894: subinterpreters shouldn't count on last resort memory error
        # when MemoryError is raised through PyErr_NoMemory() call,
        # and should preallocate memory errors as does the main interpreter.
        # interp.static_objects.last_resort_memory_error.args
        # should be initialized to empty tuple to avoid crash on attempt to print it.
        code = f"""if 1:
            import _testcapi
            _testcapi.run_in_subinterp(\"[0]*{sys.maxsize}\")
            exit(0)
        """
        rc, _, err = script_helper.assert_python_ok("-c", code)
        self.assertIn(b'MemoryError', err)


class NameErrorTests(__TestCase):
    # [CLASS DOC] Tests for NameError details: the name attribute, traceback reporting when
    # the failure happens inside assertRaisesRegex, and class-body self reference. Stateless.
    def test_name_error_has_name(self):
        # [FUNCTION DOC] A NameError for an undefined variable exposes the variable name in
        # the name attribute.
        try:
            bluch
        except NameError as exc:
            self.assertEqual("bluch", exc.name)

    def test_issue45826(self):
        # [FUNCTION DOC] Bpo-45826 regression: when assertRaisesRegex fails because the
        # NameError message does not match, the default excepthook output must still mention
        # the real missing name. Fails the test if the assertion helper does not fail.
        # regression test for bpo-45826
        def f():
            # [FUNCTION DOC] HELPER: Expects NameError matching "aaa" but references undefined
            # name aab.
            with self.assertRaisesRegex(NameError, 'aaa'):
                aab

        try:
            f()
        except self.failureException:
            with support.captured_stderr() as err:
                sys.__excepthook__(*sys.exc_info())
        else:
            self.fail("assertRaisesRegex should have failed.")

        self.assertIn("aab", err.getvalue())

    def test_issue45826_focused(self):
        # [FUNCTION DOC] Focused bpo-45826 case: with the traceback cleared on a caught
        # NameError and a ZeroDivisionError raised afterwards, the excepthook output still
        # contains both the missing name and the new exception type.
        def f():
            # [FUNCTION DOC] HELPER: Catches NameError for an undefined name, clears its
            # traceback and raises ZeroDivisionError.
            try:
                nonsense
            except BaseException as E:
                E.with_traceback(None)
                raise ZeroDivisionError()

        try:
            f()
        except ZeroDivisionError:
            with support.captured_stderr() as err:
                sys.__excepthook__(*sys.exc_info())

        self.assertIn("nonsense", err.getvalue())
        self.assertIn("ZeroDivisionError", err.getvalue())

    def test_gh_111654(self):
        # [FUNCTION DOC] Gh-111654: referencing the class name inside its own body raises
        # NameError.
        def f():
            # [FUNCTION DOC] HELPER: Defines a class whose body references the class name
            # before it is bound.
            class TestClass:
                # [CLASS DOC] HELPER: Class whose body references its own (not yet bound)
                # name.
                TestClass

        self.assertRaises(NameError, f)

    # Note: name suggestion tests live in `test_traceback`.


class AttributeErrorTests(__TestCase):
    # [CLASS DOC] Tests for AttributeError name and obj attributes. Stateless.
    def test_attributes(self):
        # [FUNCTION DOC] AttributeError name and obj default to None and can be set via
        # keyword arguments.
        # Setting 'attr' should not be a problem.
        exc = AttributeError('Ouch!')
        self.assertIsNone(exc.name)
        self.assertIsNone(exc.obj)

        sentinel = object()
        exc = AttributeError('Ouch', name='carry', obj=sentinel)
        self.assertEqual(exc.name, 'carry')
        self.assertIs(exc.obj, sentinel)

    def test_getattr_has_name_and_obj(self):
        # [FUNCTION DOC] A failed attribute lookup (normal access or direct object attribute
        # access) sets name to the missing attribute and obj to the instance.
        class A:
            # [CLASS DOC] HELPER: Class with a blech attribute set to None.
            blech = None

        obj = A()
        try:
            obj.bluch
        except AttributeError as exc:
            self.assertEqual("bluch", exc.name)
            self.assertEqual(obj, exc.obj)
        try:
            object.__getattribute__(obj, "bluch")
        except AttributeError as exc:
            self.assertEqual("bluch", exc.name)
            self.assertEqual(obj, exc.obj)

    def test_getattr_has_name_and_obj_for_method(self):
        # [FUNCTION DOC] Calling a missing method sets name and obj on the AttributeError as
        # well.
        class A:
            # [CLASS DOC] HELPER: Class with a blech method.
            def blech(self):
                # [FUNCTION DOC] HELPER: Method that returns nothing.
                return

        obj = A()
        try:
            obj.bluch()
        except AttributeError as exc:
            self.assertEqual("bluch", exc.name)
            self.assertEqual(obj, exc.obj)

    # Note: name suggestion tests live in `test_traceback`.


class ImportErrorTests(__TestCase):
    # [CLASS DOC] Tests for ImportError attributes (msg, name, path), keyword validation,
    # reset via re-running the constructor, and copy/pickle support.

    def test_attributes(self):
        # [FUNCTION DOC] ImportError accepts optional name and path keywords and stores them;
        # any other keyword raises TypeError "unexpected keyword argument".
        # Setting 'name' and 'path' should not be a problem.
        exc = ImportError('test')
        self.assertIsNone(exc.name)
        self.assertIsNone(exc.path)

        exc = ImportError('test', name='somemodule')
        self.assertEqual(exc.name, 'somemodule')
        self.assertIsNone(exc.path)

        exc = ImportError('test', path='somepath')
        self.assertEqual(exc.path, 'somepath')
        self.assertIsNone(exc.name)

        exc = ImportError('test', path='somepath', name='somename')
        self.assertEqual(exc.name, 'somename')
        self.assertEqual(exc.path, 'somepath')

        msg = r"ImportError\(\) got an unexpected keyword argument 'invalid'"
        with self.assertRaisesRegex(TypeError, msg):
            ImportError('test', invalid='keyword')

        with self.assertRaisesRegex(TypeError, msg):
            ImportError('test', name='name', invalid='keyword')

        with self.assertRaisesRegex(TypeError, msg):
            ImportError('test', path='path', invalid='keyword')

        with self.assertRaisesRegex(TypeError, msg):
            ImportError(invalid='keyword')

        with self.assertRaisesRegex(TypeError, msg):
            ImportError('test', invalid='keyword', another=True)

    def test_reset_attributes(self):
        # [FUNCTION DOC] Calling the constructor again without arguments resets args, msg,
        # name and path to empty/None.
        exc = ImportError('test', name='name', path='path')
        self.assertEqual(exc.args, ('test',))
        self.assertEqual(exc.msg, 'test')
        self.assertEqual(exc.name, 'name')
        self.assertEqual(exc.path, 'path')

        # Reset not specified attributes
        exc.__init__()
        self.assertEqual(exc.args, ())
        self.assertEqual(exc.msg, None)
        self.assertEqual(exc.name, None)
        self.assertEqual(exc.path, None)

    def test_non_str_argument(self):
        # [FUNCTION DOC] Issue 15778: ImportError built from a bytes argument stringifies the
        # same as the bytes object's str (BytesWarning ignored).
        # Issue #15778
        with check_warnings(('', BytesWarning), quiet=True):
            arg = b'abc'
            exc = ImportError(arg)
            self.assertEqual(str(arg), str(exc))

    def test_copy_pickle(self):
        # [FUNCTION DOC] Pickle (every protocol), copy and deepcopy of an ImportError keep
        # args, msg, name and path.
        for kwargs in (dict(),
                       dict(name='somename'),
                       dict(path='somepath'),
                       dict(name='somename', path='somepath')):
            orig = ImportError('test', **kwargs)
            for proto in range(pickle.HIGHEST_PROTOCOL + 1):
                exc = pickle.loads(pickle.dumps(orig, proto))
                self.assertEqual(exc.args, ('test',))
                self.assertEqual(exc.msg, 'test')
                self.assertEqual(exc.name, orig.name)
                self.assertEqual(exc.path, orig.path)
            for c in copy.copy, copy.deepcopy:
                exc = c(orig)
                self.assertEqual(exc.args, ('test',))
                self.assertEqual(exc.msg, 'test')
                self.assertEqual(exc.name, orig.name)
                self.assertEqual(exc.path, orig.path)


def run_script(source):
    # [FUNCTION DOC] HELPER: Writes source (str, dedented, or bytes as-is) to the TESTFN file,
    # runs it in a subprocess with warnings as errors-display and utf8 mode, expecting a
    # failing exit status, and returns the decoded stderr as a list of lines. Side effect:
    # creates TESTFN (callers must unlink it).
    if isinstance(source, str):
        with open(TESTFN, 'w', encoding='utf-8') as testfile:
            testfile.write(dedent(source))
    else:
        with open(TESTFN, 'wb') as testfile:
            testfile.write(source)
    _rc, _out, err = script_helper.assert_python_failure('-Wd', '-X', 'utf8', TESTFN)
    return err.decode('utf-8').splitlines()

class AssertionErrorTests(__TestCase):
    # [CLASS DOC] Tests for the location highlighting printed with AssertionError tracebacks,
    # run via subprocess scripts. Each test cleans up the TESTFN file in tearDown.
    def tearDown(self):
        # [FUNCTION DOC] Removes the TESTFN file written by run_script.
        unlink(TESTFN)

    @force_not_colorized
    def test_assertion_error_location(self):
        # [FUNCTION DOC] Runs failing assert statements (plain, with message, bytes sources
        # with different encodings or BOM, multi-line asserts) and checks the last three
        # stderr lines: source line, caret range under the failing expression, and the
        # AssertionError message. Colorized output disabled.
        cases = [
            ('assert None',
                [
                    '    assert None',
                    '           ^^^^',
                    'AssertionError',
                ],
            ),
            ('assert 0',
                [
                    '    assert 0',
                    '           ^',
                    'AssertionError',
                ],
            ),
            ('assert 1 > 2',
                [
                    '    assert 1 > 2',
                    '           ^^^^^',
                    'AssertionError',
                ],
            ),
            ('assert 1 > 2 and 3 > 2',
                [
                    '    assert 1 > 2 and 3 > 2',
                    '           ^^^^^^^^^^^^^^^',
                    'AssertionError',
                ],
            ),
            ('assert 1 > 2, "messäge"',
                [
                    '    assert 1 > 2, "messäge"',
                    '           ^^^^^',
                    'AssertionError: messäge',
                ],
            ),
            ('assert 1 > 2, "messäge"'.encode(),
                [
                    '    assert 1 > 2, "messäge"',
                    '           ^^^^^',
                    'AssertionError: messäge',
                ],
            ),
            ('# coding: latin1\nassert 1 > 2, "messäge"'.encode('latin1'),
                [
                    '    assert 1 > 2, "messäge"',
                    '           ^^^^^',
                    'AssertionError: messäge',
                ],
            ),
            (BOM_UTF8 + 'assert 1 > 2, "messäge"'.encode(),
                [
                    '    assert 1 > 2, "messäge"',
                    '           ^^^^^',
                    'AssertionError: messäge',
                ],
            ),

            # Multiline:
            ("""
             assert (
                 1 > 2)
             """,
                [
                    '    1 > 2)',
                    '    ^^^^^',
                    'AssertionError',
                ],
            ),
            ("""
             assert (
                 1 > 2), "Message"
             """,
                [
                    '    1 > 2), "Message"',
                    '    ^^^^^',
                    'AssertionError: Message',
                ],
            ),
            ("""
             assert (
                 1 > 2), \\
                 "Message"
             """,
                [
                    '    1 > 2), \\',
                    '    ^^^^^',
                    'AssertionError: Message',
                ],
            ),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                result = run_script(source)
                self.assertEqual(result[-3:], expected)

    @force_not_colorized
    def test_multiline_not_highlighted(self):
        # [FUNCTION DOC] For multi-line asserts where the failing expression spans lines,
        # checks the traceback shows the source lines without a caret marker line. Colorized
        # output disabled.
        cases = [
            ("""
             assert (
                 1 > 2
             )
             """,
                [
                    '    1 > 2',
                    'AssertionError',
                ],
            ),
            ("""
             assert (
                 1 < 2 and
                 3 > 4
             )
             """,
                [
                    '    1 < 2 and',
                    '    3 > 4',
                    'AssertionError',
                ],
            ),
        ]
        for source, expected in cases:
            with self.subTest(source=source):
                result = run_script(source)
                self.assertEqual(result[-len(expected):], expected)


@support.force_not_colorized_test_class
class SyntaxErrorTests(__TestCase):
    # [CLASS DOC] Tests SyntaxError reporting: how offsets are rendered by the excepthook,
    # subclass handling, encodings, string versus file sources, and constructor attributes.
    # Class decorated to disable colorized output. maxDiff is unlimited.
    maxDiff = None

    @force_not_colorized
    def test_range_of_offsets(self):
        # [FUNCTION DOC] Raises SyntaxError with many start/end offset combinations (normal
        # range, negative, zero, reversed, past line end) and checks the caret line printed by
        # the excepthook for each (no carets when the start offset is not positive).
        cases = [
            # Basic range from 2->7
            (("bad.py", 1, 2, "abcdefg", 1, 7),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
                  ^^^^^
             SyntaxError: bad bad
             """)),
            # end_offset = start_offset + 1
            (("bad.py", 1, 2, "abcdefg", 1, 3),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
                  ^
             SyntaxError: bad bad
             """)),
            # Negative end offset
            (("bad.py", 1, 2, "abcdefg", 1, -2),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
                  ^
             SyntaxError: bad bad
             """)),
            # end offset before starting offset
            (("bad.py", 1, 4, "abcdefg", 1, 2),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
                    ^
             SyntaxError: bad bad
             """)),
            # Both offsets negative
            (("bad.py", 1, -4, "abcdefg", 1, -2),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
             SyntaxError: bad bad
             """)),
            # Both offsets negative and the end more negative
            (("bad.py", 1, -4, "abcdefg", 1, -5),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
             SyntaxError: bad bad
             """)),
            # Both offsets 0
            (("bad.py", 1, 0, "abcdefg", 1, 0),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
             SyntaxError: bad bad
             """)),
            # Start offset 0 and end offset not 0
            (("bad.py", 1, 0, "abcdefg", 1, 5),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
             SyntaxError: bad bad
             """)),
            # End offset pass the source length
            (("bad.py", 1, 2, "abcdefg", 1, 100),
             dedent(
             """
               File "bad.py", line 1
                 abcdefg
                  ^^^^^^
             SyntaxError: bad bad
             """)),
        ]
        for args, expected in cases:
            with self.subTest(args=args):
                try:
                    raise SyntaxError("bad bad", args)
                except SyntaxError as exc:
                    with support.captured_stderr() as err:
                        sys.__excepthook__(*sys.exc_info())
                    self.assertIn(expected, err.getvalue())
                    the_exception = exc

    def test_subclass(self):
        # [FUNCTION DOC] A user subclass of SyntaxError is printed with the same caret layout.
        class MySyntaxError(SyntaxError):
            # [CLASS DOC] HELPER: Empty SyntaxError subclass.
            pass

        try:
            raise MySyntaxError("bad bad", ("bad.py", 1, 2, "abcdefg", 1, 7))
        except SyntaxError as exc:
            with support.captured_stderr() as err:
                sys.__excepthook__(*sys.exc_info())
            self.assertIn("""
  File "bad.py", line 1
    abcdefg
     ^^^^^
""", err.getvalue())

    def test_encodings(self):
        # [FUNCTION DOC] Checks source line and caret rendering for scripts with a cp437
        # coding declaration, and for a tokenizer error with an ascii declaration. Registers
        # cleanup of TESTFN.
        self.addCleanup(unlink, TESTFN)
        source = (
            '# -*- coding: cp437 -*-\n'
            '"┬ó┬ó┬ó┬ó┬ó┬ó" + f(4, x for x in range(1))\n'
        )
        err = run_script(source.encode('cp437'))
        self.assertEqual(err[-3], '    "┬ó┬ó┬ó┬ó┬ó┬ó" + f(4, x for x in range(1))')
        self.assertEqual(err[-2], '                          ^^^^^^^^^^^^^^^^^^^')

        # Check backwards tokenizer errors
        source = '# -*- coding: ascii -*-\n\n(\n'
        err = run_script(source)
        self.assertEqual(err[-3], '    (')
        self.assertEqual(err[-2], '    ^')

    def test_non_utf8(self):
        # [FUNCTION DOC] A script containing a non-UTF-8 byte fails with a "Non-UTF-8 code
        # starting with" SyntaxError message. Registers cleanup of TESTFN.
        # Check non utf-8 characters
        self.addCleanup(unlink, TESTFN)
        err = run_script(b"\x89")
        self.assertIn("SyntaxError: Non-UTF-8 code starting with '\\x89' in file", err[-1])


    @torch._dynamo.error_on_graph_break(False)
    def test_string_source(self):
        # [FUNCTION DOC] Compiling strings (str, bytes, BOM, latin1 declaration, long
        # comments) with a misplaced return gives SyntaxError with expected message, no text,
        # offset 1 and end offset 12 (line 2 for the latin1 case).
        def try_compile(source):
            # [FUNCTION DOC] HELPER: Compiles source and returns the SyntaxError it must
            # raise.
            with self.assertRaises(SyntaxError) as cm:
                compile(source, '<string>', 'exec')
            return cm.exception

        exc = try_compile('return "ä"')
        self.assertEqual(str(exc), "'return' outside function (<string>, line 1)")
        self.assertIsNone(exc.text)
        self.assertEqual(exc.offset, 1)
        self.assertEqual(exc.end_offset, 12)

        exc = try_compile('return "ä"'.encode())
        self.assertEqual(str(exc), "'return' outside function (<string>, line 1)")
        self.assertIsNone(exc.text)
        self.assertEqual(exc.offset, 1)
        self.assertEqual(exc.end_offset, 12)

        exc = try_compile(BOM_UTF8 + 'return "ä"'.encode())
        self.assertEqual(str(exc), "'return' outside function (<string>, line 1)")
        self.assertIsNone(exc.text)
        self.assertEqual(exc.offset, 1)
        self.assertEqual(exc.end_offset, 12)

        exc = try_compile('# coding: latin1\nreturn "ä"'.encode('latin1'))
        self.assertEqual(str(exc), "'return' outside function (<string>, line 2)")
        self.assertIsNone(exc.text)
        self.assertEqual(exc.offset, 1)
        self.assertEqual(exc.end_offset, 12)

        exc = try_compile('return "ä" #' + 'ä'*1000)
        self.assertEqual(str(exc), "'return' outside function (<string>, line 1)")
        self.assertIsNone(exc.text)
        self.assertEqual(exc.offset, 1)
        self.assertEqual(exc.end_offset, 12)

        exc = try_compile('return "ä" # ' + 'ä'*1000)
        self.assertEqual(str(exc), "'return' outside function (<string>, line 1)")
        self.assertIsNone(exc.text)
        self.assertEqual(exc.offset, 1)
        self.assertEqual(exc.end_offset, 12)

    def test_file_source(self):
        # [FUNCTION DOC] Same misplaced return cases run as script files: checks the printed
        # source line, caret line and message, and that very long lines are truncated as
        # expected. Registers cleanup of TESTFN.
        self.addCleanup(unlink, TESTFN)
        err = run_script('return "ä"')
        self.assertEqual(err[-3:], [
                         '    return "ä"',
                         '    ^^^^^^^^^^',
                         "SyntaxError: 'return' outside function"])

        err = run_script('return "ä"'.encode())
        self.assertEqual(err[-3:], [
                         '    return "ä"',
                         '    ^^^^^^^^^^',
                         "SyntaxError: 'return' outside function"])

        err = run_script(BOM_UTF8 + 'return "ä"'.encode())
        self.assertEqual(err[-3:], [
                         '    return "ä"',
                         '    ^^^^^^^^^^',
                         "SyntaxError: 'return' outside function"])

        err = run_script('# coding: latin1\nreturn "ä"'.encode('latin1'))
        self.assertEqual(err[-3:], [
                         '    return "ä"',
                         '    ^^^^^^^^^^',
                         "SyntaxError: 'return' outside function"])

        err = run_script('return "ä" #' + 'ä'*1000)
        self.assertEqual(err[-2:], [
                         '    ^^^^^^^^^^^',
                         "SyntaxError: 'return' outside function"])
        self.assertEqual(err[-3][:100], '    return "ä" #' + 'ä'*84)

        err = run_script('return "ä" # ' + 'ä'*1000)
        self.assertEqual(err[-2:], [
                         '    ^^^^^^^^^^^',
                         "SyntaxError: 'return' outside function"])
        self.assertEqual(err[-3][:100], '    return "ä" # ' + 'ä'*83)

    def test_attributes_new_constructor(self):
        # [FUNCTION DOC] The six-item location tuple form of the constructor populates
        # filename, lineno, offset, text, end_lineno and end_offset, plus msg.
        args = ("bad.py", 1, 2, "abcdefg", 1, 100)
        the_exception = SyntaxError("bad bad", args)
        filename, lineno, offset, error, end_lineno, end_offset = args
        self.assertEqual(filename, the_exception.filename)
        self.assertEqual(lineno, the_exception.lineno)
        self.assertEqual(end_lineno, the_exception.end_lineno)
        self.assertEqual(offset, the_exception.offset)
        self.assertEqual(end_offset, the_exception.end_offset)
        self.assertEqual(error, the_exception.text)
        self.assertEqual("bad bad", the_exception.msg)

    def test_attributes_old_constructor(self):
        # [FUNCTION DOC] The four-item location tuple form populates filename, lineno, offset
        # and text, leaving end_lineno and end_offset None.
        args = ("bad.py", 1, 2, "abcdefg")
        the_exception = SyntaxError("bad bad", args)
        filename, lineno, offset, error = args
        self.assertEqual(filename, the_exception.filename)
        self.assertEqual(lineno, the_exception.lineno)
        self.assertEqual(None, the_exception.end_lineno)
        self.assertEqual(offset, the_exception.offset)
        self.assertEqual(None, the_exception.end_offset)
        self.assertEqual(error, the_exception.text)
        self.assertEqual("bad bad", the_exception.msg)

    def test_incorrect_constructor(self):
        # [FUNCTION DOC] Location tuples with the wrong length or wrong item types raise
        # TypeError.
        args = ("bad.py", 1, 2)
        self.assertRaises(TypeError, SyntaxError, "bad bad", args)

        args = ("bad.py", 1, 2, 4, 5, 6, 7)
        self.assertRaises(TypeError, SyntaxError, "bad bad", args)

        args = ("bad.py", 1, 2, "abcdefg", 1)
        self.assertRaises(TypeError, SyntaxError, "bad bad", args)


class TestInvalidExceptionMatcher(__TestCase):
    # [CLASS DOC] Tests that matching exception handlers with invalid types is rejected.
    def test_except_star_invalid_exception_type(self):
        # [FUNCTION DOC] An except clause naming a non-class value, alone or inside a tuple,
        # raises TypeError when an exception is being matched.
        # KNOWN ISSUE: the name says except-star but the code uses plain except clauses.
        with self.assertRaises(TypeError):
            try:
                raise ValueError
            except 42:
                pass

        with self.assertRaises(TypeError):
            try:
                raise ValueError
            except (ValueError, 42):
                pass


class PEP626Tests(__TestCase):
    # [CLASS DOC] PEP 626 tests: line numbers in traceback frames after exceptions in various
    # constructs, using relative line numbers from each function's first line.

    def lineno_after_raise(self, f, *expected):
        # [FUNCTION DOC] HELPER: Calls f, expecting an exception, and walks the traceback past
        # this frame collecting each frame's line number relative to its code's first line
        # (None if the line is missing); asserts the tuple of lines equals expected.
        try:
            f()
        except Exception as ex:
            t = ex.__traceback__
        else:
            self.fail("No exception raised")
        lines = []
        t = t.tb_next # Skip this function
        while t:
            frame = t.tb_frame
            lines.append(
                None if frame.f_lineno is None else
                frame.f_lineno-frame.f_code.co_firstlineno
            )
            t = t.tb_next
        self.assertEqual(tuple(lines), expected)

    def test_lineno_after_raise_simple(self):
        # [FUNCTION DOC] Line number after a simple failing statement is the failing line.
        def simple():
            # [FUNCTION DOC] HELPER: Divides by zero, with a statement after it.
            1/0
            pass
        self.lineno_after_raise(simple, 1)

    def test_lineno_after_raise_in_except(self):
        # [FUNCTION DOC] An exception raised inside an except block reports the line within
        # the block.
        def in_except():
            # [FUNCTION DOC] HELPER: Fails in the try body, then fails again in the handler.
            try:
                1/0
            except:
                1/0
                pass
        self.lineno_after_raise(in_except, 4)

    def test_lineno_after_other_except(self):
        # [FUNCTION DOC] An exception not matched by the handler reports the try-body line.
        def other_except():
            # [FUNCTION DOC] HELPER: Division by zero in a try with a non-matching handler.
            try:
                1/0
            except TypeError as ex:
                pass
        self.lineno_after_raise(other_except, 3)

    def test_lineno_in_named_except(self):
        # [FUNCTION DOC] An exception raised inside a named handler reports the line within
        # the handler.
        def in_named_except():
            # [FUNCTION DOC] HELPER: Fails in the try body and again in the "as" handler.
            try:
                1/0
            except Exception as ex:
                1/0
                pass
        self.lineno_after_raise(in_named_except, 4)

    def test_lineno_in_try(self):
        # [FUNCTION DOC] An exception in a try body with a finally clause reports the try-body
        # line.
        def in_try():
            # [FUNCTION DOC] HELPER: Division by zero in a try body with a finally clause.
            try:
                1/0
            finally:
                pass
        self.lineno_after_raise(in_try, 4)

    def test_lineno_in_finally_normal(self):
        # [FUNCTION DOC] An exception raised in a finally block after normal try completion
        # reports the line within finally.
        def in_finally_normal():
            # [FUNCTION DOC] HELPER: Empty try, then failing finally block.
            try:
                pass
            finally:
                1/0
                pass
        self.lineno_after_raise(in_finally_normal, 4)

    def test_lineno_in_finally_except(self):
        # [FUNCTION DOC] An exception raised in finally after the try body already failed
        # reports the line within finally.
        def in_finally_except():
            # [FUNCTION DOC] HELPER: Failing try body followed by a failing finally block.
            try:
                1/0
            finally:
                1/0
                pass
        self.lineno_after_raise(in_finally_except, 4)

    def test_lineno_after_with(self):
        # [FUNCTION DOC] Line number of a failure inside a with block using a no-op context
        # manager.
        class Noop:
            # [CLASS DOC] HELPER: No-op context manager.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Does nothing and does not suppress exceptions.
                pass
        def after_with():
            # [FUNCTION DOC] HELPER: Fails inside a with block using Noop.
            with Noop():
                1/0
                pass
        self.lineno_after_raise(after_with, 2)

    def test_missing_lineno_shows_as_none(self):
        # [FUNCTION DOC] Replacing the line table of a function's code with a table without
        # line information makes the reported line None.
        def f():
            # [FUNCTION DOC] HELPER: Divides by zero.
            1/0
        self.lineno_after_raise(f, 1)
        f.__code__ = f.__code__.replace(co_linetable=b'\xf8\xf8\xf8\xf9\xf8\xf8\xf8')
        self.lineno_after_raise(f, None)

    def test_lineno_after_raise_in_with_exit(self):
        # [FUNCTION DOC] When a context manager's exit itself raises while an exception is
        # propagating, two frames at the same relative line are reported.
        class ExitFails:
            # [CLASS DOC] HELPER: Context manager whose exit always raises ValueError.
            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self
            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Raises ValueError.
                raise ValueError

        def after_with():
            # [FUNCTION DOC] HELPER: Fails inside a with block using ExitFails.
            with ExitFails():
                1/0
        self.lineno_after_raise(after_with, 1, 1)

if __name__ == "__main__":
    run_tests()
