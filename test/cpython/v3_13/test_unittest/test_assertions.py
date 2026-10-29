# ======= BEGIN Dynamo patch =======
# Owner(s): ["module: dynamo"]

# ruff: noqa
# flake8: noqa

# [MODULE DOC] Dynamo-patched copy of the CPython 3.13 unittest assertion tests. Verifies the
# pass/fail behavior of the assertAlmostEqual/assertRaises family and, in TestLongMessage, the
# exact failure message text of each assert method with longMessage off and on, with and
# without a custom msg. At import time it installs a meta path finder that redirects "test.*"
# imports to top-level modules; the test classes derive from the Dynamo CPythonTestCase.

import sys
import torch
import torch._dynamo.test_case
import unittest
from torch.testing._internal.common_utils import run_tests


__TestCase = torch._dynamo.test_case.CPythonTestCase


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
    # [CLASS DOC] HELPER: Import hook that maps selected "test.xxx" module names (those listed
    # in redirect_imports) onto the standalone top-level module "xxx". Stateless; installed
    # at the front of sys.meta_path by this module.
    def find_spec(self, fullname, path, target=None):
        # [FUNCTION DOC] HELPER: For a name in redirect_imports, imports the same name without
        # the "test." prefix, registers that module in sys.modules under the original name
        # (side effect) and returns its spec. Returns None (defers to other finders) if the
        # name is not listed or the standalone import fails with ImportError.
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

import datetime
import warnings
import weakref
import unittest
from test.support import gc_collect
from itertools import product


class Test_Assertions(__TestCase):
    # [CLASS DOC] Tests the outcome (pass or failureException) of assertAlmostEqual,
    # assertNotAlmostEqual, assertRaises and assertNotRegex, plus that assertRaises does not
    # keep traceback locals alive. No state besides what unittest provides.
    def test_AlmostEqual(self):
        # [FUNCTION DOC] Checks assertAlmostEqual/assertNotAlmostEqual with default 7 places,
        # explicit places, complex numbers (difference measured by absolute value) and
        # infinities (equal to themselves, so assertNotAlmostEqual fails). Failures are
        # detected by expecting failureException.
        self.assertAlmostEqual(1.00000001, 1.0)
        self.assertNotAlmostEqual(1.0000001, 1.0)
        self.assertRaises(self.failureException,
                          self.assertAlmostEqual, 1.0000001, 1.0)
        self.assertRaises(self.failureException,
                          self.assertNotAlmostEqual, 1.00000001, 1.0)

        self.assertAlmostEqual(1.1, 1.0, places=0)
        self.assertRaises(self.failureException,
                          self.assertAlmostEqual, 1.1, 1.0, places=1)

        self.assertAlmostEqual(0, .1+.1j, places=0)
        self.assertNotAlmostEqual(0, .1+.1j, places=1)
        self.assertRaises(self.failureException,
                          self.assertAlmostEqual, 0, .1+.1j, places=1)
        self.assertRaises(self.failureException,
                          self.assertNotAlmostEqual, 0, .1+.1j, places=0)

        self.assertAlmostEqual(float('inf'), float('inf'))
        self.assertRaises(self.failureException, self.assertNotAlmostEqual,
                          float('inf'), float('inf'))

    def test_AmostEqualWithDelta(self):
        # [FUNCTION DOC] Checks the delta form of assertAlmostEqual/assertNotAlmostEqual with
        # numbers (symmetric in argument order), that passing both places and delta raises
        # TypeError, and that delta may be a timedelta when comparing datetimes.
        self.assertAlmostEqual(1.1, 1.0, delta=0.5)
        self.assertAlmostEqual(1.0, 1.1, delta=0.5)
        self.assertNotAlmostEqual(1.1, 1.0, delta=0.05)
        self.assertNotAlmostEqual(1.0, 1.1, delta=0.05)

        self.assertAlmostEqual(1.0, 1.0, delta=0.5)
        self.assertRaises(self.failureException, self.assertNotAlmostEqual,
                          1.0, 1.0, delta=0.5)

        self.assertRaises(self.failureException, self.assertAlmostEqual,
                          1.1, 1.0, delta=0.05)
        self.assertRaises(self.failureException, self.assertNotAlmostEqual,
                          1.1, 1.0, delta=0.5)

        self.assertRaises(TypeError, self.assertAlmostEqual,
                          1.1, 1.0, places=2, delta=2)
        self.assertRaises(TypeError, self.assertNotAlmostEqual,
                          1.1, 1.0, places=2, delta=2)

        first = datetime.datetime.now()
        second = first + datetime.timedelta(seconds=10)
        self.assertAlmostEqual(first, second,
                               delta=datetime.timedelta(seconds=20))
        self.assertNotAlmostEqual(first, second,
                                  delta=datetime.timedelta(seconds=5))

    def test_assertRaises(self):
        # [FUNCTION DOC] Checks assertRaises in call form and context-manager form: passes
        # when the expected exception (class or instance) is raised, fails with a message
        # containing "KeyError not raised" when nothing is raised, lets an unexpected
        # exception type propagate, and exposes the caught exception as the manager's
        # exception attribute.
        def _raise(e):
            # [FUNCTION DOC] HELPER: Raises the given exception class or instance.
            raise e
        self.assertRaises(KeyError, _raise, KeyError)
        self.assertRaises(KeyError, _raise, KeyError("key"))
        try:
            self.assertRaises(KeyError, lambda: None)
        except self.failureException as e:
            self.assertIn("KeyError not raised", str(e))
        else:
            self.fail("assertRaises() didn't fail")
        try:
            self.assertRaises(KeyError, _raise, ValueError)
        except ValueError:
            pass
        else:
            self.fail("assertRaises() didn't let exception pass through")
        with self.assertRaises(KeyError) as cm:
            try:
                raise KeyError
            except Exception as e:
                exc = e
                raise
        self.assertIs(cm.exception, exc)

        with self.assertRaises(KeyError):
            raise KeyError("key")
        try:
            with self.assertRaises(KeyError):
                pass
        except self.failureException as e:
            self.assertIn("KeyError not raised", str(e))
        else:
            self.fail("assertRaises() didn't fail")
        try:
            with self.assertRaises(KeyError):
                raise ValueError
        except ValueError:
            pass
        else:
            self.fail("assertRaises() didn't let exception pass through")

    def test_assertRaises_frames_survival(self):
        # [FUNCTION DOC] Regression test for issue 9815: runs inner test cases whose method
        # creates a local object (tracked by a weak reference) and raises ValueError from an
        # OSError handler; after assertRaises (call and with forms) and a garbage collection,
        # the weak reference must be dead, i.e. assertRaises must not keep frame locals alive.
        # Issue #9815: assertRaises should avoid keeping local variables
        # in a traceback alive.
        class A:
            # [CLASS DOC] HELPER: Empty class whose instance is tracked by a weak reference.
            pass
        wr = None

        class Foo(unittest.TestCase):
            # [CLASS DOC] HELPER: Inner test case exercising assertRaises on a method that
            # raises ValueError while creating a weakly referenced local.

            def foo(self):
                # [FUNCTION DOC] HELPER: Creates an A instance, stores a weak reference to it
                # in the enclosing variable wr (side effect), raises OSError and re-raises
                # ValueError from its handler.
                nonlocal wr
                a = A()
                wr = weakref.ref(a)
                try:
                    raise OSError
                except OSError:
                    raise ValueError

            def test_functional(self):
                # [FUNCTION DOC] HELPER: Calls assertRaises for ValueError in call form.
                self.assertRaises(ValueError, self.foo)

            def test_with(self):
                # [FUNCTION DOC] HELPER: Uses assertRaises for ValueError as a context manager.
                with self.assertRaises(ValueError):
                    self.foo()

        Foo("test_functional").run()
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(wr())
        Foo("test_with").run()
        gc_collect()  # For PyPy or other GCs.
        self.assertIsNone(wr())

    def testAssertNotRegex(self):
        # [FUNCTION DOC] Checks assertNotRegex passes when the pattern does not occur and
        # fails with failureException whose first argument contains the custom message when
        # the pattern does match.
        self.assertNotRegex('Ala ma kota', r'r+')
        try:
            self.assertNotRegex('Ala ma kota', r'k.t', 'Message')
        except self.failureException as e:
            self.assertIn('Message', e.args[0])
        else:
            self.fail('assertNotRegex should have failed.')


class TestLongMessage(__TestCase):
    # [CLASS DOC] Verifies the exact failure message of each assert method under all four
    # combinations of longMessage (off/on) and msg (absent/"oops"). setUp builds two inner
    # test cases (testableFalse, testableTrue) whose assert methods are exercised via the
    # assertMessages and assertMessagesCM helpers.
    """Test that the individual asserts honour longMessage.
    This actually tests all the message behaviour for
    asserts that use longMessage."""

    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp, defines two inner TestCase classes that differ
        # only in longMessage (False / True) and share this test's failureException, and
        # stores one instance of each as testableFalse and testableTrue.
        super().setUp()
        class TestableTestFalse(unittest.TestCase):
            # [CLASS DOC] HELPER: Test case with longMessage disabled.
            longMessage = False
            failureException = self.failureException

            def testTest(self):
                # [FUNCTION DOC] HELPER: Placeholder test method; does nothing.
                pass

        class TestableTestTrue(unittest.TestCase):
            # [CLASS DOC] HELPER: Test case with longMessage enabled.
            longMessage = True
            failureException = self.failureException

            def testTest(self):
                # [FUNCTION DOC] HELPER: Placeholder test method; does nothing.
                pass

        self.testableTrue = TestableTestTrue('testTest')
        self.testableFalse = TestableTestFalse('testTest')

    def testDefault(self):
        # [FUNCTION DOC] Checks that unittest.TestCase.longMessage defaults to true.
        self.assertTrue(unittest.TestCase.longMessage)

    def test_formatMsg(self):
        # [FUNCTION DOC] Checks the message formatter: with longMessage off the custom message
        # wins when given, otherwise the standard one; with it on, a custom message is
        # appended to the standard one after a separator. A non-string message must not
        # break formatting (no naive string concatenation).
        self.assertEqual(self.testableFalse._formatMessage(None, "foo"), "foo")
        self.assertEqual(self.testableFalse._formatMessage("foo", "bar"), "foo")

        self.assertEqual(self.testableTrue._formatMessage(None, "foo"), "foo")
        self.assertEqual(self.testableTrue._formatMessage("foo", "bar"), "bar : foo")

        # This blows up if _formatMessage uses string concatenation
        self.testableTrue._formatMessage(object(), 'foo')

    def test_formatMessage_unicode_error(self):
        # [FUNCTION DOC] Regression check: formatting a message made of the first 255 code
        # points plus a replacement character with longMessage on must not raise a
        # UnicodeDecodeError.
        one = ''.join(chr(i) for i in range(255))
        # this used to cause a UnicodeDecodeError constructing msg
        self.testableTrue._formatMessage(one, '\uFFFD')

    def assertMessages(self, methodName, args, errors):
        # [FUNCTION DOC] HELPER: For each of the four expected regexes in errors, calls the
        # named assert method with args on testableFalse (first two) or testableTrue (last
        # two), adding the keyword msg "oops" for the odd-indexed ones, and requires a
        # failureException whose text matches that regex.
        """
        Check that methodName(*args) raises the correct error messages.
        errors should be a list of 4 regex that match the error when:
          1) longMessage = False and no msg passed;
          2) longMessage = False and msg passed;
          3) longMessage = True and no msg passed;
          4) longMessage = True and msg passed;
        """
        def getMethod(i):
            # [FUNCTION DOC] HELPER: Returns the bound assert method (named by methodName)
            # of testableFalse when i is below 2, else of testableTrue.
            useTestableFalse  = i < 2
            if useTestableFalse:
                test = self.testableFalse
            else:
                test = self.testableTrue
            return getattr(test, methodName)

        for i, expected_regex in enumerate(errors):
            testMethod = getMethod(i)
            kwargs = {}
            withMsg = i % 2
            if withMsg:
                kwargs = {"msg": "oops"}

            with self.assertRaisesRegex(self.failureException,
                                        expected_regex=expected_regex):
                testMethod(*args, **kwargs)

    def testAssertTrue(self):
        # [FUNCTION DOC] Checks assertTrue failure messages for the value False.
        self.assertMessages('assertTrue', (False,),
                            ["^False is not true$", "^oops$", "^False is not true$",
                             "^False is not true : oops$"])

    def testAssertFalse(self):
        # [FUNCTION DOC] Checks assertFalse failure messages for the value True.
        self.assertMessages('assertFalse', (True,),
                            ["^True is not false$", "^oops$", "^True is not false$",
                             "^True is not false : oops$"])

    def testNotEqual(self):
        # [FUNCTION DOC] Checks assertNotEqual failure messages for equal operands 1 and 1.
        self.assertMessages('assertNotEqual', (1, 1),
                            ["^1 == 1$", "^oops$", "^1 == 1$",
                             "^1 == 1 : oops$"])

    def testAlmostEqual(self):
        # [FUNCTION DOC] Checks assertAlmostEqual failure messages for 1 versus 2, which
        # mention 7 places and the difference.
        self.assertMessages(
            'assertAlmostEqual', (1, 2),
            [r"^1 != 2 within 7 places \(1 difference\)$", "^oops$",
             r"^1 != 2 within 7 places \(1 difference\)$",
             r"^1 != 2 within 7 places \(1 difference\) : oops$"])

    def testNotAlmostEqual(self):
        # [FUNCTION DOC] Checks assertNotAlmostEqual failure messages for 1 versus 1.
        self.assertMessages('assertNotAlmostEqual', (1, 1),
                            ["^1 == 1 within 7 places$", "^oops$",
                             "^1 == 1 within 7 places$", "^1 == 1 within 7 places : oops$"])

    def test_baseAssertEqual(self):
        # [FUNCTION DOC] Checks failure messages of the internal base equality assert for 1
        # versus 2.
        self.assertMessages('_baseAssertEqual', (1, 2),
                            ["^1 != 2$", "^oops$", "^1 != 2$", "^1 != 2 : oops$"])

    def testAssertSequenceEqual(self):
        # [FUNCTION DOC] Checks assertSequenceEqual messages for an empty list versus a list
        # with None; only the tail of the multi-line diff is matched. The list and tuple
        # equality asserts delegate to this method.
        # Error messages are multiline so not testing on full message
        # assertTupleEqual and assertListEqual delegate to this method
        self.assertMessages('assertSequenceEqual', ([], [None]),
                            [r"\+ \[None\]$", "^oops$", r"\+ \[None\]$",
                             r"\+ \[None\] : oops$"])

    def testAssertSetEqual(self):
        # [FUNCTION DOC] Checks assertSetEqual messages for an empty set versus a set holding
        # None.
        self.assertMessages('assertSetEqual', (set(), set([None])),
                            ["None$", "^oops$", "None$",
                             "None : oops$"])

    def testAssertIn(self):
        # [FUNCTION DOC] Checks assertIn messages when None is absent from an empty list.
        self.assertMessages('assertIn', (None, []),
                            [r'^None not found in \[\]$', "^oops$",
                             r'^None not found in \[\]$',
                             r'^None not found in \[\] : oops$'])

    def testAssertNotIn(self):
        # [FUNCTION DOC] Checks assertNotIn messages when None is present in the list.
        self.assertMessages('assertNotIn', (None, [None]),
                            [r'^None unexpectedly found in \[None\]$', "^oops$",
                             r'^None unexpectedly found in \[None\]$',
                             r'^None unexpectedly found in \[None\] : oops$'])

    def testAssertDictEqual(self):
        # [FUNCTION DOC] Checks assertDictEqual messages for an empty dict versus a one-item
        # dict (diff tail only).
        self.assertMessages('assertDictEqual', ({}, {'key': 'value'}),
                            [r"\+ \{'key': 'value'\}$", "^oops$",
                             r"\+ \{'key': 'value'\}$",
                             r"\+ \{'key': 'value'\} : oops$"])

    def testAssertMultiLineEqual(self):
        # [FUNCTION DOC] Checks assertMultiLineEqual messages for an empty string versus foo
        # (diff tail only).
        self.assertMessages('assertMultiLineEqual', ("", "foo"),
                            [r"\+ foo\n$", "^oops$",
                             r"\+ foo\n$",
                             r"\+ foo\n : oops$"])

    def testAssertLess(self):
        # [FUNCTION DOC] Checks assertLess messages for 2 versus 1.
        self.assertMessages('assertLess', (2, 1),
                            ["^2 not less than 1$", "^oops$",
                             "^2 not less than 1$", "^2 not less than 1 : oops$"])

    def testAssertLessEqual(self):
        # [FUNCTION DOC] Checks assertLessEqual messages for 2 versus 1.
        self.assertMessages('assertLessEqual', (2, 1),
                            ["^2 not less than or equal to 1$", "^oops$",
                             "^2 not less than or equal to 1$",
                             "^2 not less than or equal to 1 : oops$"])

    def testAssertGreater(self):
        # [FUNCTION DOC] Checks assertGreater messages for 1 versus 2.
        self.assertMessages('assertGreater', (1, 2),
                            ["^1 not greater than 2$", "^oops$",
                             "^1 not greater than 2$",
                             "^1 not greater than 2 : oops$"])

    def testAssertGreaterEqual(self):
        # [FUNCTION DOC] Checks assertGreaterEqual messages for 1 versus 2.
        self.assertMessages('assertGreaterEqual', (1, 2),
                            ["^1 not greater than or equal to 2$", "^oops$",
                             "^1 not greater than or equal to 2$",
                             "^1 not greater than or equal to 2 : oops$"])

    def testAssertIsNone(self):
        # [FUNCTION DOC] Checks assertIsNone messages when given a non-None string.
        self.assertMessages('assertIsNone', ('not None',),
                            ["^'not None' is not None$", "^oops$",
                             "^'not None' is not None$",
                             "^'not None' is not None : oops$"])

    def testAssertIsNotNone(self):
        # [FUNCTION DOC] Checks assertIsNotNone messages when given None.
        self.assertMessages('assertIsNotNone', (None,),
                            ["^unexpectedly None$", "^oops$",
                             "^unexpectedly None$",
                             "^unexpectedly None : oops$"])

    def testAssertIs(self):
        # [FUNCTION DOC] Checks assertIs messages for None versus a string.
        self.assertMessages('assertIs', (None, 'foo'),
                            ["^None is not 'foo'$", "^oops$",
                             "^None is not 'foo'$",
                             "^None is not 'foo' : oops$"])

    def testAssertIsNot(self):
        # [FUNCTION DOC] Checks assertIsNot messages when both operands are None.
        self.assertMessages('assertIsNot', (None, None),
                            ["^unexpectedly identical: None$", "^oops$",
                             "^unexpectedly identical: None$",
                             "^unexpectedly identical: None : oops$"])

    def testAssertRegex(self):
        # [FUNCTION DOC] Checks assertRegex messages when the pattern does not match.
        self.assertMessages('assertRegex', ('foo', 'bar'),
                            ["^Regex didn't match:",
                             "^oops$",
                             "^Regex didn't match:",
                             "^Regex didn't match: (.*) : oops$"])

    def testAssertNotRegex(self):
        # [FUNCTION DOC] Checks assertNotRegex messages when the pattern matches the text.
        self.assertMessages('assertNotRegex', ('foo', 'foo'),
                            ["^Regex matched:",
                             "^oops$",
                             "^Regex matched:",
                             "^Regex matched: (.*) : oops$"])


    def assertMessagesCM(self, methodName, args, func, errors):
        # [FUNCTION DOC] HELPER: Context-manager variant of assertMessages. Pairs the four
        # combinations (longMessage off/on) x (no msg / msg "oops") with the four regexes in
        # errors, enters the named context-manager assert on the matching test case with
        # args, runs func inside it, and requires a failureException matching the regex.
        """
        Check that the correct error messages are raised while executing:
          with method(*args):
              func()
        *errors* should be a list of 4 regex that match the error when:
          1) longMessage = False and no msg passed;
          2) longMessage = False and msg passed;
          3) longMessage = True and no msg passed;
          4) longMessage = True and msg passed;
        """
        p = product((self.testableFalse, self.testableTrue),
                    ({}, {"msg": "oops"}))
        for (cls, kwargs), err in zip(p, errors):
            method = getattr(cls, methodName)
            with self.assertRaisesRegex(cls.failureException, err):
                with method(*args, **kwargs) as cm:
                    func()

    def testAssertRaises(self):
        # [FUNCTION DOC] Checks assertRaises context-manager messages when TypeError is not
        # raised.
        self.assertMessagesCM('assertRaises', (TypeError,), lambda: None,
                              ['^TypeError not raised$', '^oops$',
                               '^TypeError not raised$',
                               '^TypeError not raised : oops$'])

    def testAssertRaisesRegex(self):
        # [FUNCTION DOC] Checks assertRaisesRegex messages in two cases: nothing raised, and
        # TypeError raised with text foo that does not match the regex.
        # test error not raised
        self.assertMessagesCM('assertRaisesRegex', (TypeError, 'unused regex'),
                              lambda: None,
                              ['^TypeError not raised$', '^oops$',
                               '^TypeError not raised$',
                               '^TypeError not raised : oops$'])
        # test error raised but with wrong message
        def raise_wrong_message():
            # [FUNCTION DOC] HELPER: Raises TypeError with message foo.
            raise TypeError('foo')
        self.assertMessagesCM('assertRaisesRegex', (TypeError, 'regex'),
                              raise_wrong_message,
                              ['^"regex" does not match "foo"$', '^oops$',
                               '^"regex" does not match "foo"$',
                               '^"regex" does not match "foo" : oops$'])

    def testAssertWarns(self):
        # [FUNCTION DOC] Checks assertWarns messages when no UserWarning is triggered.
        self.assertMessagesCM('assertWarns', (UserWarning,), lambda: None,
                              ['^UserWarning not triggered$', '^oops$',
                               '^UserWarning not triggered$',
                               '^UserWarning not triggered : oops$'])

    def test_assertNotWarns(self):
        # [FUNCTION DOC] Checks the internal not-warns assert messages when a FutureWarning
        # is in fact triggered inside the block.
        def warn_future():
            # [FUNCTION DOC] HELPER: Emits a FutureWarning with text xyz (stacklevel 2).
            warnings.warn('xyz', FutureWarning, stacklevel=2)
        self.assertMessagesCM('_assertNotWarns', (FutureWarning,),
                              warn_future,
                              ['^FutureWarning triggered$',
                               '^oops$',
                               '^FutureWarning triggered$',
                               '^FutureWarning triggered : oops$'])

    def testAssertWarnsRegex(self):
        # [FUNCTION DOC] Checks assertWarnsRegex messages in two cases: no warning triggered,
        # and a UserWarning with text foo that does not match the regex.
        # test error not raised
        self.assertMessagesCM('assertWarnsRegex', (UserWarning, 'unused regex'),
                              lambda: None,
                              ['^UserWarning not triggered$', '^oops$',
                               '^UserWarning not triggered$',
                               '^UserWarning not triggered : oops$'])
        # test warning raised but with wrong message
        def raise_wrong_message():
            # [FUNCTION DOC] HELPER: Emits a default UserWarning with text foo.
            warnings.warn('foo')
        self.assertMessagesCM('assertWarnsRegex', (UserWarning, 'regex'),
                              raise_wrong_message,
                              ['^"regex" does not match "foo"$', '^oops$',
                               '^"regex" does not match "foo"$',
                               '^"regex" does not match "foo" : oops$'])


if __name__ == "__main__":
    run_tests()
