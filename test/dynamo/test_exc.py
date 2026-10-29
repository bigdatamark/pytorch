# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch._dynamo.exc exception types and the error/graph-break messages
# dynamo produces: pickling of dynamo exceptions, UserError vs Unsupported, munged expected
# traceback text, graph break and WON'T CONVERT log records, translation validation failures
# (z3), and SourceLocation formatting/tracking. Defines a module-level capture dict, two
# helper functions and the ExcTests logging test case. No import-time side effects besides that.

import linecache
import os
import pickle
import re
import sys
import tempfile
import unittest
from typing import cast

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
from torch._dynamo.comptime import comptime
from torch._dynamo.exc import (
    BackendCompilerFailed,
    InvalidBackend,
    ResetRequired,
    ShortenTraceback,
    TorchDynamoException,
    Unsupported,
    UserError,
    UserErrorType,
)
from torch._dynamo.variables.base import SourceLocation
from torch.testing._internal.common_device_type import skipIf
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    munge_exc,
    skipIfWindows,
    TEST_Z3,
)
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


# Module-level storage avoids free-variable issues when capturing comptime state.
_source_location_capture: dict[str, SourceLocation] = {}


def _capture_y_source_location(ctx) -> None:
    # [FUNCTION DOC] HELPER: comptime callback. Looks up the traced local variable named y in
    # the instruction translator's symbolic locals and, if it exists and has a source location,
    # stores that location in the module-level capture dict under the key source_location.
    # Does nothing otherwise.
    tx = ctx._i_will_not_complain_if_bc_breaks_InstructionTranslator()
    y_vt = tx.symbolic_locals.get("y")
    if y_vt is not None and y_vt.source_location is not None:
        _source_location_capture["source_location"] = y_vt.source_location


def _format_multiline_source_location() -> str:
    # [FUNCTION DOC] HELPER: writes a four-line temporary python file, builds a SourceLocation
    # spanning lines one to four (columns 8 to 1) and returns its formatted text with the temp
    # path replaced by a placeholder. Always deletes the temp file and clears the linecache.
    with tempfile.NamedTemporaryFile("w", suffix=".py", delete=False) as source_file:
        source_file.write("value = (\n    foo\n    + bar\n)\n")
        source_path = source_file.name

    try:
        source_location = SourceLocation(
            filename=source_path,
            lineno=1,
            end_lineno=4,
            # Span covers the parenthesized expression `( ... )`.
            col_offset=8,
            end_col_offset=1,
        )
        return source_location.format().replace(source_path, "<source_path>")
    finally:
        os.unlink(source_path)
        linecache.clearcache()


class ExcTests(LoggingTestCase):
    # [CLASS DOC] Logging-aware test case (no instance state; maxDiff disabled so full diffs
    # print) with tests for dynamo exception classes, error message text, graph break logs,
    # translation validation errors and SourceLocation behavior.
    maxDiff = None

    @torch._dynamo.config.patch(suppress_errors=True)
    def test_user_error_backend_soft_fail(self):
        # [FUNCTION DOC] With suppress_errors on, a backend raising UserError does not fail the
        # call: the compiled function falls back to eager and returns the correct result.
        def backend(_, __):
            # [FUNCTION DOC] HELPER: backend that always raises UserError (invalid input).
            raise UserError(UserErrorType.INVALID_INPUT, "backend user error")

        def fn(x):
            # [FUNCTION DOC] HELPER: returns the input plus one.
            return x + 1

        x = torch.randn(2)
        self.assertEqual(torch.compile(fn, backend=backend)(x), fn(x))

    def test_user_error_separated_from_unsupported(self):
        # [FUNCTION DOC] Checks UserError is a TorchDynamoException but not an Unsupported,
        # and that its error_type, msg, message and string form carry the given values.
        err = UserError(UserErrorType.INVALID_INPUT, "bad input")

        self.assertIsInstance(err, TorchDynamoException)
        self.assertNotIsInstance(err, Unsupported)
        self.assertEqual(err.error_type, UserErrorType.INVALID_INPUT)
        self.assertEqual(err.msg, "bad input")
        self.assertEqual(err.message, "bad input")
        self.assertEqual(str(err), "bad input")

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_backend_compiler_failed_pickles_without_frame(self):
        # [FUNCTION DOC] Compiles with a backend that raises a RuntimeError carrying a frame
        # attribute, catches the resulting BackendCompilerFailed, round-trips it through pickle
        # and checks args, text, backend name, inner exception (type, message and recorded
        # original type name) survive while first_useful_frame becomes None.
        def backend_fn(gm, example_inputs):
            # [FUNCTION DOC] HELPER: backend that attaches the current frame to a RuntimeError
            # and raises it.
            exc = RuntimeError("backend exploded")
            exc.frame = sys._getframe()
            raise exc

        def fn(x):
            # [FUNCTION DOC] HELPER: returns the input plus one.
            return x + 1

        with self.assertRaises(BackendCompilerFailed) as cm:
            torch.compile(fn, backend=backend_fn)(torch.ones(1))

        err = cm.exception

        restored = pickle.loads(pickle.dumps(err))

        self.assertIsInstance(restored, BackendCompilerFailed)
        self.assertEqual(restored.args, err.args)
        self.assertEqual(str(restored), str(err))
        self.assertEqual(restored.backend_name, "backend_fn")
        self.assertIsInstance(restored.inner_exception, RuntimeError)
        self.assertEqual(str(restored.inner_exception), "backend exploded")
        self.assertEqual(
            restored.inner_exception._dynamo_original_exception_type,
            "builtins.RuntimeError",
        )
        self.assertIsNone(restored.first_useful_frame)

    def test_dynamo_exceptions_pickle_without_rerunning_init(self):
        # [FUNCTION DOC] Pickle round-trips InvalidBackend, ResetRequired, ShortenTraceback and
        # UserError instances and checks type, args and string form are preserved; also checks
        # the ShortenTraceback loses its first_useful_frame (None) and UserError keeps its
        # error_type.
        cases = [
            InvalidBackend("bad_backend"),
            ResetRequired(),
            ShortenTraceback("shortened", first_useful_frame=sys._getframe()),
            UserError(UserErrorType.INVALID_INPUT, "bad input"),
        ]

        for err in cases:
            with self.subTest(exc_type=type(err).__name__):
                restored = pickle.loads(pickle.dumps(err))

                self.assertIsInstance(restored, type(err))
                self.assertEqual(restored.args, err.args)
                self.assertEqual(str(restored), str(err))

        self.assertIsNone(pickle.loads(pickle.dumps(cases[2])).first_useful_frame)
        self.assertEqual(
            pickle.loads(pickle.dumps(cases[3])).error_type,
            UserErrorType.INVALID_INPUT,
        )

    def test_unsupported_real_stack(self):
        # [FUNCTION DOC] Compiles with fullgraph a function calling another that inserts a
        # graph break and asserts the raised Unsupported message (munged) has the explanation,
        # hint, doc link and a "from user code" stack of both frames.
        # exercise Unsupported constructor and augment_exc_message
        def fn002(x):
            # [FUNCTION DOC] HELPER: inserts an explicit graph break.
            torch._dynamo.graph_break()

        def fn001(x):
            # [FUNCTION DOC] HELPER: adds one to the input then calls fn002.
            x = x + 1
            fn002(x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn001, backend="eager", fullgraph=True)(
                torch.randn(1)
            ),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_exc.py", line N, in fn001
    fn002(x)
  File "test_exc.py", line N, in fn002
    torch._dynamo.graph_break()""",
        )

    @torch._dynamo.config.patch(verbose=True, suppress_errors=True)
    @make_logging_test()
    @unittest.skipIf(IS_FBCODE, "stack trace slightly different in fbcode")
    def test_internal_error_suppress_errors(self, records):
        # [FUNCTION DOC] With suppress_errors and verbose on, an AssertionError raised in a
        # comptime callback makes dynamo log a WON'T CONVERT record; asserts the munged text
        # holds the dynamo stack trace and the processed-code stack. Skipped in fbcode.
        def fn001(x):
            # [FUNCTION DOC] HELPER: runs a comptime callback that raises AssertionError.
            def f(ctx):
                # [FUNCTION DOC] HELPER: comptime callback raising AssertionError.
                raise AssertionError

            comptime(f)

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "WON'T CONVERT")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
========== TorchDynamo Stack Trace ==========
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise AssertionError
AssertionError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)


========== The above exception occurred while processing the following code ==========

  File "test_exc.py", line N, in test_internal_error_suppress_errors
    torch.compile(fn001, backend="eager")(torch.randn(1))
  File "test_exc.py", line N, in fn001
    comptime(f)

==========""",
        )

    @make_logging_test()
    def test_not_implemented_error(self, records):
        # [FUNCTION DOC] A comptime callback raising NotImplementedError inside a loop (so no
        # graph break is possible) yields a WON'T CONVERT log record whose munged text reports
        # an InternalTorchDynamoError wrapping NotImplementedError; the call itself succeeds.
        def fn001(x):
            # [FUNCTION DOC] HELPER: calls the raising callback three times in a loop.
            def f(ctx):
                # [FUNCTION DOC] HELPER: comptime callback raising NotImplementedError.
                raise NotImplementedError

            # Ensure graph break is not possible
            for _ in range(3):
                comptime(f)

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "WON'T CONVERT")

        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
WON'T CONVERT fn001 test_exc.py line N
due to:
Traceback (most recent call last):
  File "test_exc.py", line N, in f
    raise NotImplementedError
torch._dynamo.exc.InternalTorchDynamoError: NotImplementedError:

from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

    @torch._dynamo.config.patch(inject_BUILD_SET_unimplemented_TESTING_ONLY=True)
    @make_logging_test(graph_breaks=True)
    def test_unsupported_error(self, records):
        # [FUNCTION DOC] With the testing flag that makes the set-building bytecode handler
        # unimplemented, compiling a function returning a set falls back to eager and logs a
        # graph break record; asserts its munged text (reason, explanation, doc link, traceback).
        def fn001(x):
            # [FUNCTION DOC] HELPER: returns a literal set of two ints.
            return {1, 2}

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "missing BUILD_SET handler")
        self.assertExpectedInline(
            munge_exc(record.getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_exc.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

missing BUILD_SET handler
  Explanation: Missing BUILD_SET bytecode handler (for testing purposes).


  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0200.html

Stack variable source attribution:

User code traceback:
  File "test_exc.py", line N, in test_unsupported_error
    torch.compile(fn001, backend="eager")(torch.randn(1))
  File "test_exc.py", line N, in fn001
    return {1, 2}
""",
        )

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_internal_error_no_suppress(self):
        # [FUNCTION DOC] With suppress_errors off, a comptime callback raising AssertionError
        # propagates as AssertionError whose message ends with the munged "from user code" stack.
        def fn001(x):
            # [FUNCTION DOC] HELPER: runs a comptime callback that raises AssertionError.
            # NB: avoid decorator, as 3.11 changed the line number attributed
            # in this situation
            def f(ctx):
                # [FUNCTION DOC] HELPER: comptime callback raising AssertionError.
                raise AssertionError

            comptime(f)

        # NB: OK for user code to be truncated here, because the regular
        # exception backtrace has the rest of the crumbs
        self.assertExpectedInlineMunged(
            AssertionError,
            lambda: torch.compile(fn001, backend="eager")(torch.randn(1)),
            """\


from user code:
   File "test_exc.py", line N, in fn001
    comptime(f)""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log(self, records):
        # [FUNCTION DOC] Compiles a function that calls one containing an explicit graph break
        # and asserts the logged "Graph break in user code" record text (reason, explanation,
        # hint, doc link and user code traceback through both frames).
        def fn002(x):
            # [FUNCTION DOC] HELPER: adds one, inserts a graph break, adds one again.
            x = x + 1
            torch._dynamo.graph_break()
            x = x + 1
            return x

        def fn001(x):
            # [FUNCTION DOC] HELPER: returns the result of fn002.
            return fn002(x)

        torch.compile(fn001, backend="eager")(torch.randn(1))

        record = self.getRecord(records, "Graph break in user code")

        # TODO: This should also report the enclosing frames; need to plumb
        # frame object to it
        self.assertExpectedInline(
            munge_exc(record.getMessage()),
            """\
Graph break in user code at test_exc.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_exc.py", line N, in test_graph_break_log
    torch.compile(fn001, backend="eager")(torch.randn(1))
  File "test_exc.py", line N, in fn001
    return fn002(x)
  File "test_exc.py", line N, in fn002
    torch._dynamo.graph_break()
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_log_generic_jump(self, records):
        # [FUNCTION DOC] Compiles a function branching on a tensor value (data-dependent jump)
        # and only checks that a "Graph break in user code" record was logged.
        def fn(x):
            # [FUNCTION DOC] HELPER: returns x plus one if the sum is positive, else x minus one.
            if x.sum() > 0:
                return x + 1
            else:
                return x - 1

        torch.compile(fn, backend="eager")(torch.ones(3, 3))

        # check for record existence
        self.getRecord(records, "Graph break in user code")

    @make_logging_test(graph_breaks=True)
    def test_reraised_observed_exception_graph_break_log(self, records):
        # [FUNCTION DOC] Calls a compiled function that catches and re-raises a KeyError from a
        # missing dict key; expects KeyError to reach the caller and exactly one graph break
        # record mentioning the lookup line but not the bare re-raise, not a chained-exception
        # note, and no log record carrying exception info.
        def inner(d):
            # [FUNCTION DOC] HELPER: returns the value for the key abc in the mapping.
            return d["abc"]

        @torch.compile(backend="eager", fullgraph=False)
        def fn(d):
            # [FUNCTION DOC] HELPER: calls inner inside a try block and re-raises any exception.
            try:
                inner(d)
            except Exception:  # noqa: TRY203
                raise

        with self.assertRaisesRegex(KeyError, "abc"):
            fn({"def": torch.randn(3, 4)})

        full_records = [
            r for r in records if "Graph break in user code" in r.getMessage()
        ]
        self.assertEqual(len(full_records), 1)

        msg = full_records[0].getMessage()
        self.assertIn('return d["abc"]', msg)
        self.assertNotIn("\n    raise\n", msg)
        self.assertNotIn("During handling of the above exception", msg)
        self.assertFalse(any(record.exc_info is not None for record in records))

    @torch._dynamo.config.patch(suppress_errors=False)
    def test_backend_suppress_line(self):
        # [FUNCTION DOC] A testing backend that fails on relu makes compilation raise
        # BackendCompilerFailed; asserts the message names the backend and ReluCompileError
        # without attributing the failure to the later add.
        def fn001(x):
            # [FUNCTION DOC] HELPER: applies relu then adds one.
            x = torch.relu(x)
            return x + 1

        # Do NOT let this get attributed to x + 1
        self.assertExpectedInlineMunged(
            torch._dynamo.exc.BackendCompilerFailed,
            lambda: torch.compile(fn001, backend="relu_compile_error_TESTING_ONLY")(
                torch.randn(1)
            ),
            """\
backend='relu_compile_error_TESTING_ONLY' raised:
ReluCompileError:""",
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    def test_z3op_sym_not(self):
        # [FUNCTION DOC] Skipped without z3. Checks the z3 translation of the symbolic not
        # operation maps a z3 boolean to its negation and a constant one to a false value.
        import z3

        from torch.fx.experimental.validator import TranslationValidator, z3op

        validator = TranslationValidator()
        b = z3.Bool("b")

        self.assertTrue(z3op(torch.sym_not, validator)(b).eq(z3.Not(b)))
        self.assertTrue(
            z3.simplify(z3op(torch.sym_not, validator)(1)).eq(z3.BoolVal(False))
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    @torch.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
        translation_validation_no_bisect=True,
    )
    @skipIfWindows(
        msg='AssertionError: "tran[551 chars]s1 s2 s3) s0)\n  ==> (<= (+ s1 s2) (+ s0 (* -1[511 chars][0])'  # noqa: PLR0133
        != 'tran[551 chars]s1 s2) (+ s0 (* -1 s3)))\n  ==> (<= (+ s1 s2) [483 chars][0])"'
    )
    def test_trigger_on_error(self):
        # [FUNCTION DOC] Skipped without z3. With dynamic shapes and an injected flipped
        # equality bug, translation validation (no bisect) must raise ValidationException on
        # splitting a 20-element tensor by (5, 10, 5); asserts the model, assertions, target
        # and failed expression text.
        from torch.fx.experimental.validator import ValidationException

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def fn(x, shape):
            # [FUNCTION DOC] HELPER: splits the tensor by the given sizes.
            return x.split(shape)

        self.assertExpectedInlineMunged(
            ValidationException,
            lambda: fn(torch.randn(20), (5, 10, 5)),
            """\
translation validation failed.

Model:
  ==> L['shape'][0]: 0
  ==> L['shape'][1]: 0
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s3: 0
  ==> s52: 0
  ==> s77: 3
  ==> s86: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 1)

Target Expressions:
  ==> (!= (+ s3 s52 s86) s77)
  ==> (<= 0 s3)
  ==> (<= 0 s52)
  ==> (<= 0 s86)
  ==> (<= 2 s77)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 0)
  ==> (>= 0 s86)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )

    @skipIf(not TEST_Z3, "z3 not installed")
    @torch._dynamo.config.patch(
        assume_static_by_default=False,
        suppress_errors=False,
    )
    @torch.fx.experimental._config.patch(
        inject_EVALUATE_EXPR_flip_equality_TESTING_ONLY=True,
        translation_validation=True,
    )
    def test_trigger_bisect_on_error(self):
        # [FUNCTION DOC] Skipped without z3. Same setup as the non-bisect test but with bisection
        # enabled: expects BisectValidationException naming the failing expression and the
        # split graph node, plus model, assertions, targets and failed source expressions.
        from torch.fx.experimental.validator import BisectValidationException

        @torch.compile  # noqa: UNSPECIFIED_BACKEND
        def fn(x, shape):
            # [FUNCTION DOC] HELPER: splits the tensor by the given sizes.
            return x.split(shape)

        self.assertExpectedInlineMunged(
            BisectValidationException,
            lambda: fn(torch.randn(20), (5, 10, 5)),
            """\
translation validation failed when evaluating: Eq(s3 + s52 + s86, s77)

Failure occurred while running node:
    %split : [num_users=3] = call_method[target=split](args = (%l_x_, (%l_shape_0_, %l_shape_1_, %l_shape_2_)), kwargs = {})

Model:
  ==> L['shape'][0]: 0
  ==> L['shape'][1]: 0
  ==> L['shape'][2]: 0
  ==> L['x'].size()[0]: 3
  ==> L['x'].storage_offset(): 0
  ==> L['x'].stride()[0]: 1
  ==> s3: 0
  ==> s52: 0
  ==> s77: 3
  ==> s86: 0

Assertions:
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 1)

Target Expressions:
  ==> (!= (+ s3 s52 s86) s77)
  ==> (<= 0 s3)
  ==> (<= 0 s52)
  ==> (<= 0 s86)
  ==> (<= 2 s77)
  ==> (== 0 L['x'].storage_offset())
  ==> (== 1 L['x'].stride()[0])
  ==> (== L['shape'][0] s86)
  ==> (== L['shape'][1] s52)
  ==> (== L['shape'][2] s3)
  ==> (== L['x'].size()[0] s77)
  ==> (> s77 0)

Failed Source Expressions:
  ==> (== (+ L['shape'][0] L['shape'][1] L['shape'][2]) L['x'].size()[0])""",
        )

    def test_source_location_format_no_col_info(self):
        # [FUNCTION DOC] Formatting a SourceLocation with only filename and line yields text
        # with the file and line one and no caret marker characters.
        source_location = SourceLocation(filename=__file__, lineno=1)
        result = source_location.format()
        self.assertIn(f'File "{__file__}", line 1', result)
        self.assertNotIn("^", result)

    def test_source_location_format_with_col_info(self):
        # [FUNCTION DOC] Formatting a SourceLocation of this test file with columns zero to ten
        # on line one includes the file/line text and a run of ten carets.
        source_location = SourceLocation(
            filename=__file__,
            lineno=1,
            end_lineno=1,
            col_offset=0,
            end_col_offset=10,
        )
        result = source_location.format()
        self.assertIn(f'File "{__file__}", line 1', result)
        self.assertIn("^" * 10, result)

    def test_source_location_format_without_source_line(self):
        # [FUNCTION DOC] For a pseudo file name with no retrievable source line, formatting
        # yields exactly the two-space-indented file and line header and nothing else.
        source_location = SourceLocation(
            filename="<string>",
            lineno=1,
            end_lineno=1,
            col_offset=0,
            end_col_offset=10,
        )
        result = source_location.format()
        self.assertEqual(result, '  File "<string>", line 1\n')

    def test_source_location_format_multiline(self):
        # [FUNCTION DOC] Formats a location spanning four lines via the helper; checks the text
        # with marker lines stripped equals the four source lines, and on Python 3.11+ checks
        # the full text including tilde/caret marker lines.
        result = _format_multiline_source_location()

        self.assertExpectedInline(
            re.sub(r"(?m)^[ ]*[~^]+\n?", "", result),
            """\
  File "<source_path>", line 1
    value = (
        foo
        + bar
    )
""",
        )

        if sys.version_info >= (3, 11):
            self.assertExpectedInline(
                result,
                """\
  File "<source_path>", line 1
    value = (
            ~
        foo
        ~~~
        + bar
        ^~~~~
    )
    ~
""",
            )

    def test_vt_source_location_set_during_tracing(self):
        # [FUNCTION DOC] Clears the capture dict, compiles a function that captures the source
        # location of local y at a comptime point, then checks it was recorded with this
        # file's name (pyc mapped to py) and a non-None line number.
        _source_location_capture.clear()

        def fn(x):
            # [FUNCTION DOC] HELPER: computes y as x plus one, triggers the capture, returns y.
            y = x + 1
            comptime(_capture_y_source_location)
            return y

        torch.compile(fn, backend="eager")(torch.ones(3))

        source_location = _source_location_capture.get("source_location")
        self.assertIsNotNone(source_location)
        source_location = cast(SourceLocation, source_location)
        self.assertEqual(source_location.filename, __file__.replace(".pyc", ".py"))
        self.assertIsNotNone(source_location.lineno)

    @make_logging_test(graph_breaks=True)
    def test_graph_break_source_attribution_on_stack(self, records):
        # [FUNCTION DOC] Compiles a function with a graph break while a value sits on the
        # stack and asserts the logged graph break message contains the "Stack variable source
        # attribution" section.
        def fn(x):
            # [FUNCTION DOC] HELPER: builds a tuple of x plus one and a graph break call result
            # and returns the first element.
            return (x + 1, torch._dynamo.graph_break())[0]  # noqa: GB_REGISTRY

        torch.compile(fn, backend="eager")(torch.ones(3))

        record = self.getRecord(records, "Graph break in user code")
        self.assertIn("Stack variable source attribution", record.getMessage())


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
