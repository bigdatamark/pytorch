# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo graph break and error messages are well formed. Each test
# compiles a tiny function with the eager backend, usually requesting a full graph, and
# compares the raised Unsupported (or TorchRuntimeError) text, or a captured graph break log
# record, against an inline expected string after normalizing addresses, file names and line
# numbers. Also covers traceback shape, verbose bytecode traces, resume-function frames, loop
# and context manager graph breaks, HOP errors and recompile limit messages. No import-time
# side effects beyond defining a module level logger subclass instance.

import asyncio
import logging
import re
import sys
import traceback
import unittest
import unittest.mock
import warnings
from functools import lru_cache

import torch
import torch._dynamo
import torch._dynamo.config
import torch._dynamo.test_case
import torch.utils._pytree as python_pytree
from torch._dynamo.exc import ResumePrologueTracingError, TorchRuntimeError, Unsupported
from torch._dynamo.testing import skipIfNotPy312, skipIfOnlyNotPy312
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import IS_FBCODE, munge_exc
from torch.testing._internal.logging_utils import LoggingTestCase, make_logging_test


"""
NOTE Adding tests to this file:

It is good practice to add a minimal repro for each graph break site (i.e. `unimplemented()` call
to make sure that there aren't any errors that occur when generating graph break messages.

If a graph break message test fails because the graph break no longer repros,
it is good practice to find a new minimal repro that causes the graph break.
If this is too much work, it is likely safe to skip/remove the test, assuming
it was previously passing and the graph break message is not changed.
However, if you add a new graph break or modify a graph break message, you should
make sure that there is a test for it.
"""


class GenericCtxMgr:
    # [CLASS DOC] HELPER: Minimal do-nothing context manager used to trigger graph breaks
    # inside a generic (unsupported-for-graph-break) context manager. Holds no state.
    def __enter__(self):
        # [FUNCTION DOC] HELPER: Enter hook; returns the manager itself.
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC] HELPER: Exit hook; ignores the exception info and does nothing, so it
        # never suppresses exceptions.
        pass


class WarningOnceLogger(logging.Logger):
    # [CLASS DOC] HELPER: Logger subclass that adds a once-style warning method, used to check
    # the message for logger methods Dynamo does not know about. No extra state.
    def warning_once(self, msg):
        # [FUNCTION DOC] HELPER: Forwards the message to the ordinary warning method (no real
        # once-only deduplication is done).
        self.warning(msg)


warning_once_logger = WarningOnceLogger("warning_once_logger")


class ErrorMessagesTest(LoggingTestCase):
    # [CLASS DOC] Test case (logging-capable) holding all graph break message
    # tests. Most tests are self contained: they define a small function, compile
    # it, and check the exact message.
    def test_inplace_view_on_graph_input_hint(self):
        # [FUNCTION DOC] Checks the message and hints when an in-place view op (transpose in
        # place) is applied to a graph input under a full graph compile: raises Unsupported
        # with the delayed function-call explanation.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; transposes its argument in place and
            # returns it.
            x.transpose_(1, 2)
            return x

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                torch.randn(2, 3, 4)
            ),
            """\
Unsupported function call (delayed)
  Explanation: Dynamo determined that a graph break should occur when calling `L['x'].transpose_`. Reason: Getting an inplace view on a graph input is not supported
  Hint: Avoid mutating a graph input's tensor metadata with in-place view ops. If the mutation is only needed inside the compiled region, replace the in-place call with an out-of-place view, for example `x = x.transpose(1, 2)` instead of `x.transpose_(1, 2)`.
  Hint: If you need to mutate the input tensor's metadata, move the in-place view call outside `torch.compile`.

  Developer debug context: source: AttrSource(base=LocalSource(local_name='x', is_input=True, dynamism=None, is_derefed_cell_contents=False), member='transpose_')

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html

from user code:
   File "test_error_messages.py", line N, in fn
    x.transpose_(1, 2)""",
        )

    def test_dynamic_shape_operator_no_meta_kernel(self):
        # [FUNCTION DOC] Checks the message for a dynamic-output-shape operator without a meta
        # kernel (least squares solve), with scalar output capture enabled via config patch.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; returns least squares of two random
            # square matrices.
            return torch.linalg.lstsq(torch.rand(10, 10), torch.rand(10, 10))

        with torch._dynamo.config.patch(capture_dynamic_output_shape_ops=True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
                """\
Dynamic shape operator (no meta kernel)
  Explanation: Operator `aten.linalg_lstsq.default` does not have a meta kernel that supports dynamic output shapes
  Hint: Please report an issue to PyTorch

  Developer debug context: aten.linalg_lstsq.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0037.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.linalg.lstsq(torch.rand(10, 10), torch.rand(10, 10))""",
            )

    def test_data_dependent_operator2(self):
        # [FUNCTION DOC] Checks the "data dependent operator" message for a tensor
        # equality call whose non-tensor result depends on tensor data (scalar
        # output capture is enabled).
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; returns whether the input equals itself.
            return torch.equal(x, x)

        with torch._dynamo.config.patch(capture_scalar_outputs=True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                    torch.ones(3)
                ),
                """\
Data dependent operator
  Explanation: Operator `aten.equal.default` has a non-Tensor output whose value is dependent on the data of Tensor inputs.
  Hint: Consider wrapping the operator into a PyTorch-understood custom operator (see https://pytorch.org/tutorials/advanced/custom_ops_landing_page.html)

  Developer debug context: aten.equal.default

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0033.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.equal(x, x)""",
            )

    def test_sort_with_nonconstant_keys(self):
        # [FUNCTION DOC] Checks the message when sorting a list of tensors, whose keys are not
        # compile-time constants.
        lst = [
            torch.tensor(4),
            torch.tensor(1),
            torch.tensor(2),
            torch.tensor(3),
        ]

        def fn(lst):
            # [FUNCTION DOC] HELPER: Compiled target; returns the sorted version
            # of its list argument.
            return sorted(lst)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(lst),
            """\
sort with non-constant keys
  Explanation: Cannot perform sort whose key comparison is not a compile-time constant. Key type: <class 'torch.Tensor'>. Most notably, we cannot sort with Tensor or SymInt keys, but we can sort ints.
  Hint: Use something else as the key.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: LazyVariableTracker(realized: TensorVariable())

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0207.html

from user code:
   File "test_error_messages.py", line N, in fn
    return sorted(lst)""",
        )

    def test_super_call_method(self):
        # [FUNCTION DOC] Checks the "C-implemented special method without model" message when
        # iterating a zip object passed as input.
        def fn(it):
            # [FUNCTION DOC] HELPER: Compiled target; builds a list with one added to each
            # element of the iterable.
            return [x + 1 for x in it]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                zip(range(5), range(10))
            ),
            """\
C-implemented special method without VariableTracker model
  Explanation: 'zip' implements '__iter__' in C and Dynamo has no model for it.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: name=__iter__, type=zip, attr=<slot wrapper '__iter__' of 'zip' objects>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb9493.html

from user code:
   File "test_error_messages.py", line N, in fn
    return [x + 1 for x in it]""",
        )

    def test_dict_items_input(self):
        # [FUNCTION DOC] Checks the same C-implemented iteration message for a dictionary
        # items view input, where the error appears at the point of creating an iterator.
        def fn(x, items):
            # [FUNCTION DOC] HELPER: Compiled target; makes an iterator over the items, takes
            # the next element and returns it with the sine of the tensor input.
            it = iter(items)
            return next(it), x.sin()

        x = torch.randn(3)
        dct = {"a": 3, "b": 3}

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(x, dct.items()),
            """\
C-implemented special method without VariableTracker model
  Explanation: 'dict_items' implements '__iter__' in C and Dynamo has no model for it.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: name=__iter__, type=dict_items, attr=<slot wrapper '__iter__' of 'dict_items' objects>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb9493.html

from user code:
   File "test_error_messages.py", line N, in fn
    it = iter(items)""",
        )

    def test_super_call_function(self):
        # [FUNCTION DOC] Checks the observed exception message produced when the
        # compiled code calls a zip object as if it were a function (raises a type
        # error that is not caught).
        def fn(it):
            # [FUNCTION DOC] HELPER: Compiled target; calls its argument and adds
            # one to each result.
            return [x + 1 for x in it()]

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                zip(range(5), range(10))
            ),
            """\
Observed exception
  Explanation: Dynamo found no exception handler at the top-level compiled function when encountering an exception. Exception will propagate outside the compiled region.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: raised exception TypeError("'zip' object is not callable")

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0088.html

from user code:
   File "test_error_messages.py", line N, in fn
    return [x + 1 for x in it()]""",
        )

    def test_unsupported_context(self):
        # [FUNCTION DOC] Checks the "unsupported context manager" message when an integer is
        # used in a with statement.
        def fn(obj):
            # [FUNCTION DOC] HELPER: Compiled target; enters its argument as a context manager
            # and returns one.
            with obj:
                return 1

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(3),
            """\
Unsupported context manager
  Explanation: Dynamo does not know how to enter a `int` context manager.
  Hint: Avoid using the unsupported context manager.
  Hint: If the context manager seems like it should be supported (e.g. torch.set_grad_enabled), then it may be the case that it was created outside the compiled region, which Dynamo does not support. Supported context managers can cross graph break boundaries only if they are local non-closure variables, or are intermediate values.
  Hint: File an issue to PyTorch. Simple context managers can potentially be supported, but note that context managers can't be supported in general

  Developer debug context: Attempted SETUP_WITH/BEFORE_WITH/LOAD_SPECIAL on LazyVariableTracker(unrealized: <class 'int'>)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0142.html

from user code:
   File "test_error_messages.py", line N, in fn
    with obj:""",
        )

    def test_backend_fake_tensor_exc(self):
        # [FUNCTION DOC] Checks the backend compiler exception message when a custom backend
        # raises the fake tensor unsupported exception; the backend name and traceback appear
        # in it.
        def bad_backend(gm, ex):
            # [FUNCTION DOC] HELPER: Backend that ignores the graph and inputs and raises the
            # fake tensor unsupported exception with the text test.
            raise torch._subclasses.fake_tensor.UnsupportedFakeTensorException("test")

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; adds one to the input.
            return x + 1

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend=bad_backend, fullgraph=True)(
                torch.ones(3, 3)
            ),
            """\
Backend compiler exception
  Explanation: Backend compiler `bad_backend` failed with test. Adding a graph break.
  Hint: Report an issue to the backend compiler repo.

  Developer debug context: Backend: bad_backend
    Exception:test
    Traceback:
      File "test_error_messages.py", line N, in fn
        return x + 1


 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0219.html""",
        )

    def test_unsupported_builtin(self):
        # [FUNCTION DOC] Checks the message for a builtin that Dynamo cannot trace (print of a
        # string constant).
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; prints a string.
            print("abc")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Failed to trace builtin operator
  Explanation: Dynamo does not know how to trace builtin operator `print` with argument types ['str'] (has_kwargs False)
  Hint: Avoid calling builtin `print` with argument types ['str']. Consider using an equivalent alternative function/method to `print`.
  Hint: If you are attempting to call a logging function (e.g. `print`), you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.
  Hint: Please report an issue to PyTorch.

  Developer debug context: builtin print [<class 'torch._dynamo.variables.constant.ConstantVariable'>] False

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0059.html

from user code:
   File "test_error_messages.py", line N, in fn
    print("abc")""",
        )

    def test_skipfile_call(self):
        # [FUNCTION DOC] Checks the message when the compiled code calls a function from a
        # file the skip rules mark as skipped (the unittest skip decorator factory). The file
        # path in the message is normalized by a post-processing function.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; returns the result of the
            # unittest skip factory.
            return unittest.skip("test")

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces the machine specific unittest case file path in
            # the text with a short fixed name.
            return re.sub(r"file `.*case\.py`", "file `case.py`", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `skip` in file `case.py` should not be traced.
  Hint: Avoid calling the function `skip`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `skip` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: module: unittest.case, qualname: skip, skip reason: file is in unittest directory

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "test_error_messages.py", line N, in fn
    return unittest.skip("test")""",
            post_munge=post_munge,
        )

    def test_skipfile_dynamo_call(self):
        # [FUNCTION DOC] Checks the message for a call to the dynamo disable API inside a
        # compiled region.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; calls the dynamo disable API
            # with no arguments.
            torch._dynamo.disable()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.disable()`
  Explanation: `torch._dynamo.disable()` was called inside a compiled region. This API disables compilation when used as a decorator or wrapper outside the compiled region.
  Hint: Move the `torch._dynamo.disable()` call outside the compiled function and apply it to the function that should run eagerly.
  Hint: Use `torch._dynamo.graph_break()` to intentionally insert a graph break at this point.

  Developer debug context: Called `torch._dynamo.disable()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0376.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.disable()""",
        )

    def test_skipfile_compiler_disable_call(self):
        # [FUNCTION DOC] Checks the message for a call to the compiler-level disable API
        # inside a compiled region.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; calls the compiler disable
            # API with no arguments.
            torch.compiler.disable()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch.compiler.disable()`
  Explanation: `torch.compiler.disable()` was called inside a compiled region. This API disables compilation when used as a decorator or wrapper outside the compiled region.
  Hint: Move the `torch.compiler.disable()` call outside the compiled function and apply it to the function that should run eagerly.
  Hint: Use `torch._dynamo.graph_break()` to intentionally insert a graph break at this point.

  Developer debug context: Called `torch.compiler.disable()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0377.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch.compiler.disable()""",
        )

    def test_skipfile_dynamo_disable_wrapped_method(self):
        # [FUNCTION DOC] Checks the message when inlining a method that was wrapped by the
        # dynamo disable decorator; memory addresses in the text are normalized.
        class Scheduler:
            # [CLASS DOC] HELPER: Local class with one method wrapped by the
            # dynamo disable decorator.
            @torch._dynamo.disable
            def index_for_timestep(self, x):
                # [FUNCTION DOC] HELPER: Decorated method; returns its argument plus one.
                # Dynamo refuses to inline it.
                return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; creates a Scheduler and calls the
            # disabled method.
            return Scheduler().index_for_timestep(x)

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces hexadecimal addresses in the text
            # with a placeholder.
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
Skip inlining `torch.compiler.disable()`d function
  Explanation: Skip inlining function <function ErrorMessagesTest.test_skipfile_dynamo_disable_wrapped_method.<locals>.Scheduler.index_for_timestep at 0xmem_addr> since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_skipfile_dynamo_disable_wrapped_method.<locals>.Scheduler.index_for_timestep at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0099.html

from user code:
   File "test_error_messages.py", line N, in fn
    return Scheduler().index_for_timestep(x)""",
            post_munge=post_munge,
        )

    def test_skipfile_inline(self):
        # [FUNCTION DOC] Checks the message when Dynamo tries to inline a function
        # from a skipped file reached through a class attribute (the unittest skip
        # factory); path is normalized.
        class Foo:
            # [CLASS DOC] HELPER: Local class whose attribute is the unittest skip
            # factory function.
            fn = unittest.skip

        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; calls the skip factory
            # through a new instance.
            Foo().fn()

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces the machine specific case file path in backticks
            # with a short fixed name.
            return re.sub(r"`.*case\.py`", "`case.py`", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to inline function marked as skipped
  Explanation: Dynamo developers have intentionally marked that the function `skip` should not be traced.
  Hint: Avoid calling the function `skip`.
  Hint: Apply `@torch._dynamo.dont_skip_tracing` to the function `skip` to force tracing into the function. More graph breaks may occur as a result of attempting to trace into the function.
  Hint: Please file an issue to PyTorch.

  Developer debug context: qualname: skip, name: skip, filename: `case.py`, skip reason: file is in unittest directory

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0008.html

from user code:
   File "test_error_messages.py", line N, in fn
    Foo().fn()""",
            post_munge=post_munge,
        )

    def test_dynamo_graph_break_fn(self):
        # [FUNCTION DOC] Checks the message for an explicit dynamo graph break
        # call with no message.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; inserts a graph break.
            torch._dynamo.graph_break()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()""",
        )

    def test_dynamo_graph_break_fn_with_msg(self):
        # [FUNCTION DOC] Checks the message for an explicit graph break call that
        # supplies a message keyword; the message text appears in the explanation
        # and the keyword in the context.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; inserts a graph break with a
            # custom message.
            torch._dynamo.graph_break(msg="test graph break")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: test graph break
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{'msg': ConstantVariable(str: 'test graph break')}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break(msg="test graph break")""",
        )

    def test_warnings(self):
        # [FUNCTION DOC] Checks the message when the compiled code calls the standard warnings
        # warn function, which is an untraceable C builtin.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; emits a warning.
            warnings.warn("test")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo does not know how to trace the Python builtin `_warnings.warn`.
  Hint: If you are attempting to call a logging function (e.g. `_warnings.warn`), you can try adding it to `torch._dynamo.config.reorderable_logging_functions`.
  Hint: Please file an issue on GitHub so the PyTorch team can add support for it.

  Developer debug context: module: _warnings, qualname: warn, skip reason: cannot determine source file for _warnings (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html

from user code:
   File "test_error_messages.py", line N, in fn
    warnings.warn("test")""",
        )

    def test_logger_method_message(self):
        # [FUNCTION DOC] Checks the message for calling a method on a Logger subclass instance
        # that is not registered as reorderable or ignored. Object reprs and qualified names
        # are normalized by a post-processing function before comparison.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; calls the module level logger's once-
            # style method.
            warning_once_logger.warning_once("test")

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Rewrites the method description and the fully qualified
            # method name in the text to short stable names.
            s = re.sub(
                r"method: <WarningOnceLogger .*?>\.warning_once",
                "method: <WarningOnceLogger>.warning_once",
                s,
            )
            logger_method_name = (
                f"{WarningOnceLogger.__module__}."
                f"{WarningOnceLogger.__qualname__}.warning_once"
            )
            return s.replace(
                f"`{logger_method_name}`", "`WarningOnceLogger.warning_once`"
            )

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
logging.Logger method not supported for non-export cases
  Explanation: For non-export cases, logging.Logger methods are only supported if the logger has a source and the method is registered as reorderable.
  Hint: If you do not need this logging side effect, add the exact method being called to `torch._dynamo.config.ignore_logging_functions`. Dynamo will skip the call and return `None`.
  Hint: For example, for `logger.warning_once(...)`, use `torch._dynamo.config.ignore_logging_functions.add(logger.warning_once)`. If `warning_once` is defined on the logger class, add the class method `WarningOnceLogger.warning_once` to ignore this method for all instances of that class.
  Hint: Dynamo does not trace into logging.Logger method bodies, so only the method you call directly (`warning_once`) is checked against the ignore set. Ignoring a method that `warning_once` calls internally has no effect.
  Hint: If you need the log side effect to run, then you can try one of (1) create the logger outside the compiled region and add the method to `torch._dynamo.config.reorderable_logging_functions` (e.g. `torch._dynamo.config.reorderable_logging_functions.add(logger.warning_once)`) so that it runs after the compiled region, as long as it is called without kwargs and its arguments are tensors, constants, or string formatters, (2) `torch._higher_order_ops.print(...)`, (3) wrap the logging call in a custom op (marked as mutable), or (4) preserve the logging contents and move the logging call outside the compiled region.

  Developer debug context: method: <WarningOnceLogger>.warning_once, args: [ConstantVariable(str: 'test')], kwargs: {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0291.html

from user code:
   File "test_error_messages.py", line N, in fn
    warning_once_logger.warning_once("test")""",
            post_munge=post_munge,
        )

    @unittest.skipIf(not python_pytree._cxx_pytree_exists, "missing optree package")
    def test_optree_graph_break_message(self):
        # [FUNCTION DOC] Skipped unless the C++ optree pytree backend exists. Compiles one
        # function using an optree tree map and asserts no graph break is counted, then
        # compiles another using an optree deque spec and asserts a graph break is recorded;
        # compares the first recorded graph break key to the expected skipped-function text
        # after path normalization.
        import optree

        @torch.compile(backend="eager")
        def fn1(x):
            # [FUNCTION DOC] HELPER: Compiled target; transposes a nested tree and returns
            # sine and cosine trees.
            tree = {"a": x, "b": (x - 1, 2 * x)}
            sin, cos = optree.tree_transpose_map(
                lambda t: (torch.sin(t), torch.cos(t)),
                tree,
            )
            return sin, cos

        fn1(torch.randn(4))
        self.assertEqual(len(counters["graph_break"]), 0)

        @torch.compile(backend="eager")
        def fn2(x):
            # [FUNCTION DOC] HELPER: Compiled target; builds an optree deque tree spec and
            # returns it with the input.
            spec = optree.treespec_deque([])
            return spec, x

        fn2(torch.randn(4))
        self.assertGreaterEqual(len(counters["graph_break"]), 1)
        first_graph_break = next(iter(counters["graph_break"].keys()))

        def post_munge(string):
            # [FUNCTION DOC] HELPER: Replaces the module path prefix of the make-from-
            # collection name with a placeholder.
            return re.sub(
                r"(optree\.|qualname: )\S*(\.make_from_collection)",
                r"\1<path>\2",
                string,
            )

        self.assertExpectedInline(
            post_munge(first_graph_break),
            """\
Attempted to call function marked as skipped
  Explanation: Dynamo cannot trace optree C/C++ function optree.<path>.make_from_collection.
  Hint: Consider using torch.utils._pytree - https://github.com/pytorch/pytorch/blob/main/torch/utils/_pytree.py

  Developer debug context: module: optree._C, qualname: <path>.make_from_collection, skip reason: cannot determine source file for optree._C (likely a C extension or builtin)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0007.html""",
        )

    def test_unpack_sequence_with_wrong_length(self):
        # [FUNCTION DOC] Checks that unpacking a one element list into two names is reported
        # as an observed exception (value error) for a full graph compile.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; unpacks a one element list
            # into two variables.
            a, b = [1]  # noqa: F841

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(
                torch.randn(
                    2,
                )
            ),
            """\
Observed exception
  Explanation: Dynamo found no exception handler at the top-level compiled function when encountering an exception. Exception will propagate outside the compiled region.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: raised exception ValueError('not enough values to unpack (expected 2, got 1)')

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0088.html

from user code:
   File "test_error_messages.py", line N, in fn
    a, b = [1]  # noqa: F841""",
        )

    def test_unpack_sequence_with_non_iterable(self):
        # [FUNCTION DOC] Checks that unpacking an integer into two names is reported as an
        # observed type error.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; unpacks an integer into two variables.
            a, b = 1  # noqa: RUF100

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(
                torch.randn(
                    2,
                )
            ),
            """\
Observed exception
  Explanation: Dynamo found no exception handler at the top-level compiled function when encountering an exception. Exception will propagate outside the compiled region.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: raised exception TypeError("'int' object is not iterable")

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0088.html

from user code:
   File "test_error_messages.py", line N, in fn
    a, b = 1  # noqa: RUF100""",
        )

    def test_unpack_ex_failure(self):
        # [FUNCTION DOC] Checks that star-unpacking an empty list needing at least one element
        # is reported as an observed value error.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; star-unpacks an empty list.
            a, *b = []  # noqa: F841

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: fn(torch.randn(2)),
            """\
Observed exception
  Explanation: Dynamo found no exception handler at the top-level compiled function when encountering an exception. Exception will propagate outside the compiled region.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: raised exception ValueError('not enough values to unpack (expected at least 1, got 0)')

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0088.html

from user code:
   File "test_error_messages.py", line N, in fn
    a, *b = []  # noqa: F841""",
        )

    def test_observed_exception(self):
        # [FUNCTION DOC] Checks the observed exception message when the compiled function
        # raises a runtime error with no handler.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; raises a runtime error.
            raise RuntimeError("test")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Observed exception
  Explanation: Dynamo found no exception handler at the top-level compiled function when encountering an exception. Exception will propagate outside the compiled region.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: raised exception RuntimeError('test')

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0088.html

from user code:
   File "test_error_messages.py", line N, in fn
    raise RuntimeError("test")""",
        )

    def test_uninitialized_module(self):
        # [FUNCTION DOC] Checks the message when a module whose constructor never called the
        # parent constructor is called inside a compiled function.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: Module subclass whose constructor is deliberately a no-op,
            # leaving it uninitialized.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Does nothing, so the parent init is skipped.
                pass

        def fn(mod):
            # [FUNCTION DOC] HELPER: Compiled target; calls the module with the constant one.
            return mod(1)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(Foo()),
            """\
Uninitialized nn.Module
  Explanation: Attempted to trace an uninitialized nn.Module of type Foo.
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: Ensure your nn.Module instance has called `super().__init__()`.

  Developer debug context: Foo

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0119.html

from user code:
   File "test_error_messages.py", line N, in fn
    return mod(1)""",
        )

    def test_generic_ctx_mgr_graph_break_fullgraph_true(self):
        # [FUNCTION DOC] Checks the plain graph break message for a graph break inside a
        # generic context manager when a full graph is required.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; inserts a graph break inside a generic
            # context manager block.
            with GenericCtxMgr():
                torch._dynamo.graph_break()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()""",
        )

    @make_logging_test(graph_breaks=True)
    def test_generic_ctx_mgr_graph_break_fullgraph_false(self, records):
        # [FUNCTION DOC] Compiles without the full graph requirement and checks the single
        # logged graph break record: the original graph break followed by the second break
        # caused by active generic context managers (two nested ones active at that point).
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; uses sequential and nested generic
            # context managers and inserts a graph break while two of them are active.
            with GenericCtxMgr():
                with GenericCtxMgr():
                    pass
                with GenericCtxMgr():
                    with GenericCtxMgr():
                        pass
                    torch._dynamo.graph_break()

        torch.compile(fn, backend="eager")()
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_generic_ctx_mgr_graph_break_fullgraph_false
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

    def test_load_build_class(self):
        # [FUNCTION DOC] Checks the message for defining a class inside a compiled function
        # (the class construction builtin is invoked with a non-constant body function).
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; defines a local class and returns it.
            class Foo:
                # [CLASS DOC] HELPER: Empty local class defined inside the compiled function.
                pass

            return Foo

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Invalid call to __build_class__
  Explanation: Cannot trace class definition: the class body function is unsupported or the base class argument are not compile-time constants
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: Non-constant args to __build_class__: (NestedUserFunctionVariable(), ConstantVariable(str: 'Foo')) {}

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb8977.html

from user code:
   File "test_error_messages.py", line N, in fn
    class Foo:""",
        )

    @skipIfNotPy312
    def test_async_return_bytecode_exception_table(self):
        # [FUNCTION DOC] Only runs on Python 3.12 or newer. Checks that compiling
        # an async function that simply returns one works and running it with the
        # event loop yields one.
        async def fn():
            # [FUNCTION DOC] HELPER: Async function returning one.
            return 1

        self.assertEqual(
            asyncio.run(torch.compile(fn, backend="eager", fullgraph=True)()), 1
        )

    @skipIfNotPy312
    def test_unsupported_bytecode(self):
        # [FUNCTION DOC] Only runs on Python 3.12 or newer. Checks the missing bytecode
        # handler message for an async iteration loop (the async iterator getter instruction);
        # addresses and the instruction repr are normalized.
        async def fn():
            # [FUNCTION DOC] HELPER: Async function with an async for loop over a range that
            # prints each item and returns one.
            async for i in range(3):
                print(i)
            return 1

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces hex addresses and the full instruction repr for
            # the async iterator getter with short placeholders; returns the cleaned text.
            s = re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)
            s = re.sub(
                r"Instruction\(.*opname='GET_AITER'.*\)\n",
                "Instruction(GET_AITER)",
                s,
            )
            return s

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Missing bytecode handler
  Explanation: Dynamo does not know how to handle the bytecode instruction `GET_AITER`.
  Hint: Do not trace code that produces the `GET_AITER` bytecode instruction (see https://docs.python.org/3/library/dis.html for bytecode semantics).
  Hint: It may be possible to write Dynamo tracing rules for this code. Please report an issue to PyTorch if you encounter this graph break often and it is causing performance issues.

  Developer debug context: GET_AITER with args (<torch._dynamo.symbolic_convert.InstructionTranslator object at 0xmem_addr>, Instruction(GET_AITER)
 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0082.html

from user code:
   File "test_error_messages.py", line N, in fn
    async for i in range(3):""",
            post_munge=post_munge,
        )

    def test_reconstruction_failure(self):
        # [FUNCTION DOC] Checks the reconstruction failure message when a compiled function
        # returns a bound method of an object created inside it; addresses are normalized.
        class Foo:
            # [CLASS DOC] HELPER: Local class with one trivial method.
            def meth(self):
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; returns a bound method of a
            # fresh instance.
            return Foo().meth

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces hex addresses with a placeholder.
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(),
            """\
Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

from user code:
   File "test_error_messages.py", line N, in fn
    return Foo().meth""",
            post_munge=post_munge,
        )

    @make_logging_test(graph_breaks=True)
    def test_reconstruction_failure_gb(self, records):
        # [FUNCTION DOC] Compiles without full graph and checks two log records: the first
        # graph break for an explicit break, and the second saying the graph break could not
        # be handled gracefully due to a reconstruction failure of a sourceless bound method.
        class Foo:
            # [CLASS DOC] HELPER: Local class with one trivial method.
            def meth(self):
                # [FUNCTION DOC] HELPER: Returns zero.
                return 0

        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; keeps a bound method of a new object
            # across an explicit graph break and returns it.
            f = Foo().meth
            torch._dynamo.graph_break()
            return f

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces hex addresses with a placeholder.
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        torch.compile(fn, backend="eager")()

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_reconstruction_failure_gb
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Reconstruction failure
  Explanation: Dynamo has no bytecode reconstruction implemented for sourceless variable UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure_gb.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo)).
  Hint: If Dynamo is attempting to trace a return statement and your code is attempting to return a variable that Dynamo cannot reconstruct, then remove it from the return statement.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.
  Hint: Report an issue to PyTorch if you need reconstrtuction support. Note that objects that don't have reconstruction rules may be fundamentally unreconstructable.

  Developer debug context: UserMethodVariable(<function ErrorMessagesTest.test_reconstruction_failure_gb.<locals>.Foo.meth at 0xmem_addr>, UserDefinedObjectVariable(Foo))

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0092.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_reconstruction_failure_gb
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

    def test_faketensor_nyi(self):
        # [FUNCTION DOC] Registers a custom op and a fake implementation that raises not
        # implemented error, then checks the message for running the op under fake tensors
        # during compile.
        op_name = "mylib::error_messages_faketensor"

        @torch.library.custom_op(op_name, mutates_args=())
        def foo(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Custom op implementation; returns the sine of the input.
            return x.sin()

        @foo.register_fake
        def _(x):
            # [FUNCTION DOC] HELPER: Fake implementation; always raises not-implemented.
            raise NotImplementedError

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls the registered custom op.
            return torch.ops.mylib.error_messages_faketensor(x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
NotImplementedError/UnsupportedFakeTensorException when running FX node
  Explanation: Dynamo failed to run FX node with fake tensors: call_function mylib.error_messages_faketensor(*(FakeTensor(..., size=(3,)),), **{}): got NotImplementedError()
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.
  Hint: If the op is a custom op, did you implement a fake tensor implementation? (e.g. with `@my_custom_op.register_fake`)
  Hint: If the op is a PyTorch op, please file an issue to PyTorch.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0087.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.ops.mylib.error_messages_faketensor(x)""",
        )

    def test_fx_node_error_bad_user_code(self):
        # [FUNCTION DOC] Checks that adding tensors of lengths three and four
        # raises the runtime error type with a readable message about fake tensor
        # call failure and broadcasting.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled target; adds its two arguments.
            return x + y

        self.assertExpectedInlineMunged(
            TorchRuntimeError,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(
                torch.randn(3), torch.randn(4)
            ),
            """\
RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <built-in function add>(*(FakeTensor(..., size=(3,)), FakeTensor(..., size=(4,))), **{}): got RuntimeError('Attempting to broadcast a dimension of length 4 at -1! Mismatching argument at index 1 had torch.Size([4]); but expected shape should be broadcastable to [3]')
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html

from user code:
   File "test_error_messages.py", line N, in fn
    return x + y""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_fx_node_error_cross_device(self):
        # [FUNCTION DOC] Skipped without CUDA. Builds a linear layer on the GPU,
        # calls it with a CPU input in a compiled function and checks the raised
        # runtime error text mentions the device mismatch and both devices and
        # does not contain internal fake tensor wording.
        linear = torch.nn.Linear(10, 20, device="cuda").eval()

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; applies the GPU linear layer.
            return linear(x)

        with self.assertRaises(TorchRuntimeError) as cm:
            torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(1, 10))

        msg = str(cm.exception)
        self.assertIn("Tensor device mismatch", msg)
        self.assertIn("Expected all tensors to be on the same device", msg)
        self.assertIn("cpu", msg)
        self.assertIn("cuda", msg)
        self.assertNotIn("Dynamo failed to run FX node with fake tensors", msg)
        self.assertNotIn("Unhandled FakeTensor Device Propagation", msg)

    def test_data_dependent_branching_fullgraph(self):
        # [FUNCTION DOC] Checks the data-dependent branching message including the printed
        # computation that forms the branch condition.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; branches on whether the
            # tensor sum is positive.
            if x.sum() > 0:
                return x.sin()
            return x.cos()

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "test_error_messages.py", line N, in fn, code: if x.sum() > 0:
        gt = gt(sum_1, 0)

  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

from user code:
   File "test_error_messages.py", line N, in fn
    if x.sum() > 0:""",
        )

    def test_data_dependent_branching_bool_tensor_hints(self):
        # [FUNCTION DOC] Checks the data-dependent branching hints for the overflow-clamping
        # pattern, including the hint about logical or between tensor expressions.
        def cast_overflow_tensors(tensors, offset=1000):
            # [FUNCTION DOC] HELPER: Compiled target; if any element is infinite or NaN,
            # clamps the tensor to the dtype maximum minus an offset; returns the tensor.
            if tensors.isinf().any() or tensors.isnan().any():
                clamp_value = torch.finfo(tensors.dtype).max - offset
                tensors = torch.clamp(tensors, min=-clamp_value, max=clamp_value)
            return tensors

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(
                cast_overflow_tensors, backend="eager", fullgraph=True
            )(torch.randn(3)),
            """\
Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.
  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: Note: Python `or`/`and` between tensor expressions (e.g. `tensor.any() or other_tensor.any()`) triggers implicit bool conversion. Use `torch.logical_or`/`torch.logical_and` or the `|`/`&` operators instead.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

from user code:
   File "test_error_messages.py", line N, in cast_overflow_tensors
    if tensors.isinf().any() or tensors.isnan().any():""",
        )

    # Test that the bytecode source attribution is correct with VariableTracker
    @make_logging_test(trace_bytecode=True)
    def test_variable_tracker_source_attribution(self, records):
        # [FUNCTION DOC] Enables bytecode trace logging, runs a compiled function twice
        # calling an inner function, collects all log lines mentioning lazy variable trackers,
        # and asserts at least one exists, the first describes an unrealized function and the
        # fourth an unrealized tensor.
        def inner(x):
            # [FUNCTION DOC] HELPER: Adds one to the input.
            return x + 1

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; applies the inner function twice.
            x = inner(x)
            return inner(x)

        fn(torch.ones(3))

        def find_trace_bytecode_lines(long_string):
            # [FUNCTION DOC] HELPER: Splits the text into lines and returns the list of lines
            # that contain a lazy variable tracker description.
            # Split the string into lines
            lines = long_string.split("\n")
            # More comprehensive pattern to capture LazyVariableTracker info
            pattern = r"LazyVariableTracker\([^)]*\)"
            # Find all lines containing the pattern
            result = [line for line in lines if re.search(pattern, line)]
            return result

        # Get all log messages, not just the last one
        all_messages = []
        for record in records:
            msg = munge_exc(record.getMessage(), skip=0)

            all_messages.append(msg)

        # Combine all messages to search through
        combined_msg = "\n".join(all_messages)
        all_lines = find_trace_bytecode_lines(combined_msg)

        # For now, just check that we found some lines with LazyVariableTracker
        self.assertGreater(
            len(all_lines), 0, "Should find at least one LazyVariableTracker line"
        )

        self.assertIn(
            "LazyVariableTracker(unrealized: <class 'function'>)", all_lines[0]
        )
        self.assertIn(
            "LazyVariableTracker(unrealized: <class 'torch.Tensor'>)", all_lines[3]
        )

    @make_logging_test(graph_breaks=True)
    def test_data_dependent_branching_gb(self, records):
        # [FUNCTION DOC] Compiles without full graph and compares the first logged graph break
        # record to the expected data-dependent branching text with user traceback.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; branches on the tensor sum and returns
            # sine or cosine.
            if x.sum() > 0:
                return x.sin()
            return x.cos()

        torch.compile(fn, backend="eager")(torch.randn(3))

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "test_error_messages.py", line N, in fn, code: if x.sum() > 0:
        gt = gt(sum_1, 0)

  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

User code traceback:
  File "test_error_messages.py", line N, in test_data_dependent_branching_gb
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    if x.sum() > 0:
""",
        )

    @unittest.skipIf(IS_FBCODE, "assert gets patched in internal pytest")
    @make_logging_test(graph_breaks=True)
    def test_assert_failure_in_generic_ctx_mgr(self, records):
        # [FUNCTION DOC] Skipped in FBCODE where asserts are rewritten. Checks that a failing
        # assert inside a generic context manager raises an assertion error to the caller and
        # logs exactly one graph break record about a data-dependent assertion.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; asserts inside a generic context manager
            # that the argument is none, which fails.
            with GenericCtxMgr():
                assert x is None  # noqa: S101

        with self.assertRaises(AssertionError):
            torch.compile(fn, backend="eager")(torch.randn(3))

        # only 1 graph break message
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent assertion failed (cannot compile partial graph)
  Explanation: Dynamo has determined when encountering a data-dependent assert failure that it should not compile the partial graph.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch._assert()` to raise a hard AssertionError when the check fails. This error will propagate back the user code that called the compiled function (i.e. Dynamo will not trace any exception handling).
  Hint: Remove the assert statement.
  Hint: Move the assert statement outside of any context managers in order to graph break with partial graph compilation (if fullgraph=False).

  Developer debug context: value: ConstantVariable(bool: False)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0034.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_assert_failure_in_generic_ctx_mgr
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    assert x is None  # noqa: S101
""",
        )

    def test_no_internal_compiler_stacktrace(self):
        # [FUNCTION DOC] Catches the error raised by a full graph compile
        # manually, formats its traceback, normalizes file names, line numbers and
        # carets, and checks that the traceback shows only the user call and
        # compiler wrapper frames, not internal Dynamo frames.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; calls gn.
            gn()

        def gn():
            # [FUNCTION DOC] HELPER: Inserts a graph break.
            torch._dynamo.graph_break()

        # assertRaises suppresses the traceback, so manually catch
        e = None
        try:
            torch.compile(fn, backend="eager", fullgraph=True)()
        except Exception as exn:
            e = exn

        self.assertIsNotNone(e)

        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # only keep the filenames in the traceback
        msg = re.sub(r'File ".*\W(\w+\.py)"', 'File "\\1"', msg)
        # remove line numbers
        msg = re.sub(r"line (\d+)", "line N", msg)
        # remove carets
        msg = re.sub(r"\n\s*~*\^+\n", "\n", msg)
        # normalize multi-line raise: 3.10 only shows the first line,
        # 3.11+ shows the full statement. Replace either form with a single line.
        msg = re.sub(
            r"    raise e\.with_traceback\((?:.*\) from e\.__cause__  # User compiler error|[^\n]*)",
            "    raise e.with_traceback(None) from e.__cause__  # User compiler error",
            msg,
            flags=re.DOTALL,
        )
        self.assertExpectedInline(
            msg,
            """\
Traceback (most recent call last):
  File "test_error_messages.py", line N, in test_no_internal_compiler_stacktrace
    torch.compile(fn, backend="eager", fullgraph=True)()
  File "eval_frame.py", line N, in compile_wrapper
    raise e.with_traceback(None) from e.__cause__  # User compiler error
torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    gn()
  File "test_error_messages.py", line N, in gn
    torch._dynamo.graph_break()

Set TORCHDYNAMO_VERBOSE=1 for the internal stack trace (please do this especially if you're reporting a bug to PyTorch). For even more developer context, set TORCH_LOGS="+dynamo"

""",
        )

    @torch._dynamo.config.patch(verbose=True)
    def test_internal_compiler_stacktrace_verbose(self):
        # [FUNCTION DOC] With verbose config patched on, catches the error from a full graph
        # compile and checks that the internal traceback is present (replaced by a
        # placeholder) ahead of the user facing message.
        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; calls gn.
            gn()

        def gn():
            # [FUNCTION DOC] HELPER: Inserts a graph break.
            torch._dynamo.graph_break()

        # assertRaises suppresses the traceback, so manually catch
        e = None
        try:
            torch.compile(fn, backend="eager", fullgraph=True)()
        except Exception as exn:
            e = exn

        self.assertIsNotNone(e)

        msg = "".join(traceback.format_exception(type(e), e, e.__traceback__))
        # only keep the filenames in the traceback
        msg = re.sub(r'File ".*\W(\w+\.py)"', 'File "\\1"', msg)
        # remove line numbers
        msg = re.sub(r"line (\d+)", "line N", msg)
        # remove carets
        msg = re.sub(r"\n\s*~*\^+\n", "\n", msg)
        msg = re.sub(
            r"""(?s)Traceback \(most recent call last\):.*
  File "exc.py", line N, in unimplemented
    raise Unsupported\([^\)]+\)""",
            "<Internal traceback>\n",
            msg,
        )

        self.assertExpectedInline(
            msg,
            """\
<Internal traceback>

torch._dynamo.exc.Unsupported: Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    gn()
  File "test_error_messages.py", line N, in gn
    torch._dynamo.graph_break()

""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_in_loop(self, records):
        # [FUNCTION DOC] Checks the "graph break in loop" message for a loop containing an
        # explicit graph break and then for a loop containing a data-dependent branch; each
        # time exactly one new record is logged and the frame is skipped.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; loops twice inserting a graph break,
            # then adds one.
            for i in range(2):
                torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(3))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: fn (test_error_messages.py line N)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_in_loop
    fn(torch.ones(3))
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

        @torch.compile(backend="eager")
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled target; loops twice choosing between adding one
            # or two based on the tensor sum, then adds four.
            for i in range(2):
                if x.sum() > 0:
                    x = x + 1
                else:
                    x = x + 2
            return x + 4

        gn(torch.ones(3))
        self.assertEqual(len(records), 2)
        self.assertExpectedInline(
            munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "test_error_messages.py", line N, in gn, code: if x.sum() > 0:
        gt = gt(sum_1, 0)

  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: gn (test_error_messages.py line N)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_in_loop
    gn(torch.ones(3))
  File "test_error_messages.py", line N, in gn
    if x.sum() > 0:
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_skip_frame_in_loop_message(self, records):
        # [FUNCTION DOC] Checks the logged message when a data-dependent branch
        # inside a generic context manager within a loop forces skipping the
        # frame; one record is expected.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; loops twice, entering a context manager
            # and conditionally adding one based on the tensor sum.
            for i in range(2):
                with GenericCtxMgr():
                    if x.sum() > 0:
                        x = x + 1
            return x

        torch.compile(fn, backend="eager")(torch.randn(3))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "test_error_messages.py", line N, in fn, code: if x.sum() > 0:
        gt = gt(sum_1, 0)

  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_skip_frame_in_loop_message
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    if x.sum() > 0:
""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_skip_frame_empty_function_message(self, records):
        # [FUNCTION DOC] Uses debug-level dynamo logging to check the message
        # emitted when a function traces no operations and the frame is skipped;
        # the line number is normalized.
        def empty_fn(x):
            # [FUNCTION DOC] HELPER: Compiled target that does nothing.
            pass

        torch.compile(empty_fn, backend="eager")(torch.randn(3))
        skip_messages = [
            r for r in records if "No ops traced for the FX graph." in r.getMessage()
        ]
        self.assertEqual(len(skip_messages), 1)
        msg = munge_exc(skip_messages[0].getMessage(), suppress_suffix=True, skip=0)
        msg = re.sub(r" (\d+)$", r" N", msg, flags=re.MULTILINE)

        self.assertExpectedInline(
            msg,
            """\
skipping: empty_fn (reason: Dynamo decided to skip the frame while tracing: No ops traced for the FX graph. `torch.compile` will skip the frame and fall back to eager.
Frame info: empty_fn (test_error_messages.py line N), file: test_error_messages.py)""",
        )

    @make_logging_test(graph_breaks=True)
    def test_nested_compile_user_frames(self, records):
        # [FUNCTION DOC] Checks the user stack in the last graph break record when the break
        # occurs in nested calls: it lists the test frame and each of the three nested
        # function frames, and reports the later of two consecutive breaks.
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls gn with the input plus one.
            gn(x + 1)

        def gn(x):
            # [FUNCTION DOC] HELPER: Calls hn with the input plus one.
            hn(x + 1)

        def hn(x):
            # [FUNCTION DOC] HELPER: Inserts two consecutive graph breaks.
            torch._dynamo.graph_break()  # 0
            torch._dynamo.graph_break()  # 1

        torch.compile(fn, backend="eager")(torch.randn(3))

        self.assertExpectedInline(
            munge_exc(records[-1].getMessage(), skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_nested_compile_user_frames
    torch.compile(fn, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in fn
    gn(x + 1)
  File "test_error_messages.py", line N, in gn
    hn(x + 1)
  File "test_error_messages.py", line N, in hn
    torch._dynamo.graph_break()  # 1
""",
        )

    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_latest_bytecode_to_graph_break_fullgraph(self, records):
        # [FUNCTION DOC] With verbose config and graph break logging patched on, checks the
        # full graph error message for an explicit break after tensor additions (no bytecode
        # listing appears in the exception).
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; adds tensors, inserts a graph break,
            # returns the sum.
            y = x + 1
            z = x + y
            torch._dynamo.graph_break()
            return z

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
""",
        )

    @skipIfOnlyNotPy312
    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_latest_bytecode_to_graph_break_python_versioning(self, records):
        # [FUNCTION DOC] Only runs on Python 3.12 exactly. With verbose logging,
        # compares the whole first graph break record including the list of recent
        # traced bytecode instructions.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; adds tensors, inserts a graph break,
            # returns the sum.
            y = x + 1
            z = x + y
            torch._dynamo.graph_break()
            return z

        fn(torch.ones(3))

        s = munge_exc(records[0].getMessage(), skip=0)

        self.assertExpectedInline(
            s,
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_latest_bytecode_to_graph_break_python_versioning
    fn(torch.ones(3))

========== most recent `torch.compile` tracing attempt started here ==========

  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.

Most recent bytecode instructions traced (max 20):
TRACE RESUME 0 []
TRACE LOAD_FAST 'x' []
TRACE LOAD_CONST 1 [LazyVariableTracker(unrealized: <class 'torch.Tensor'>)]
TRACE BINARY_OP 0 [LazyVariableTracker(unrealized: <class 'torch.Tensor'>), ConstantVariable(int: 1)]
TRACE STORE_FAST 'y' [TensorVariable()]
TRACE LOAD_FAST 'x' []
TRACE LOAD_FAST 'y' [TensorVariable()]
TRACE BINARY_OP 0 [TensorVariable(), TensorVariable()]
TRACE STORE_FAST 'z' [TensorVariable()]
TRACE LOAD_GLOBAL 'torch' []
TRACE LOAD_ATTR '_dynamo' [LazyVariableTracker(unrealized: <class 'module'>)]
TRACE LOAD_ATTR 'graph_break' [LazyVariableTracker(unrealized: <class 'module'>)]
TRACE PUSH_NULL <class 'torch._dynamo.bytecode_transformation._NotProvided'> [LazyVariableTracker(unrealized: <class 'function'>)]
TRACE SWAP <class 'torch._dynamo.bytecode_transformation._NotProvided'> [LazyVariableTracker(unrealized: <class 'function'>), NullVariable]
TRACE CALL 0 [NullVariable, LazyVariableTracker(unrealized: <class 'function'>)]
""",
        )

    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_latest_bytecode_to_graph_break(self, records):
        # [FUNCTION DOC] With verbose logging, checks that the first record includes the
        # recent bytecode header and between eleven and twenty trace lines.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; adds tensors, inserts a graph break,
            # returns the sum.
            y = x + 1
            z = x + y
            torch._dynamo.graph_break()
            return z

        fn(torch.ones(3))

        pattern = r"TRACE.*"
        s = munge_exc(records[0].getMessage(), skip=0)
        matches = re.findall(pattern, s)
        self.assertEqual((len(matches) > 10), True)
        self.assertEqual((len(matches) <= 20), True)
        self.assertIn("Most recent bytecode instructions traced (max 20):", s)

    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_graph_break_traceback_above_dynamo_shows_user_code(self, records):
        # [FUNCTION DOC] With verbose logging, checks that records for repeated breaks in a
        # plain function, a function with data-dependent branches and an object attribute
        # assignment hook each show the correct source line marker in the traceback, then
        # compares the last record (with trace lines removed and resume function line numbers
        # normalized) to expected text.
        @torch.compile(backend="eager")
        # NOTE: comments in this test are used to differentiate lines!
        def f1(x):
            # [FUNCTION DOC] HELPER: Compiled target; three consecutive graph breaks.
            torch._dynamo.graph_break()  # 0
            torch._dynamo.graph_break()  # 1
            torch._dynamo.graph_break()

        @torch.compile(backend="eager")
        def f2(x):
            # [FUNCTION DOC] HELPER: Compiled target; three consecutive data-dependent
            # branches each adding one.
            if x.sum() > 0:  # 0
                x = x + 1
            if x.sum() > 0:  # 1
                x = x + 1
            if x.sum() > 0:
                x = x + 1

        class Foo:
            # [CLASS DOC] HELPER: Local class whose attribute assignment hook
            # inserts a graph break.
            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: Attribute assignment hook that inserts a graph break.
                torch._dynamo.graph_break()

        @torch.compile(backend="eager")
        def f3(x):
            # [FUNCTION DOC] HELPER: Compiled target; assigns an attribute on
            # three fresh instances.
            Foo().attr = x  # 0
            Foo().attr = x  # 1
            Foo().attr = x

        f1(torch.randn(3))
        self.assertIn("torch._dynamo.graph_break()  # 0", records[-1].getMessage())
        self.assertIn("torch._dynamo.graph_break()  # 1", records[-1].getMessage())
        f2(torch.ones(3))
        self.assertIn("if x.sum() > 0:  # 0", records[-1].getMessage())
        self.assertIn("if x.sum() > 0:  # 1", records[-1].getMessage())
        f3(torch.randn(3))
        self.assertIn("Foo().attr = x  # 0", records[-1].getMessage())
        self.assertIn("Foo().attr = x  # 1", records[-1].getMessage())

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Normalizes resume function line numbers and strips
            # everything from the first trace line onward.
            s = re.sub(
                r"torch_dynamo_resume_in_f(\d)_at_(\d+)",
                r"torch_dynamo_resume_in_f\1_at_N",
                s,
            )
            # remove most recent bytecode instructions
            # DOTALL is needed to entirely remove TRACE ... lines (including the newline)
            return re.sub(r"TRACE.*$", "", s, flags=re.DOTALL)

        self.assertExpectedInline(
            post_munge(munge_exc(records[-1].getMessage(), skip=0)),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_traceback_above_dynamo_shows_user_code
    f3(torch.randn(3))
  File "test_error_messages.py", line N, in f3
    Foo().attr = x  # 0
  File "test_error_messages.py", line N, in torch_dynamo_resume_in_f3_at_N
    Foo().attr = x  # 1

========== most recent `torch.compile` tracing attempt started here ==========

  File "test_error_messages.py", line N, in torch_dynamo_resume_in_f3_at_N
    Foo().attr = x
  File "test_error_messages.py", line N, in __setattr__
    torch._dynamo.graph_break()

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.

Most recent bytecode instructions traced (max 20):
""",
        )

    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_graph_break_traceback_inlined_data_dependent_branch(self, records):
        # [FUNCTION DOC] With verbose logging, checks that a data-dependent branch inside an
        # inlined helper is attributed to the right lines in the traceback (call site, not the
        # helper source); the final record is compared after normalization.
        # Tests that data-dependent branches in inlined functions show correct
        # source lines in the verbose traceback (from the call site, not the
        # inlined function's source).
        def helper(x):
            # [FUNCTION DOC] HELPER: Branches twice on the tensor sum, adding one each time.
            if x.sum() > 0:  # 0
                x = x + 1
            if x.sum() > 0:  # 1
                x = x + 1
            return x

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled target; returns the helper result plus one.
            return helper(x) + 1  # call site

        f(torch.ones(3))

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Normalizes resume function names with line numbers and
            # strips trace lines.
            s = re.sub(
                r"torch_dynamo_resume_in_(\w+)_at_(\d+)",
                r"torch_dynamo_resume_in_\1_at_N",
                s,
            )
            return re.sub(r"TRACE.*$", "", s, flags=re.DOTALL)

        self.assertExpectedInline(
            post_munge(munge_exc(records[-1].getMessage(), skip=0)),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "test_error_messages.py", line N, in torch_dynamo_resume_in_helper_at_N, code: if x.sum() > 0:  # 1
        gt = gt(sum_1, 0)

  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_traceback_inlined_data_dependent_branch
    f(torch.ones(3))
  File "test_error_messages.py", line N, in f
    return helper(x) + 1  # call site

========== most recent `torch.compile` tracing attempt started here ==========

  File "test_error_messages.py", line N, in torch_dynamo_resume_in_f_at_N
    return helper(x) + 1  # call site
  File "test_error_messages.py", line N, in torch_dynamo_resume_in_helper_at_N
    if x.sum() > 0:  # 1

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.

Most recent bytecode instructions traced (max 20):
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_graph_break_traceback_collapsed_resume_frames(self, records):
        # [FUNCTION DOC] Checks that the traceback in the last graph break record collapses
        # repeated resume frames across three nested functions, each with three breaks, to the
        # correct frames.
        @torch.compile(backend="eager")
        def f1(x):
            # [FUNCTION DOC] HELPER: Compiled target; three graph breaks then calls f2.
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            f2(x)

        def f2(x):
            # [FUNCTION DOC] HELPER: Three graph breaks then calls f3.
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            f3(x)

        def f3(x):
            # [FUNCTION DOC] HELPER: Three graph breaks then returns the input plus one.
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()
            torch._dynamo.graph_break()  # correct
            return x + 1

        f1(torch.randn(3))

        self.assertExpectedInline(
            munge_exc(records[-1].getMessage(), skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_graph_break_traceback_collapsed_resume_frames
    f1(torch.randn(3))
  File "test_error_messages.py", line N, in f1
    f2(x)
  File "test_error_messages.py", line N, in f2
    f3(x)
  File "test_error_messages.py", line N, in f3
    torch._dynamo.graph_break()  # correct
""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_logs_user_stack_trace(self, records):
        # [FUNCTION DOC] With debug-level dynamo logging, compiles a function decorated with
        # the standard least-recently-used cache wrapper, finds the log record warning about
        # calling such a function, asserts it exists, and compares the normalized text, which
        # lists the user call stack beginning at the test frame.
        @lru_cache
        def foo(x):
            # [FUNCTION DOC] HELPER: Cached target; adds one to the input.
            return x + 1

        torch.compile(foo, backend="eager")(torch.randn(4))

        lru_cache_log = None
        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                lru_cache_log = record.getMessage()
                break

        self.assertIsNotNone(lru_cache_log, "No lru_cache warning was logged")

        self.assertExpectedInline(
            munge_exc(lru_cache_log),
            """\
call to a lru_cache wrapped function at: _dynamo/external_utils.py:N
  File "test_error_messages.py", line N, in test_lru_cache_warning_logs_user_stack_trace
    torch.compile(foo, backend="eager")(torch.randn(4))
""",
        )

    @make_logging_test(dynamo=logging.DEBUG)
    def test_lru_cache_warning_logs_nested_call(self, records):
        # [FUNCTION DOC] Same as the previous check but the cached function is called from a
        # nested helper; the warning text lists both the test frame and the helper frame.
        @lru_cache
        def foo(x):
            # [FUNCTION DOC] HELPER: Cached function; adds one to the input.
            return x + 1

        def nested(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls the cached function.
            return foo(x)

        torch.compile(nested, backend="eager")(torch.randn(4))

        lru_cache_log = None
        for record in records:
            if "call to a lru_cache wrapped function at:" in record.getMessage():
                lru_cache_log = record.getMessage()
                break

        self.assertIsNotNone(lru_cache_log, "No lru_cache warning was logged")

        self.assertExpectedInline(
            munge_exc(lru_cache_log),
            """\
call to a lru_cache wrapped function at: test_error_messages.py:N
  File "test_error_messages.py", line N, in test_lru_cache_warning_logs_nested_call
    torch.compile(nested, backend="eager")(torch.randn(4))
  File "test_error_messages.py", line N, in nested
    return foo(x)
""",
        )

    def test_lru_cache_warning(self):
        # [FUNCTION DOC] Checks only that a user warning is issued (once-style assertion)
        # whose text says the call originates from a named line in the intermediate helper,
        # when a full graph compile reaches a cached function.
        # test only the warning message itself
        @lru_cache
        def bax(x):
            # [FUNCTION DOC] HELPER: Cached function; adds one to the input.
            return x + 1

        def bar(x):
            # [FUNCTION DOC] HELPER: Calls the cached function; the frame the warning must
            # point to.
            return bax(x)

        @torch.compile(backend="eager", fullgraph=True)
        def foo(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls bar.
            return bar(x)

        x = torch.randn(2)
        with self.assertWarnsOnceRegex(
            UserWarning,
            r"(?s).*This call originates from:\n.*File .*, line (\d+), in bar",
        ):
            foo(x)

    def test_disable_message(self):
        # [FUNCTION DOC] Checks the messages for calling functions wrapped by the compiler
        # disable API: one without a reason, one with a reason, and an already compiled module
        # wrapped by it (reported as an unsupported function call with the module reason).
        # Addresses normalized.
        @torch.compile(backend="eager", fullgraph=True)
        def outer(fn, x):
            # [FUNCTION DOC] HELPER: Compiled target; calls its first argument
            # with the second.
            return fn(x)

        @torch.compiler.disable
        def f(x):
            # [FUNCTION DOC] HELPER: Disabled function; adds one to the input.
            return x + 1

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces hex addresses with a placeholder.
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: outer(f, torch.randn(3)),
            """\
Skip calling `torch.compiler.disable()`d function
  Explanation: Skip calling function `<function ErrorMessagesTest.test_disable_message.<locals>.f at 0xmem_addr>` since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_disable_message.<locals>.f at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0098.html

from user code:
   File "test_error_messages.py", line N, in outer
    return fn(x)""",
            post_munge=post_munge,
        )

        @torch.compiler.disable(reason="test message")
        def g(x):
            # [FUNCTION DOC] HELPER: Disabled (with a reason) function; adds two to the input.
            return x + 2

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: outer(g, torch.randn(3)),
            """\
Skip calling `torch.compiler.disable()`d function
  Explanation: Skip calling function `<function ErrorMessagesTest.test_disable_message.<locals>.g at 0xmem_addr>` since it was wrapped with `torch.compiler.disable` (reason: test message)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_disable_message.<locals>.g at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0098.html

from user code:
   File "test_error_messages.py", line N, in outer
    return fn(x)""",
            post_munge=post_munge,
        )

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds three to the input; compiled and
            # then wrapped with the disable API in the test.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds three to the input.
                return x + 3

        mod = Mod()
        mod.compile()
        mod = torch.compiler.disable(mod, reason="test message 2")

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: outer(mod, torch.randn(3)),
            """\
Unsupported function call (delayed)
  Explanation: Dynamo determined that a graph break should occur when calling `L['fn']`. Reason: Optimized `nn.Module` is wrapped with `torch.compiler.disable` (reason: test message 2)


  Developer debug context: source: LocalSource(local_name='fn', is_input=True, dynamism=None, is_derefed_cell_contents=False)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0148.html

from user code:
   File "test_error_messages.py", line N, in outer
    return fn(x)""",
            post_munge=post_munge,
        )

    # Test that errors while tracing resume function prologues do not get suppressed
    def test_graph_break_in_buggy_resume_prologue(self):
        # [FUNCTION DOC] Patches the bytecode assembly helper so that an invalid global load
        # is injected right after the prologue flag store in generated resume code, then
        # asserts the compile raises the resume prologue tracing error rather than silently
        # suppressing it. Side effect: the patch is active only inside the with block.
        import torch._dynamo.bytecode_transformation as bt
        import torch._dynamo.resume_execution as rex

        # NOTE: do not define non_global as a global in this file!
        @torch.compile(backend="eager")
        def fn(non_global):
            # [FUNCTION DOC] HELPER: Compiled target; adds one, inserts a graph break, adds
            # one more.
            non_global = non_global + 1
            torch._dynamo.graph_break()
            return non_global + 1

        orig_clean_and_assemble_instructions = bt.clean_and_assemble_instructions

        def bad_clean_and_assemble_instructions(instructions, *args):
            # [FUNCTION DOC] HELPER: Replacement for the instruction assembler. Scans the
            # instruction list for the first store of the resume prologue flag variable,
            # splices in a load of the global named non_global right after it (mutating the
            # list in place), then delegates to the original assembler. Returns the original
            # assembler result.
            # Inject an invalid LOAD_GLOBAL after the first STORE_FAST IS_TRACING_RESUME_PROLOGUE_VARNAME
            for i, inst in enumerate(instructions):
                if (
                    inst.opname == "STORE_FAST"
                    and inst.argval == rex.IS_TRACING_RESUME_PROLOGUE_VARNAME
                ):
                    instructions[:] = (
                        instructions[: i + 1]
                        + [
                            # this should cause a graph break
                            bt.create_instruction("LOAD_GLOBAL", argval="non_global"),
                        ]
                        + instructions[i + 1 :]
                    )
                    break
            return orig_clean_and_assemble_instructions(instructions, *args)

        with unittest.mock.patch(
            "torch._dynamo.bytecode_transformation.clean_and_assemble_instructions",
            bad_clean_and_assemble_instructions,
        ):
            with self.assertRaisesRegex(
                ResumePrologueTracingError,
                "Error while tracing through a Dynamo-generated resume function prologue.",
            ):
                fn(torch.randn(3))

    @make_logging_test(graph_breaks=True)
    def test_step_graph_break(self, records):
        # [FUNCTION DOC] Checks the log record for the user step-unsupported call (a break
        # that cannot be resumed), then resets dynamo and, with error on graph break enabled,
        # checks the raised Unsupported text for the same call.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; adds one and two, calls the step-
            # unsupported debugging hook, then adds four.
            x = x + 1
            x = x + 2
            torch._dynamo.step_unsupported()
            return x + 4

        fn(torch.ones(3))

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Call to `torch._dynamo.step_unsupported()`
  Explanation: User-inserted step_unsupported.
  Hint: Remove the `torch._dynamo.step_unsupported()` call.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4636.html

User code traceback:
  File "test_error_messages.py", line N, in test_step_graph_break
    fn(torch.ones(3))
  File "test_error_messages.py", line N, in fn
    torch._dynamo.step_unsupported()
""",
        )

        torch._dynamo.reset()

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: fn(torch.ones(3)),
                """\
cannot resume from torch._dynamo.step_unsupported()
  Explanation: traced torch._dynamo.step_unsupported(), but Dynamo is instructed to error on graph break. This graph break is used for debugging only.
  Hint: Remove the torch._dynamo.step_unsupported() call.
  Hint: Make sure fullgraph=False and error_on_graph_break=False.
  Hint: This is likely to be a Dynamo bug. Please report an issue to PyTorch.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0284.html

from user code:
   File "test_error_messages.py", line N, in fn
    torch._dynamo.step_unsupported()""",
            )

    @make_logging_test(graph_breaks=True)
    def test_store_attr_graph_break(self, records):
        # [FUNCTION DOC] Checks the graph break record when an object's attribute-assignment
        # hook is wrapped with the disable API, then resets and checks the same message raised
        # as an error when error on graph break is enabled. Addresses are normalized.
        class Foo:
            # [CLASS DOC] HELPER: Local class whose attribute-assignment hook is wrapped by
            # the disable API and forwards to the parent behavior.
            @torch.compiler.disable
            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: Disabled attribute-assignment hook; delegates to the
                # parent hook.
                super().__setattr__(name, value)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; assigns an attribute on a new instance.
            Foo().attr = x

        fn(torch.ones(3))

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces hex addresses with a placeholder.
            return re.sub(r"0x[0-9A-Fa-f]+", "0xmem_addr", s)

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[-1].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace STORE_ATTR: storing an object's attribute, e.g. x.attr = y:

Skip inlining `torch.compiler.disable()`d function
  Explanation: Skip inlining function <function ErrorMessagesTest.test_store_attr_graph_break.<locals>.Foo.__setattr__ at 0xmem_addr> since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_store_attr_graph_break.<locals>.Foo.__setattr__ at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0099.html

User code traceback:
  File "test_error_messages.py", line N, in test_store_attr_graph_break
    fn(torch.ones(3))
  File "test_error_messages.py", line N, in fn
    Foo().attr = x
""",
        )

        torch._dynamo.reset()

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: fn(torch.ones(3)),
                """\
Skip inlining `torch.compiler.disable()`d function
  Explanation: Skip inlining function <function ErrorMessagesTest.test_store_attr_graph_break.<locals>.Foo.__setattr__ at 0xmem_addr> since it was wrapped with `torch.compiler.disable` (reason: None)
  Hint: Remove the `torch.compiler.disable` call

  Developer debug context: <function ErrorMessagesTest.test_store_attr_graph_break.<locals>.Foo.__setattr__ at 0xmem_addr>

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0099.html

from user code:
   File "test_error_messages.py", line N, in fn
    Foo().attr = x""",
                post_munge=post_munge,
            )

    def test_runtime_error_readable_shape_mismatch(self):
        # [FUNCTION DOC] Marks one dimension of each input as dynamic and checks that the
        # runtime error for a size mismatch in addition shows readable symbolic sizes with
        # hints; the symbol number is normalized.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Compiled target; adds its two arguments.
            return x + y

        x = torch.randn(4, 4)
        y = torch.randn(10, 10)
        torch._dynamo.mark_dynamic(x, 2)
        torch._dynamo.mark_dynamic(y, 1)

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Replaces the varying symbol name before a ten hint with a
            # fixed one.
            s = re.sub(r"s\d+: hint = 10", "s94: hint = 10", s)
            return s

        self.assertExpectedInlineMunged(
            TorchRuntimeError,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(x, y),
            """\
RuntimeError when making fake tensor call
  Explanation: Dynamo failed to run FX node with fake tensors: call_function <built-in function add>(*(FakeTensor(..., size=(4, 4)), FakeTensor(..., size=(10, s94))), **{}): got RuntimeError('The size of tensor a (4) must match the size of tensor b (s94: hint = 10) at non-singleton dimension 1)')
  Hint: Your code may result in an error when running in eager. Please double check that your code doesn't contain a similar error when actually running eager/uncompiled. You can do this by removing the `torch.compile` call, or by using `torch.compiler.set_stance("force_eager")`.

  Developer debug context:

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb4315.html

from user code:
   File "test_error_messages.py", line N, in fn
    return x + y""",
            post_munge=post_munge,
        )

    def test_hop_side_effect_error_includes_hop_context(self):
        # [FUNCTION DOC] Checks that mutating an outer list inside a function run by non-
        # reentrant activation checkpointing raises Unsupported mentioning the higher order
        # operator context.
        # Test that graph breaks inside HOPs include the HOP context
        stack = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Checkpointed function; appends to an outer list (side
            # effect) and returns the sine of the input.
            stack.append(1)  # Mutation outside checkpoint scope
            return x.sin()

        def model(x):
            # [FUNCTION DOC] HELPER: Compiled target; runs the checkpointed
            # function without reentrancy.
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(model, backend="eager", fullgraph=True)(
                torch.rand(4)
            ),
            """\
HOP: Unsafe side effect
  Higher Order Operator: torch.utils.checkpoint.checkpoint
  Explanation: Mutating a variable from outside the scope of this HOP is not supported.
  Hint: If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a side effect in forward method. Eager activation checkpointing replays that side-effect while recomputing the forward in the backward. If you are ok with side-effect not replayed in the backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`

  Developer debug context: Attempted to mutate ListVariable(length=0)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0067.html

from user code:
   File "test_error_messages.py", line N, in model
    return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
  File "test_error_messages.py", line N, in fn
    stack.append(1)  # Mutation outside checkpoint scope""",
        )

    @make_logging_test(graph_breaks=True)
    def test_hop_side_effect_error_includes_hop_context_fullgraph_false(self, records):
        # [FUNCTION DOC] Same side effect check with reentrant checkpointing and no full
        # graph; the single graph break record shows the higher order operator context.
        # Test that graph breaks inside HOPs include the HOP context
        # even with fullgraph=False (checkpoint allows graph breaks)
        stack = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Checkpointed function; appends to an outer list and
            # returns sine.
            stack.append(1)  # Mutation outside checkpoint scope
            return x.sin()

        def model(x):
            # [FUNCTION DOC] HELPER: Compiled target; runs the function under
            # reentrant checkpointing.
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=True)

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Rewrites the instruction name part of the
            # break reason (function call with keywords or the newer keyword call)
            # to a generic call description.
            # deal with CALL_FUNCTION_KW/CALL_KW
            return re.sub(
                r"attempting to trace (CALL_FUNCTION_KW|CALL_KW):.*$",
                "attempting to trace CALL: a function call, e.g. f(x, y):",
                s,
                flags=re.MULTILINE,
            )

        torch.compile(model, backend="eager", fullgraph=False)(torch.rand(4))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            post_munge(
                munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

HOP: Unsafe side effect
  Higher Order Operator: torch.utils.checkpoint.checkpoint
  Explanation: Mutating a variable from outside the scope of this HOP is not supported.
  Hint: If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a side effect in forward method. Eager activation checkpointing replays that side-effect while recomputing the forward in the backward. If you are ok with side-effect not replayed in the backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`

  Developer debug context: Attempted to mutate ListVariable(length=0)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0067.html

User code traceback:
  File "test_error_messages.py", line N, in test_hop_side_effect_error_includes_hop_context_fullgraph_false
    torch.compile(model, backend="eager", fullgraph=False)(torch.rand(4))
  File "test_error_messages.py", line N, in model
    return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=True)
  File "test_error_messages.py", line N, in fn
    stack.append(1)  # Mutation outside checkpoint scope
""",
        )

    def test_nested_hop_side_effect_error(self):
        # [FUNCTION DOC] Checks that when checkpoint is nested inside a wrap higher order
        # operator, only the innermost one (checkpoint) is named in the error, while the user
        # frames of all levels are listed.
        # Test that only the innermost HOP context is shown (checkpoint, not wrap)
        stack = []

        def fn(x):
            # [FUNCTION DOC] HELPER: Checkpointed function; appends to an outer
            # list, returns sine.
            stack.append(1)  # Mutation outside checkpoint scope
            return x.sin()

        def inner(x):
            # [FUNCTION DOC] HELPER: Function run by wrap; runs checkpoint on the
            # inner function.
            return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)

        def model(x):
            # [FUNCTION DOC] HELPER: Compiled target; invokes the wrap higher order operator.
            return torch.ops.higher_order.wrap(inner, x)

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(model, backend="eager", fullgraph=True)(
                torch.rand(4)
            ),
            """\
HOP: Unsafe side effect
  Higher Order Operator: torch.utils.checkpoint.checkpoint
  Explanation: Mutating a variable from outside the scope of this HOP is not supported.
  Hint: If the HOP is activation checkpointing (torch.utils.checkpoint.checkpoint), this points to a side effect in forward method. Eager activation checkpointing replays that side-effect while recomputing the forward in the backward. If you are ok with side-effect not replayed in the backward, try setting `torch._dynamo.config.skip_fwd_side_effects_in_bwd_under_checkpoint = True`

  Developer debug context: Attempted to mutate ListVariable(length=0)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0067.html

from user code:
   File "test_error_messages.py", line N, in model
    return torch.ops.higher_order.wrap(inner, x)
  File "test_error_messages.py", line N, in inner
    return torch.utils.checkpoint.checkpoint(fn, x, use_reentrant=False)
  File "test_error_messages.py", line N, in fn
    stack.append(1)  # Mutation outside checkpoint scope""",
        )

    def test_cond_with_graph_break_shows_hop_context(self):
        # [FUNCTION DOC] Checks that a graph break inside the true branch of a
        # conditional operator raises the uncaptured higher order operator error
        # naming the conditional operator.
        # Test that torch.cond graph breaks include HOP context
        def true_fn(x):
            # [FUNCTION DOC] HELPER: True branch; inserts a graph break then returns sine.
            torch._dynamo.graph_break()
            return x.sin()

        def false_fn(x):
            # [FUNCTION DOC] HELPER: False branch; returns cosine.
            return x.cos()

        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; runs the conditional operator on the
            # sign of the tensor sum.
            return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        from torch._dynamo.exc import UncapturedHigherOrderOpError

        self.assertExpectedInlineMunged(
            UncapturedHigherOrderOpError,
            lambda: torch.compile(fn, backend="eager", fullgraph=True)(torch.randn(3)),
            """\
This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

Call to `torch._dynamo.graph_break()`
  Higher Order Operator: torch.cond
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.cond(x.sum() > 0, true_fn, false_fn, [x])
  File "test_error_messages.py", line N, in true_fn
    torch._dynamo.graph_break()""",
        )

    @make_logging_test(graph_breaks=True)
    def test_cond_with_generator_gb_skip(self, records):
        # [FUNCTION DOC] Checks the uncaptured higher order operator error when a generator
        # with a graph break is consumed inside a generic context manager within a conditional
        # branch; the message chains the generator skip note and the context manager break.
        # Then repeats with error on graph break enabled and expects only the first part.
        def generator():
            # [FUNCTION DOC] HELPER: Generator yielding one and two, inserting a graph break,
            # then yielding four.
            yield 1
            yield 2
            torch._dynamo.graph_break()
            yield 4

        def true_fn(x):
            # [FUNCTION DOC] HELPER: True branch; collects the generator values inside a
            # generic context manager and returns sine plus their sum.
            with GenericCtxMgr():
                l = list(generator())
            return x.sin() + sum(l)

        def false_fn(x):
            # [FUNCTION DOC] HELPER: False branch; returns cosine.
            return x.cos()

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; runs the conditional operator on the
            # sign of the tensor sum.
            return torch.cond(x.sum() > 0, true_fn, false_fn, [x])

        from torch._dynamo.exc import UncapturedHigherOrderOpError

        self.assertExpectedInlineMunged(
            UncapturedHigherOrderOpError,
            lambda: fn(torch.ones(3)),
            """\
This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

Call to `torch._dynamo.graph_break()`
  Higher Order Operator: torch.cond
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

Skipping frame due to graph break in a generator's next() call.

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.cond(x.sum() > 0, true_fn, false_fn, [x])
  File "test_error_messages.py", line N, in true_fn
    l = list(generator())
  File "test_error_messages.py", line N, in generator
    torch._dynamo.graph_break()""",
        )

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                UncapturedHigherOrderOpError,
                lambda: fn(torch.ones(3)),
                """\
This higher order operator doesn't work unless it is captured completely with torch.compile. Got graph break/error:

Call to `torch._dynamo.graph_break()`
  Higher Order Operator: torch.cond
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_error_messages.py", line N, in fn
    return torch.cond(x.sum() > 0, true_fn, false_fn, [x])
  File "test_error_messages.py", line N, in true_fn
    l = list(generator())
  File "test_error_messages.py", line N, in generator
    torch._dynamo.graph_break()""",
            )

    @make_logging_test()
    @torch._dynamo.config.patch(recompile_limit=1)
    def test_recompile_limit_hit_message(self, records):
        # [FUNCTION DOC] With the recompile limit patched to one, calls a function with a
        # different integer value twice (static shapes) to hit the limit and checks the logged
        # limit-hit text (after removing the user stack section and normalizing the path).
        # Then resets the compiler and checks the Unsupported error raised under error on
        # graph break.
        @torch.compile(backend="eager", dynamic=False)
        def fn(x, n):
            # [FUNCTION DOC] HELPER: Compiled target; adds an integer to the tensor.
            return x + n

        def outer(x):
            # [FUNCTION DOC] HELPER: Calls the compiled function twice with loop indices zero
            # and one.
            for i in range(2):
                x = fn(x, i)
            return x

        outer(torch.ones(3))

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Removes the user stack trace section and shortens the
            # test file path in a comment in the log text.
            # Remove user stack trace section that appears in recompile limit messages
            s = re.sub(
                r"\nUser stack trace:.*?(?=\nTo log all)", "", s, flags=re.DOTALL
            )
            return re.sub(
                r"# \S+/test_error_messages\.py", "# test_error_messages.py", s
            )

        self.assertExpectedInline(
            post_munge(
                munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0)
            ),
            """\
torch._dynamo hit config.recompile_limit (1)
   function: 'fn' (test_error_messages.py:N)
   last reason: 0/0: n == 0                                                   # return x + n  # test_error_messages.py:N in fn
To log all recompilation reasons, use TORCH_LOGS="recompiles".
To diagnose recompilation issues, see https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html""",
        )

        torch.compiler.reset()

        with torch._dynamo.error_on_graph_break(True):
            self.assertExpectedInlineMunged(
                Unsupported,
                lambda: outer(torch.ones(3)),
                """\
Dynamo recompile limit exceeded
  Explanation: Dynamo attempted to recompile the code object too many times, exceeding the recompile_limit cache size limit (currently set to 1). Excessive recompilations can degrade performance due to the compilation overhead of each recompilation.
  Hint: To monitor recompilations, enable TORCH_LOGS=recompiles. If recompilations are expected, consider increasing torch._dynamo.config.recompile_limit to an appropriate value.
  Hint: See https://docs.pytorch.org/docs/main/user_guide/torch_compiler/compile/programming_model.recompilation.html for tips on dealing with recompilations.

  Developer debug context: Limit type: recompile_limit

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0039.html""",
            )

    @make_logging_test(graph_breaks=True)
    def test_nested_generic_ctx_mgr(self, records):
        # [FUNCTION DOC] Compiles a function with nested generic context managers that calls a
        # helper with more nesting and a graph break. Expects two log records: the first is
        # the full message chaining the break with two context manager failures (three and two
        # active managers); the second is a duplicate-suppressed shortened record for the
        # first failure.
        def inner():
            # [FUNCTION DOC] HELPER: Opens three nested generic context managers and inserts a
            # graph break inside them.
            with GenericCtxMgr():
                with GenericCtxMgr():
                    with GenericCtxMgr():
                        torch._dynamo.graph_break()

        def fn():
            # [FUNCTION DOC] HELPER: Compiled target; opens two nested generic context
            # managers and calls inner within them.
            with GenericCtxMgr():
                with GenericCtxMgr():
                    inner()

        torch.compile(fn, backend="eager")()
        self.assertEqual(len(records), 2)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_nested_generic_ctx_mgr
    torch.compile(fn, backend="eager")()
  File "test_error_messages.py", line N, in fn
    inner()
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )
        self.assertExpectedInline(
            munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break (user stack suppressed due to duplicate graph break) in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr), GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html""",
        )

    @make_logging_test(graph_breaks=True)
    def test_skipped_frame_with_verbose_traceback_nested(self, records):
        # [FUNCTION DOC] Declares f1, f2 and f3 as module globals, then compiles the outermost
        # one. Checks the single log record for a break inside a context manager reached
        # through three nested calls: unresumable break message with user traceback through
        # all three frames. Side effect: rebinds module level names f1, f2, f3.
        global f1, f2, f3

        def f1(x):
            # [FUNCTION DOC] HELPER: Innermost function; inserts a graph break inside a
            # generic context manager, then returns the input plus one.
            with GenericCtxMgr():
                torch._dynamo.graph_break()
                return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 with the input plus two.
            return f1(x + 2)

        def f3(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls f2 with the input plus three.
            return f2(x + 3)

        torch.compile(f3, backend="eager")(torch.randn(3))
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

Graph break under GenericContextWrappingVariable
  Explanation: Attempted to graph break in an active context manager(s) that doesn't support graph breaking.
  Hint: Move the offending context manager(s) to outside the compiled region.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: Active generic context managers: [GenericContextWrappingVariable(GenericCtxMgr)]

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0066.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_skipped_frame_with_verbose_traceback_nested
    torch.compile(f3, backend="eager")(torch.randn(3))
  File "test_error_messages.py", line N, in f3
    return f2(x + 3)
  File "test_error_messages.py", line N, in f2
    return f1(x + 2)
  File "test_error_messages.py", line N, in f1
    torch._dynamo.graph_break()
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_skip_frame_in_loop_message_nested(self, records):
        # [FUNCTION DOC] Declares f1, f2, f3 as module globals and compiles the outermost one.
        # Checks the single record for a data-dependent branch in a loop with a context
        # manager reached through three nested calls (unresumable break). Side effect: rebinds
        # module level names.
        global f1, f2, f3

        def f1(x):
            # [FUNCTION DOC] HELPER: Innermost function; loops twice with a generic context
            # manager and a data-dependent branch adding one; returns the tensor.
            for i in range(2):
                with GenericCtxMgr():
                    if x.sum() > 0:
                        x = x + 1
            return x

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 with the input plus four.
            return f1(x + 4)

        def f3(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls f2 with the input plus five.
            return f2(x + 5)

        result = torch.compile(f3, backend="eager")(torch.randn(3))  # noqa: F841
        self.assertEqual(len(records), 1)
        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break that we cannot resume from. Compiling up to the previous resumable state, then skipping the rest of the function. Graph break encountered:

Data-dependent branching
  Explanation: Detected data-dependent branching (e.g. `if my_tensor.sum() > 0:`). Dynamo does not support tracing dynamic control flow.

      The branch condition involves a tensor computed as follows:
        # File "test_error_messages.py", line N, in f1, code: if x.sum() > 0:
        gt = gt(sum_1, 0)

  Hint: For the common pattern `if tensor_cond: x = transform(x)` (e.g. clamping inf/nan values), consider making the code branchless by always applying the transform. Operations like torch.clamp, torch.nan_to_num, and torch.where are typically no-ops on well-behaved inputs and compile without graph breaks.
  Hint: This graph break is fundamental - it is unlikely that Dynamo will ever be able to trace through your code. Consider finding a workaround.
  Hint: Use `torch.cond` to express dynamic control flow.

  Developer debug context: attempted to jump with TensorVariable()

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0170.html

Stack variable source attribution:

User code traceback:
  File "test_error_messages.py", line N, in test_skip_frame_in_loop_message_nested
    result = torch.compile(f3, backend="eager")(torch.randn(3))  # noqa: F841
  File "test_error_messages.py", line N, in f3
    return f2(x + 5)
  File "test_error_messages.py", line N, in f2
    return f1(x + 4)
  File "test_error_messages.py", line N, in f1
    if x.sum() > 0:
""",
        )

    @make_logging_test(graph_breaks=True)
    def test_try_block_with_graph_break_suppression(self, records):
        # [FUNCTION DOC] Declares inner, middle_with_try and outer as module globals. Compiles
        # outer where a graph break in inner is wrapped by a try block in the middle function,
        # with verbose off. Expects exactly one full graph break record and two duplicate-
        # suppressed records (one for each re-traced top frame), and compares all three. Side
        # effect: rebinds module level names.
        global inner, middle_with_try, outer

        def inner(x):
            # [FUNCTION DOC] HELPER: Adds one, inserts a graph break, then adds another one.
            result = x + 1
            torch._dynamo.graph_break()
            return result + 1

        def middle_with_try(x):
            # [FUNCTION DOC] HELPER: Calls inner inside a try block that swallows exceptions;
            # falls back to returning its input if the call raised.
            try:
                return inner(x)
            except Exception:
                pass
            return x

        def outer(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls the middle function.
            return middle_with_try(x)

        with torch._dynamo.config.patch(verbose=False):
            torch.compile(outer, backend="eager")(torch.ones(3))

        full_messages = [
            r for r in records if "Graph break in user code" in r.getMessage()
        ]
        suppressed_messages = [
            r
            for r in records
            if "user stack suppressed due to duplicate" in r.getMessage()
        ]

        self.assertEqual(
            len(full_messages),
            1,
            lambda msg: f"{msg}\nExpected 1 full graph break message, got {len(full_messages)}",
        )
        self.assertEqual(
            len(suppressed_messages),
            2,
            lambda msg: f"{msg}\nExpected 2 suppressed messages, got {len(suppressed_messages)}",
        )

        self.assertExpectedInline(
            munge_exc(full_messages[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_try_block_with_graph_break_suppression
    torch.compile(outer, backend="eager")(torch.ones(3))
  File "test_error_messages.py", line N, in outer
    return middle_with_try(x)
  File "test_error_messages.py", line N, in middle_with_try
    return inner(x)
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )

        # graph break in middle_with_try as top frame
        self.assertExpectedInline(
            munge_exc(
                suppressed_messages[0].getMessage(), suppress_suffix=True, skip=0
            ),
            """\
Graph break (user stack suppressed due to duplicate graph break) in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html""",
        )

        # graph break in inner as top frame
        self.assertExpectedInline(
            munge_exc(
                suppressed_messages[1].getMessage(), suppress_suffix=True, skip=0
            ),
            """\
Graph break (user stack suppressed due to duplicate graph break) in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html""",
        )

    @make_logging_test(graph_breaks=True)
    def test_nested_graph_break_different_call_sites_not_suppressed(self, records):
        # [FUNCTION DOC] Declares inner and outer as module globals; outer calls inner at two
        # different call sites with verbose off. Expects two graph break records, both with
        # full user traces, showing duplicate suppression does not apply across different call
        # sites. Side effect: rebinds module level names.
        global inner, outer

        def inner(x):
            # [FUNCTION DOC] HELPER: Adds one, inserts a graph break, then adds two.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        @torch.compile(backend="eager")
        def outer(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls inner at two separate call sites
            # with additions in between.
            x = inner(x + 4) + 8
            return inner(x) + 16

        with torch._dynamo.config.patch(verbose=False):
            outer(torch.ones(3))

        self.assertEqual(
            len(records),
            2,
            lambda msg: f"{msg}\nExpected 2 graph break messages (one per call site), got {len(records)}",
        )

        self.assertExpectedInline(
            munge_exc(records[0].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_nested_graph_break_different_call_sites_not_suppressed
    outer(torch.ones(3))
  File "test_error_messages.py", line N, in outer
    x = inner(x + 4) + 8
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )

        self.assertExpectedInline(
            munge_exc(records[1].getMessage(), suppress_suffix=True, skip=0),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_nested_graph_break_different_call_sites_not_suppressed
    outer(torch.ones(3))
  File "test_error_messages.py", line N, in outer
    return inner(x) + 16
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
""",
        )

    @unittest.skipIf(
        sys.version_info < (3, 13), "carets in tracebacks require Python 3.13+"
    )
    @torch._dynamo.config.patch(verbose=True)
    @make_logging_test(graph_breaks=True)
    def test_inlined_frame_carets(self, records):
        # [FUNCTION DOC] Skipped before Python 3.13 (no carets in tracebacks). With verbose
        # logging, checks the first record keeps caret markers under source lines, in both the
        # outer frame and the inlined inner frame and the user frame above Dynamo. The
        # bytecode trace section is removed before comparison.
        """Carets come from frame_summary (outer), _populate_loc_in_frame_summary
        (inner), and _extract_stack_with_positions (stack above dynamo)."""

        def inner(x):
            # [FUNCTION DOC] HELPER: Inserts a graph break and returns the input.
            torch._dynamo.graph_break()
            return x

        def outer(x):
            # [FUNCTION DOC] HELPER: Compiled target; calls inner and adds one.
            return inner(x) + 1

        torch.compile(outer, backend="eager")(torch.ones(3))

        def post_munge(s):
            # [FUNCTION DOC] HELPER: Removes the recent bytecode instruction section from the
            # end of the text.
            return re.sub(
                r"\nMost recent bytecode instructions traced.*", "", s, flags=re.DOTALL
            )

        self.assertExpectedInline(
            post_munge(
                munge_exc(
                    records[0].getMessage(),
                    suppress_suffix=True,
                    skip=0,
                    strip_carets=False,
                )
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Encountered graph break when attempting to trace CALL: a function call, e.g. f(x, y):

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

User code traceback:
  File "test_error_messages.py", line N, in test_inlined_frame_carets
    torch.compile(outer, backend="eager")(torch.ones(3))
    ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~^^^^^^^^^^^^^^^

========== most recent `torch.compile` tracing attempt started here ==========

  File "test_error_messages.py", line N, in outer
    return inner(x) + 1
           ~~~~~^^^
  File "test_error_messages.py", line N, in inner
    torch._dynamo.graph_break()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^

NOTE: the most recent `torch.compile` tracing attempt might not be where you applied `torch.compile`! This is due to how graph breaks are implemented - the optimized code object returned by Dynamo will call another Dynamo-generated resume function and tracing is re-enabled by calling the resume function as a normal Python function, which Dynamo intercepts as a top-level frame.
""",
        )

    @unittest.skipIf(
        sys.version_info < (3, 13),
        "stack variable source attribution requires Python 3.13+",
    )
    @make_logging_test(graph_breaks=True)
    def test_stack_variable_source_attribution(self, records):
        # [FUNCTION DOC] Skipped before Python 3.13. Checks that, with carets and stack
        # attribution not stripped, the record for a graph break inside a loop reports which
        # source expression produced the iterator left on the stack.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled target; loops twice inserting a graph break,
            # then adds one.
            for i in range(2):
                torch._dynamo.graph_break()
            return x + 1

        fn(torch.ones(3))
        self.assertExpectedInline(
            munge_exc(
                records[0].getMessage(),
                suppress_suffix=True,
                skip=0,
                strip_carets=False,
                strip_stack_attribution=False,
            ),
            """\
Graph break in user code at test_error_messages.py:N
Graph Break Reason: Failed to handle graph break gracefully. Skipping the function and falling back to eager. Graph break encountered:

Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

*** While handling this graph break, another graph break occurred: ***

graph break in loop
  Explanation: torch.compile detected a graph break in a for/while loop. Skipping the frame and falling back to eager, as graph breaks in loops are not supported.
  Hint: This graph break may have been caused by an earlier graph break. Resolving the earlier graph break may resolve this one.

  Developer debug context: frame skipped: fn (test_error_messages.py line N)

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb7000.html

Stack variable source attribution:
  RangeIteratorVariable() originated from:
  File "test_error_messages.py", line N
                for i in range(2):
                         ~~~~~^^^

User code traceback:
  File "test_error_messages.py", line N, in test_stack_variable_source_attribution
    fn(torch.ones(3))
    ~~^^^^^^^^^^^^^^^
  File "test_error_messages.py", line N, in fn
    torch._dynamo.graph_break()
    ~~~~~~~~~~~~~~~~~~~~~~~~~^^
""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
