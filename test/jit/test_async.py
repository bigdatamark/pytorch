# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] TorchScript tests for asynchronous execution: fork and wait in plain Python,
# scripted functions and modules, nesting, multiple waits and forks, keyword arguments, tracing
# across forks, error propagation with source highlights, gradient-mode propagation into forked
# tasks, future subtyping and error messages. Import-time side effect: appends the test directory
# to the module search path. Raises an error if run directly; run via the main jit test file.

import os
import sys
from typing import Any, Tuple

import torch
import torch.nn as nn


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from typing import List

from torch import Tensor
from torch.jit import Future
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import _inline_everything, JitTestCase


class TestAsync(JitTestCase):
    # [CLASS DOC] JIT test case grouping the fork, wait and future tests.
    def test_async_python(self):
        # [FUNCTION DOC] Forks a scripted function from plain Python, calls it directly, and waits
        # on the future. Asserts nothing; only checks the Python fallback path of fork and wait
        # works.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the negation of x.
            return torch.neg(x)

        x = torch.rand(3, 4)
        fut = torch.jit.fork(foo, x)
        y_hat = foo(x)
        y = torch.jit.wait(fut)
        # assert nothing; only to make sure the fake python path works

    def test_async_future_type_python(self):
        # [FUNCTION DOC] Calls a plain Python function that annotates a list of futures of
        # tensors, forks an identity function five times and waits on each. Asserts nothing;
        # checks Python parsing of the future type annotation.
        def foo(inp):
            # [FUNCTION DOC] HELPER: Forks an identity function five times, collects the futures,
            # waits on each and returns the list of results.
            futures = torch.jit.annotate(List[torch.jit.Future[torch.Tensor]], [])
            for _ in range(5):
                futures.append(torch.jit.fork(lambda x: x, inp))
            all_outputs = []
            for future in futures:
                all_outputs.append(torch.jit.wait(future))
            return all_outputs

        # assert nothing, just to make sure python type parsing works
        foo(torch.randn(3, 4))

    def test_async_parsing(self):
        # [FUNCTION DOC] Scripts a function that forks another scripted function three times using
        # an explicit future-of-list-of-tensors annotation, then waits on all. Checks three
        # results come back.
        @torch.jit.script
        def foo(x: Tensor) -> List[Tensor]:
            # [FUNCTION DOC] HELPER: Scripted function returning a list of the negated input and
            # its transpose.
            return [torch.neg(x), x.t()]

        @torch.jit.script
        def bar(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the list-returning function three
            # times with typed future annotations, then waiting on each and returning the
            # collected results.
            futures = torch.jit.annotate(List[Future[List[Tensor]]], [])
            for _ in range(3):
                future = torch.jit.annotate(
                    Future[List[Tensor]], torch.jit.fork(foo, x)
                )
                futures.append(future)

            output = torch.jit.annotate(List[List[Tensor]], [])
            for i in range(3):
                output.append(torch.jit.wait(futures[i]))
            return output

        x = torch.rand(3, 3)
        result = bar(x)
        self.assertEqual(len(result), 3)

    def test_async_script(self):
        # [FUNCTION DOC] Scripts a function that forks another scripted function, also calls it
        # directly, then waits. Asserts the awaited result equals the direct result.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning a pair of the negated input and
            # the input.
            return torch.neg(x), x

        x = torch.rand(3, 4)

        @torch.jit.script
        def wait_script(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the pair-returning function,
            # calling it directly too, waiting, and returning both results.
            fut = torch.jit.fork(foo, x)
            y_hat = foo(x)
            y = torch.jit.wait(fut)
            return y, y_hat

        y, y_hat = wait_script(x)

        self.assertEqual(y, y_hat)

    def test_async_script_capture(self):
        # [FUNCTION DOC] Forks a script method of a script module that uses a constant, a
        # parameter and the inputs, with optimized execution disabled. Asserts the awaited and
        # direct results match.
        class Mod(torch.jit.ScriptModule):
            # [FUNCTION DOC] HELPER: Script module with an integer constant and a random 2 by 2
            # parameter.
            __constants__ = ["const"]

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets the constant to 42 and creates the parameter.
                super().__init__()
                self.const = 42
                self.param = nn.Parameter(torch.randn(2, 2))

            @torch.jit.script_method
            def foo(self, x1, x2):
                # [FUNCTION DOC] HELPER: Script method returning the negated first input, the
                # parameter, the constant, the negated second input and the parameter again.
                return torch.neg(x1), self.param, self.const, torch.neg(x2), self.param

            @torch.jit.script_method
            def forward(self, x1, x2):
                # [FUNCTION DOC] HELPER: Script method forking the other method, calling it
                # directly, waiting, and returning both results.
                fut = torch.jit.fork(self.foo, x1, x2)
                y_hat = self.foo(x1, x2)
                y = torch.jit.wait(fut)
                return y, y_hat

        x1 = torch.rand(3, 4)
        x2 = torch.rand(5, 6)

        m = Mod()

        with torch.jit.optimized_execution(False):
            y, y_hat = m.forward(x1, x2)

        self.assertEqual(y, y_hat)

    def test_async_script_nested(self):
        # [FUNCTION DOC] Scripts a function that forks a scripted function that itself forks
        # another, using the private fork and wait. Asserts the nested awaited result equals the
        # direct result.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the negated input and the input.
            return torch.neg(x), x

        x = torch.rand(3, 4)

        @torch.jit.script
        def wait_script(x):
            # [FUNCTION DOC] HELPER: Scripted function forking and awaiting the pair function and
            # returning both the awaited and direct results.
            fut = torch.jit._fork(foo, x)
            y_hat = foo(x)
            y = torch.jit._wait(fut)
            return y, y_hat

        @torch.jit.script
        def wait_script_nest(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the previous wrapper and returning
            # its awaited value.
            fut = torch.jit._fork(wait_script, x)
            return torch.jit._wait(fut)

        y, y_hat = wait_script_nest(x)

        self.assertEqual(y, y_hat)

    def test_async_script_no_script_mod(self):
        # [FUNCTION DOC] Expects scripting a function that forks a plain tensor argument (not
        # callable) to fail with a runtime error saying a value cannot be called, with the
        # highlighted source text being the fork call.
        x = torch.rand(3, 4)

        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "cannot call a value", "torch.jit._fork(x"
        ):

            @torch.jit.script
            def wait_script(x):
                # [FUNCTION DOC] HELPER: Function whose compilation is expected to fail; forks its
                # argument and returns the future.
                fut = torch.jit._fork(x)
                return fut

    def test_async_script_multi_waits(self):
        # [FUNCTION DOC] Scripts a function that waits twice on the same future. Asserts both
        # results are equal.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning the sum of the transposed
            # negation of x and x.
            return torch.neg(x).t() + x

        @torch.jit.script
        def wait_script(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the helper, waiting twice on the
            # same future and returning both values.
            fut = torch.jit._fork(foo, x)

            # wait twice on the same future
            y1 = torch.jit._wait(fut)
            y2 = torch.jit._wait(fut)
            return y1, y2

        x = torch.rand(2, 2)
        y1, y2 = wait_script(x)
        self.assertEqual(y1, y2)

    def test_async_script_multi_forks(self):
        # [FUNCTION DOC] Scripts a function that forks three different helper functions and two
        # more forks it never waits on. Asserts the waited results match direct calls of the
        # helpers.
        @torch.jit.script
        def foo1(x):
            # [FUNCTION DOC] HELPER: Scripted helper of one argument: negation transposed, plus x.
            return torch.neg(x).t() + x

        @torch.jit.script
        def foo2(x, y):
            # [FUNCTION DOC] HELPER: Scripted helper of two arguments: negated-transposed x plus x
            # plus negated-transposed y.
            return torch.neg(x).t() + x + torch.neg(y).t()

        @torch.jit.script
        def foo3(x, y, z):
            # [FUNCTION DOC] HELPER: Scripted helper of three arguments: negated-transposed z plus
            # transposed y plus x.
            return torch.neg(z).t() + y.t() + x

        x1 = torch.rand(10, 10)
        x2 = torch.rand(10, 10)
        x3 = torch.rand(10, 10)

        @torch.jit.script
        def wait_script(x1, x2, x3):
            # [FUNCTION DOC] HELPER: Scripted function creating five forks, waiting on only the
            # first three and returning their results.
            f1 = torch.jit._fork(foo1, x1)
            f2 = torch.jit._fork(foo2, x1, x2)
            f3 = torch.jit._fork(foo3, x1, x2, x3)
            f4 = torch.jit._fork(foo1, x2)
            f5 = torch.jit._fork(foo2, x2, x3)

            # ignore some forks
            y1 = torch.jit._wait(f1)
            y2 = torch.jit._wait(f2)
            y3 = torch.jit._wait(f3)

            return y1, y2, y3

        y1, y2, y3 = wait_script(x1, x2, x3)
        self.assertEqual(y1, foo1(x1))
        self.assertEqual(y2, foo2(x1, x2))
        self.assertEqual(y3, foo3(x1, x2, x3))

    def test_async_kwargs(self):
        # [FUNCTION DOC] Checks fork and wait with positional and keyword argument permutations,
        # both in plain functions and their traced versions, and in scripted functions that fork
        # with positional or keyword arguments. Each wrapper is called with several argument
        # spellings and its result must equal the direct call of the underlying function.
        def foo(x1, x2):
            # [FUNCTION DOC] HELPER: Returns twice the first argument plus the second.
            return 2 * x1 + x2

        x1 = torch.rand(3, 4)
        x2 = torch.rand(3, 4)
        y_hat = foo(x1, x2)

        # Cover tracing and bare functions with permutations of args, kwargs
        for func in [
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x1, x2)),
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x1, x2=x2)),
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x1=x1, x2=x2)),
            lambda x1, x2: torch.jit._wait(torch.jit._fork(foo, x2=x2, x1=x1)),
        ]:
            for wrapper in [
                func,
                torch.jit.trace(func, (x1, x2)),
            ]:
                self.assertEqual(wrapper(x1, x2), y_hat)
                self.assertEqual(wrapper(x1, x2=x2), y_hat)
                self.assertEqual(wrapper(x1=x1, x2=x2), y_hat)
                self.assertEqual(wrapper(x2=x2, x1=x1), y_hat)

        # Cover scripting
        @torch.jit.script
        def foo_script_args(x1, x2):
            # [FUNCTION DOC] HELPER: Scripted function forking the plain helper with positional
            # arguments and returning the awaited result.
            return torch.jit._wait(torch.jit._fork(foo, x1, x2))

        @torch.jit.script
        def foo_script_kwargs(x1, x2):
            # [FUNCTION DOC] HELPER: Scripted function forking the plain helper with keyword
            # arguments and returning the awaited result.
            return torch.jit._wait(torch.jit._fork(foo, x1=x1, x2=x2))

        for wrapper in [
            foo_script_args,
            foo_script_kwargs,
        ]:
            self.assertEqual(wrapper(x1, x2), y_hat)
            self.assertEqual(wrapper(x1, x2=x2), y_hat)
            self.assertEqual(wrapper(x1=x1, x2=x2), y_hat)
            self.assertEqual(wrapper(x2=x2, x1=x1), y_hat)

    @_inline_everything
    def test_async_script_trace(self):
        # [FUNCTION DOC] Traces a module that contains a script module which forks a traced
        # submodule and a builtin negation, returning a nested structure; the wrapper flattens it
        # into a tuple. Runs with the inline-everything mode enabled by the decorator (which
        # discards the return value). Checks the graph has two fork nodes, one negation in the
        # root graph and three including subgraphs, and that outputs equal expected negation and
        # identity values.
        class Traced(nn.Module):
            # [CLASS DOC] HELPER: Plain module returning a pair of the negated input and the
            # input.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the negated input and the input as a tuple.
                return (torch.neg(x), x)

        class Mod(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module holding a traced version of the pair module and
            # forking it.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Traces the pair module on a random 3 by 3 tensor with
                # forced out-of-place ops and stores it as an attribute.
                super().__init__()
                x = torch.rand(3, 3)
                self.traced = torch.jit.trace(Traced(), (x), _force_outplace=True)

            @torch.jit.script_method
            def forward(
                self, x: Tensor
            ) -> Tuple[List[Tensor], Tuple[Tensor, Tensor], Tensor]:
                # [FUNCTION DOC] HELPER: Script method forking the traced module and a builtin
                # negation, waiting on both, and returning a list of tensors, the tuple result and
                # its second element.
                future1 = torch.jit._fork(self.traced, x)
                future2 = torch.jit._fork(torch.neg, x)

                tensor_tuple = torch.jit._wait(future1)
                tensor_single = torch.jit._wait(future2)

                tensor_list = []
                tensor_list.append(tensor_tuple[0])
                tensor_list.append(tensor_single)

                # return a nested structure of tensors
                return (tensor_list, tensor_tuple, tensor_tuple[1])

        class TupleCl(nn.Module):
            # [CLASS DOC] HELPER: Plain module wrapping the script module and flattening its
            # nested outputs.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a new script module as an attribute.
                super().__init__()
                self.module = Mod()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a six-element tuple: the negated input followed
                # by five elements flattened out of the nested output of the wrapped module.
                z = torch.neg(x)
                y = self.module(x)
                list = [z, y[0][0], y[0][1], y[1][0], y[1][1], y[2]]
                return tuple(list)

        x = torch.rand(3, 3)
        module = torch.jit.trace(TupleCl(), (x), _force_outplace=True)

        # Make sure we have forks
        self.assertGraphContainsExactly(
            module.graph, kind="prim::fork", num_kind_nodes=2
        )
        # Make sure 1 ::neg is in the root graph and 2 ::negs are in the subgraphs
        self.assertGraphContainsExactly(
            module.graph, kind="aten::neg", num_kind_nodes=1
        )
        self.assertGraphContainsExactly(
            module.graph, kind="aten::neg", num_kind_nodes=3, consider_subgraphs=True
        )

        y = torch.neg(x)
        self.assertEqual(module(x), (y, y, y, y, x, x))

    def test_async_script_error(self):
        # [FUNCTION DOC] Checks error propagation across forks: an error raised in a scripted
        # function is reported directly, through one fork (highlighting the fork call), and
        # through two nested forks with a different error for a three-dimensional input. Messages
        # are matched by regular expression together with the highlighted source text.
        x = torch.rand(3, 4)

        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function that fails for a non-square input by adding
            # a transpose of x to x.
            # error here
            return x.t() + x

        @torch.jit.script
        def wait_script(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the failing function and returning
            # the wait.
            fut = torch.jit._fork(foo, x)
            return torch.jit._wait(fut)

        @torch.jit.script
        def wait_script_nest(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the previous wrapper and returning
            # the wait.
            fut = torch.jit._fork(wait_script, x)
            return torch.jit._wait(fut)

        # no future
        error_msg = "The size.*must match the size of tensor"
        with self.assertRaisesRegexWithHighlight(Exception, error_msg, "x.t() + x"):
            foo(x)

        # one future
        with self.assertRaisesRegexWithHighlight(
            Exception, error_msg, "torch.jit._fork(foo, x"
        ):
            wait_script(x)

        # two futures with a different error
        x = torch.rand(3, 4, 5)
        with self.assertRaisesRegexWithHighlight(
            Exception,
            "expects a tensor with <= 2 dimensions",
            "torch.jit._fork(wait_script, x",
        ):
            wait_script_nest(x)

    def test_async_grad_guard_with_grad(self):
        # [FUNCTION DOC] Runs a scripted function that forks a task and reads requires_grad inside
        # the fork and in the caller, under gradient mode enabled. Asserts both are true.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning whether twice x requires grad.
            y = x * 2
            return y.requires_grad

        @torch.jit.script
        def bar(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the previous function and returning
            # the awaited flag with the caller's own requires_grad flag.
            fut = torch.jit._fork(foo, x)
            requires_grad_in_fork = torch.jit._wait(fut)
            z = x * 2
            return (requires_grad_in_fork, z.requires_grad)

        x = torch.randn(3, requires_grad=True)

        with torch.enable_grad():
            (inside_fork, after_wait) = bar(x)

        self.assertEqual(inside_fork, True)
        self.assertEqual(after_wait, True)

    def test_async_grad_guard_no_grad(self):
        # [FUNCTION DOC] Same as the enabled case but called under no-grad mode. Asserts both
        # flags are false, so the mode propagates into the forked task.
        @torch.jit.script
        def foo(x):
            # [FUNCTION DOC] HELPER: Scripted function returning whether twice x requires grad.
            y = x * 2
            return y.requires_grad

        @torch.jit.script
        def bar(x):
            # [FUNCTION DOC] HELPER: Scripted function forking the previous function and returning
            # the awaited flag with the caller's own requires_grad flag.
            fut = torch.jit._fork(foo, x)
            requires_grad_in_fork = torch.jit._wait(fut)
            z = x * 2
            return (requires_grad_in_fork, z.requires_grad)

        x = torch.randn(3, requires_grad=True)

        with torch.no_grad():
            (inside_fork, after_wait) = bar(x)

        self.assertEqual(inside_fork, False)
        self.assertEqual(after_wait, False)

    def test_trace_fork_wait(self):
        # [FUNCTION DOC] Traces a function that forks and waits and checks that outputs match
        # eager. Checks the traced graph has one fork, one wait, and two negation nodes when
        # subgraphs count.
        def fork_body(x):
            # [FUNCTION DOC] HELPER: Returns the negation of x and the negation plus one.
            return x.neg(), x.neg() + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Forks the body, waits, and returns its two values and x minus
            # one.
            fut = torch.jit._fork(fork_body, x)
            vals = torch.jit._wait(fut)
            return vals[0], vals[1], x - 1

        traced = torch.jit.trace(fn, (torch.rand(3, 4),))
        x = torch.rand(3, 4)
        self.assertEqual(fn(x), traced(x))

        self.assertGraphContainsExactly(
            traced.graph, kind="prim::fork", num_kind_nodes=1
        )
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::wait", num_kind_nodes=1
        )
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::neg", num_kind_nodes=2, consider_subgraphs=True
        )

    def test_trace_fork_wait_leaking(self):
        # [FUNCTION DOC] Traces a function whose forked body appends its result to an outer list
        # that the function then returns. Expects a runtime error that the output had no
        # observable data dependence with the trace inputs.
        my_list = []

        def fork_body(x):
            # [FUNCTION DOC] HELPER: Appends x plus one to the enclosing list and returns x plus
            # one.
            my_list.append(x + 1)
            return x + 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Forks the body, waits, then returns the first element of the
            # enclosing list rather than the awaited value.
            fut = torch.jit._fork(fork_body, x)
            val = torch.jit._wait(fut)
            return my_list[0]

        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            "did not have observable data dependence with trace inputs; "
            "this probably indicates your program cannot be understood "
            "by the tracer.",
            "",
        ):
            traced = torch.jit.trace(fn, (torch.rand(3, 4),), check_trace=False)

    def test_trace_fork_wait_inline(self):
        # [FUNCTION DOC] Traces a fork and wait function, runs the pass inlining fork and wait on
        # the graph, and checks no fork or wait nodes remain and two add nodes exist.
        def fork_body(x):
            # [FUNCTION DOC] HELPER: Returns a pair of x plus one and x plus two.
            return x + 1, x + 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Forks the body, waits, and returns the second element.
            fut = torch.jit._fork(fork_body, x)
            val = torch.jit._wait(fut)
            return val[1]

        traced = torch.jit.trace(fn, (torch.rand(3, 4),))
        torch._C._jit_pass_inline_fork_wait(traced.graph)
        self.assertGraphContainsExactly(
            traced.graph, kind="prim::fork", num_kind_nodes=0
        )
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::wait", num_kind_nodes=0
        )
        self.assertGraphContainsExactly(
            traced.graph, kind="aten::add", num_kind_nodes=2
        )

    def test_trace_fork_wait_list_modulecalls(self):
        # [FUNCTION DOC] Runs the trace-checking helper on a module that calls a submodule
        # returning a list of futures, to check tracing of module calls returns the right type.
        def add_one(input):
            # [FUNCTION DOC] HELPER: Returns input plus a ones tensor of the same size.
            return input + torch.ones(input.size())

        class TestListFutureModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning a list of three futures, each forking the add
            # helper.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Builds a list with the input three times and returns a
                # list of forked add-helper futures, one per element.
                input_list = []
                for _ in range(3):
                    input_list.append(input)

                fut_list: List[Future[torch.Tensor]] = []
                for input_tensor in input_list:
                    fut_list.append(torch.jit._fork(add_one, input_tensor))
                # return list[future[tensor]] here to ensure tracing
                # module calls return the correct types
                return fut_list

        class TestModuleWrapper(nn.Module):
            # [CLASS DOC] HELPER: Module calling the list-of-futures submodule and summing the
            # awaited results onto the input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the list-of-futures submodule as an attribute.
                super().__init__()
                self.list_fut_mod = TestListFutureModule()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the submodule, then adds each future's awaited
                # value to the input and returns the sum.
                fut_list = self.list_fut_mod(input)
                res = input
                for fut in fut_list:
                    res = res + fut.wait()
                return res

        self.checkTrace(TestModuleWrapper(), (torch.randn(5, 5),))

    def test_trace_modulecalls_with_different_output_types(self):
        # [FUNCTION DOC] Runs the trace-checking helper on a module whose submodule returns both a
        # tensor and a future, to check module-call tracing with mixed output types.
        def add_one(input):
            # [FUNCTION DOC] HELPER: Returns input plus a ones tensor of the same size.
            return input + torch.ones(input.size())

        class DifferentOutputModule(nn.Module):
            # [CLASS DOC] HELPER: Module returning the input and a future from forking the add
            # helper.
            def forward(self, input):
                # [FUNCTION DOC] HELPER: Forks the add helper and returns the input with that
                # future.
                fut_res = torch.jit._fork(add_one, (input))

                # return different types from module call
                return input, fut_res

        class TestModule(nn.Module):
            # [CLASS DOC] HELPER: Module adding the awaited future from the submodule onto its
            # tensor output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the mixed-output submodule as an attribute.
                super().__init__()
                self.gen_output = DifferentOutputModule()

            def forward(self, input):
                # [FUNCTION DOC] HELPER: Calls the submodule, adds the awaited future value to the
                # tensor output and returns the sum.
                res, fut_res = self.gen_output(input)
                res = res + fut_res.wait()
                return res

        self.checkTrace(TestModule(), (torch.randn(5, 5),))

    def test_no_future_subtype_message(self):
        # [FUNCTION DOC] Expects scripting a function that annotates a list of futures without a
        # contained type to raise a runtime error about a future lacking a contained type.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError, "Future without a contained type", ""
        ):

            @torch.jit.script
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Function whose compilation is expected to fail; creates a
                # list of untyped futures.
                futs = torch.jit.annotate(List[torch.jit.Future], [])

    def test_future_subtyping(self):
        # [FUNCTION DOC] Checks that futures subtype properly: a function returning a future of
        # any is accepted from a fork of an int function and scripted code can wait on it. Then
        # expects scripting code that treats a future of int as a future of float to raise a
        # runtime error saying the annotation does not match the actual type.
        """
        Test that futures subtype each other properly.
        """

        # Successful subtyping.
        def returns_int(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns x plus x plus one.
            return x + x + 1

        def returns_future_any(x: int) -> torch.jit.Future[Any]:
            # [FUNCTION DOC] HELPER: Forks the int function and returns the future annotated as
            # future of any.
            return torch.jit._fork(returns_int, (x))

        @torch.jit.script
        def fn_int(x: int) -> Any:
            # [FUNCTION DOC] HELPER: Scripted function returning the awaited value of the
            # future-of-any helper.
            fut = returns_future_any(x)
            return fut.wait()

        # Unsuccessful subtyping.
        with self.assertRaisesRegexWithHighlight(
            RuntimeError,
            r"was annotated as having type Future\[float\] but is actually of type Future\[int\]",
            "fut = returns_future_float(x",
        ):

            def returns_future_float(x: int) -> torch.jit.Future[float]:
                # [FUNCTION DOC] HELPER: Annotated as returning a future of float but forks the
                # int function.
                return torch.jit._fork(returns_int, (x))

            @torch.jit.script
            def fn_float(x: int) -> Any:
                # [FUNCTION DOC] HELPER: Scripted function expected to fail to compile; awaits the
                # future-of-float helper.
                fut = returns_future_float(x)
                return fut.wait()


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
