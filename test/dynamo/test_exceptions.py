# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for tracing Python exception handling (raise, try/except/else/finally,
# chaining, context managers, generators, exception groups, custom exception classes, builtin
# argument errors, speculation log divergence) under torch.compile with the eager backend,
# usually comparing against eager results. Defines a few custom exception classes at module
# level and the ExceptionTests class, which is instantiated as parametrized tests at the end.

import contextlib
import dataclasses
import operator
import sys

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.nn
import torch.utils.checkpoint
from torch._dynamo.bytecode_transformation import Instruction
from torch._dynamo.exc import Unsupported
from torch._dynamo.symbolic_convert import SpeculationLog, SpeculationLogDivergence
from torch._dynamo.testing import CompileCounter
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class CustomException(Exception):
    # [CLASS DOC] HELPER: plain Exception subclass with no extra behavior, used as a test type.
    pass


class CustomExceptionMeta(type):
    # [CLASS DOC] HELPER: metaclass whose instance check reports every object as an instance
    # of the classes it creates.
    def __instancecheck__(cls, instance):
        # [FUNCTION DOC] HELPER: always returns True regardless of the instance argument.
        return True


class CustomExceptionWithInstanceCheck(Exception, metaclass=CustomExceptionMeta):
    # [CLASS DOC] HELPER: Exception subclass using the always-true instance check metaclass.
    pass


class CustomExceptionWithArgs(Exception):
    # [CLASS DOC] HELPER: Exception subclass storing two attributes, a and b (b optional).
    def __init__(self, a, b=None):
        # [FUNCTION DOC] HELPER: stores a and b as attributes; does not call the base init, so
        # the args tuple of the exception is not populated by it.
        self.a = a
        self.b = b


class MyException(OSError):
    # [CLASS DOC] HELPER: OSError subclass with no extra behavior, used as a test type.
    pass


class ExceptionTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Main dynamo test case for exception handling; tests are stateless and mostly
    # compile a function with the eager backend and compare to the eager result.
    def test_exception(self):
        # [FUNCTION DOC] Compiles (fullgraph) a function that raises NotImplementedError in a
        # try block and handles it with a broad except; result must equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: cos, sin, raise and catch, then sigmoid; returns tensor.
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError
            except Exception:
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception2(self):
        # [FUNCTION DOC] Like the basic test but the except clause lists a tuple of exception
        # types; compiled fullgraph result must equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: raises NotImplementedError and catches via a type tuple.
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError
            except (NotImplementedError, AttributeError):
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception3(self):
        # [FUNCTION DOC] Checks multiple except clauses plus a finally block: the matching
        # handler runs and finally always runs; compiled fullgraph result must equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: raises NotImplementedError, handled by the second except,
            # then finally applies one more cos.
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError("Not implemented")
            except AssertionError:
                x = torch.sigmoid(x)
            except NotImplementedError:
                x = torch.cos(x)
            finally:
                x = torch.cos(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception4(self):
        # [FUNCTION DOC] Checks try/except inside a loop with an early return; compiled
        # fullgraph result must equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: loops ten times, returning at the fifth iteration; each
            # earlier iteration applies sin, raises and catches, then applies sigmoid.
            for i in range(10):
                if i == 5:
                    return x
                try:
                    x = torch.sin(x)
                    raise NotImplementedError
                except Exception:
                    x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_with_another_exception(self):
        # [FUNCTION DOC] Checks a second exception raised and caught inside an except handler;
        # compiled fullgraph run must equal eager (function returns None).
        def fn(x):
            # [FUNCTION DOC] HELPER: raises NotImplementedError, and in its handler raises and
            # catches AssertionError; returns None.
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                raise NotImplementedError("Not implemented")
            except NotImplementedError:
                x = torch.sigmoid(x)
                try:
                    x = torch.cos(x)
                    raise AssertionError
                except AssertionError:
                    x = torch.cos(x)

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_with_vars(self):
        # [FUNCTION DOC] Checks that the vars builtin applied to an int raises TypeError inside
        # compiled code (fullgraph) and is catchable, matching eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: calls vars on an int, expecting TypeError; returns sin.
            try:
                vars(42)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_vars_too_many_args_type_error(self):
        # [FUNCTION DOC] Compiled code calling vars with two arguments must raise TypeError
        # with the message that at most one argument is expected, got two.
        def fn(x):
            # [FUNCTION DOC] HELPER: returns x plus one and the result of vars with two args.
            return x + 1, vars(1, 2)

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(
            TypeError, "vars expected at most 1 argument, got 2"
        ):
            opt_fn(torch.ones(1))

    def test_vars_keyword_args_type_error(self):
        # [FUNCTION DOC] Checks that vars called with a keyword argument raises TypeError in
        # compiled fullgraph code, matching eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: calls vars with a keyword argument, catches TypeError.
            try:
                vars(obj=x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_range_too_many_args_type_error(self):
        # [FUNCTION DOC] Compiled code calling range with six arguments must raise TypeError
        # saying at most three arguments are expected, got six.
        def fn(x):
            # [FUNCTION DOC] HELPER: returns x plus one and a range built with six arguments.
            return x + 1, range(1, 2, 3, 4, 5, 6)

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaisesRegex(
            TypeError, "range expected at most 3 arguments, got 6"
        ):
            opt_fn(torch.ones(1))

    def test_raise_non_exception_type_error(self):
        # [FUNCTION DOC] Raising a non-exception builtin type (int) must give a catchable
        # TypeError under fullgraph compile, with the same result and message as eager.
        # PyExceptionClass_Check must reject non-exception builtins: they are
        # BuiltinVariables but not BaseException subclasses.
        def fn(x):
            # [FUNCTION DOC] HELPER: raises the int type, catches TypeError, returns sin of x
            # and the error text.
            try:
                raise int
            except TypeError as e:
                return x.sin(), str(e)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_raise_from_non_exception_type_error(self):
        # [FUNCTION DOC] Raising ValueError chained from a non-exception type (dict) must give a
        # catchable TypeError under fullgraph compile, matching eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: raises with a bad cause, catches TypeError, returns sin of
            # x and the error text.
            try:
                raise ValueError("v") from dict
            except TypeError as e:
                return x.sin(), str(e)

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_builtin_arg_count_type_errors(self):
        # [FUNCTION DOC] Defines ten functions that each call a builtin (len, operator getitem,
        # next, range) with a wrong number or form of arguments inside a try block expecting
        # TypeError, and checks each compiles under fullgraph and matches eager.
        def check(fn):
            # [FUNCTION DOC] HELPER: compiles fn fullgraph with the eager backend and asserts
            # its output on a random tensor equals the eager output.
            x = torch.randn(4)
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            self.assertEqual(fn(x), opt_fn(x))

        def len_no_args(x):
            # [FUNCTION DOC] HELPER: calls len with no args, expects TypeError, returns sin.
            try:
                len()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def len_too_many_args(x):
            # [FUNCTION DOC] HELPER: calls len with two args, expects TypeError, returns sin.
            try:
                len(x, x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_too_few_args(x):
            # [FUNCTION DOC] HELPER: operator getitem with one arg, expects TypeError.
            try:
                operator.getitem(x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_no_args(x):
            # [FUNCTION DOC] HELPER: operator getitem with no args, expects TypeError.
            try:
                operator.getitem()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_too_many_args(x):
            # [FUNCTION DOC] HELPER: operator getitem with three args, expects TypeError.
            try:
                operator.getitem(x, 0, 1)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def getitem_keyword_args(x):
            # [FUNCTION DOC] HELPER: operator getitem with keyword args, expects TypeError.
            try:
                operator.getitem(a=x, b=0)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def next_no_args(x):
            # [FUNCTION DOC] HELPER: calls next with no args, expects TypeError, returns sin.
            try:
                next()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def next_too_many_args(x):
            # [FUNCTION DOC] HELPER: calls next with three args, expects TypeError.
            try:
                next(iter([x]), x, x)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def range_no_args(x):
            # [FUNCTION DOC] HELPER: calls range with no args, expects TypeError, returns sin.
            try:
                range()
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        def range_too_many_args(x):
            # [FUNCTION DOC] HELPER: calls range with six args, expects TypeError.
            try:
                range(1, 2, 3, 4, 5, 6)
                raise RuntimeError("Should not be raised")
            except TypeError:
                return x.sin()

        check(len_no_args)
        check(len_too_many_args)
        check(getitem_no_args)
        check(getitem_too_few_args)
        check(getitem_too_many_args)
        check(getitem_keyword_args)
        check(next_no_args)
        check(next_too_many_args)
        check(range_no_args)
        check(range_too_many_args)

    def test_user_class_as_tensor_method_arg(self):
        # [FUNCTION DOC] Passing a user class as the size to a tensor new_full method raises
        # TypeError; the compiled function must catch it and give the same result as eager.
        class MyClass:
            # [CLASS DOC] HELPER: empty user class passed as a bad argument.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: tries new_full with the class, falling back to x plus 1.0.
            try:
                y = x.new_full(MyClass, 3.14)
            except TypeError:
                y = x + 1.0
            return y

        x = torch.ones(4)
        ref = fn(x)
        res = torch.compile(fn, backend="eager")(x)
        self.assertEqual(ref, res)

    def test_autocast_with_exception(self):
        # [FUNCTION DOC] An autograd Function whose forward raises NotImplementedError is
        # applied inside an autocast region within a try block in compiled code; the handler
        # must run and return the input plus one.
        class Optimizer(torch.autograd.Function):
            # [CLASS DOC] HELPER: autograd Function with a forward that always raises.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: always raises NotImplementedError.
                raise NotImplementedError("Not implemented")

            @staticmethod
            def backward(ctx, grad_out):
                # [FUNCTION DOC] HELPER: returns the incoming gradient unchanged.
                return grad_out

        @torch.compile(backend="eager")
        def f(x: torch.Tensor):
            # [FUNCTION DOC] HELPER: applies the failing Function under CPU autocast; on
            # NotImplementedError returns x plus one.
            try:
                with torch.autocast(device_type="cpu", dtype=None):
                    Optimizer.apply(x)
            except NotImplementedError:
                return x + 1

        inp = torch.ones(3)
        out = f(inp)
        self.assertTrue(torch.equal(out, inp + 1))

    @make_dynamo_test
    def test_isinstance_CustomException(self):
        # [FUNCTION DOC] Runs under the dynamo test harness: checks the class is a type, an
        # instance is not, and the always-true metaclass makes the class an instance of itself.
        assert isinstance(CustomException, type)  # noqa: S101
        assert not isinstance(CustomException(), type)  # noqa: S101
        C = CustomExceptionWithInstanceCheck
        assert isinstance(C, C)  # noqa: S101
        assert isinstance(C(), C)  # noqa: S101

    @make_dynamo_test
    def test_propagate_exception_inside_ctx_manager(self):
        # [FUNCTION DOC] Runs under the dynamo test harness: an IndexError raised inside a
        # generator context manager that converts any exception to ValueError must surface as
        # ValueError (z becomes 1) even nested within another context manager.
        @contextlib.contextmanager
        def cm():
            # [FUNCTION DOC] HELPER: context manager that turns any exception thrown in
            # into a ValueError.
            try:
                yield
            except BaseException:
                raise ValueError  # noqa: B904

        @contextlib.contextmanager
        def nothing():
            # [FUNCTION DOC] HELPER: context manager that yields with a no-op finally.
            try:
                yield
            finally:
                pass

        z = 0
        with nothing():
            try:
                with cm():
                    raise IndexError
            except ValueError:
                z = 1
            except IndexError:
                z = 2
            assert z == 1  # noqa: S101

    def test_exception_else(self):
        # [FUNCTION DOC] Checks try/except/else where no exception occurs so the else branch
        # runs; compiled fullgraph result must equal eager.
        def gn(x):
            # [FUNCTION DOC] HELPER: returns cos of x.
            return torch.cos(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: cos, then in try sin and gn; else branch applies cos.
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                x = gn(x)
            except Exception:
                x = torch.sigmoid(x)
            else:
                x = torch.cos(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @make_dynamo_test
    def test_raise_match(self):
        # [FUNCTION DOC] Runs under the dynamo test harness. Emulates ExitStack-style exception
        # context fixing: inside a context manager cleanup, two raise/finally-raise pairs are run
        # and their contexts relinked, then the last is re-raised. Asserts the escaping exception
        # is an AttributeError whose context chain is BytesWarning, ConnectionError,
        # DeprecationWarning, Exception in that order.
        a = AttributeError
        b = BytesWarning
        c = ConnectionError
        d = DeprecationWarning
        e = Exception

        def fn(a, b):
            # [FUNCTION DOC] HELPER: raises exception a, and in the finally block raises b
            # (so b carries a as context).
            try:
                raise a
            finally:
                raise b

        def fix_exc_context(frame_exc, new_exc, old_exc):
            # [FUNCTION DOC] HELPER: walks the context chain of new_exc until reaching either
            # nothing or old_exc (returns, nothing to fix) or frame_exc; in the latter case sets
            # the context of the exception found there to old_exc. Mutates context links.
            # slightly change from ExitStack.fix_exc_context function
            while 1:
                exc_context = new_exc.__context__
                if exc_context is None or exc_context is old_exc:
                    return
                if exc_context is frame_exc:
                    break
                new_exc = exc_context
            new_exc.__context__ = old_exc

        @contextlib.contextmanager
        def ctx():
            # [FUNCTION DOC] HELPER: context manager whose cleanup runs two failing calls of
            # fn, fixes the context links after each, then re-raises the last exception with
            # its fixed context restored.
            try:
                yield
            finally:
                frame_exc = prev_exc = sys.exc_info()
                args = [(d, c), (b, a)]
                for x, y in args:
                    try:
                        fn(x, y)
                    except BaseException:
                        new_exc = sys.exc_info()
                        fix_exc_context(frame_exc[1], new_exc[1], prev_exc[1])
                        prev_exc = new_exc

                try:
                    fixed_ctx = prev_exc[1].__context__
                    raise prev_exc[1]
                except BaseException:
                    prev_exc[1].__context__ = fixed_ctx
                    raise

        try:
            with ctx():
                raise e
        except Exception as exc:
            assert isinstance(exc, a)  # noqa: S101
            assert isinstance(exc.__context__, b)  # noqa: S101
            assert isinstance(exc.__context__.__context__, c)  # noqa: S101
            assert isinstance(exc.__context__.__context__.__context__, d)  # noqa: S101
            assert isinstance(exc.__context__.__context__.__context__.__context__, e)  # noqa: S101

    # TODO(anijain2305) - does not work with fullgraph=True
    def test_exception_with_another_exception2(self):
        # [FUNCTION DOC] Compiles without fullgraph (bare re-raise is unsupported) a function
        # calling a helper that catches and re-raises; only checks the run completes.
        def gn(x):
            # [FUNCTION DOC] HELPER: raises NotImplementedError, catches it, and re-raises it.
            try:
                x = torch.cos(x)
                raise NotImplementedError("Not implemented")
            except NotImplementedError:
                x = torch.sigmoid(x)
                raise

        def fn(x):
            # [FUNCTION DOC] HELPER: calls gn in try, swallows any Exception, returns x.
            try:
                x = torch.cos(x)
                gn(x)
            except Exception:
                pass
            return x

        x = torch.randn(4)
        fn(x)
        # Can't use fullgraph=True because RERAISE is not supported
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)

    def test_exception_with_ctx_manager(self):
        # [FUNCTION DOC] Raises inside a no_grad block within try; compiled fullgraph result
        # must equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: raises NotImplementedError inside no_grad, handles it,
            # then applies sigmoid.
            x = torch.cos(x)
            try:
                with torch.no_grad():
                    x = torch.sin(x)
                    raise NotImplementedError("Not implemented")
            except NotImplementedError:
                x = torch.sigmoid(x)
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_raised_from_child(self):
        # [FUNCTION DOC] An exception raised in an inlined callee is caught by the caller;
        # compiled fullgraph result must equal eager.
        def gn():
            # [FUNCTION DOC] HELPER: always raises NotImplementedError.
            raise NotImplementedError("foo")

        def fn(x):
            # [FUNCTION DOC] HELPER: calls gn between sin ops; handler applies sigmoid.
            x = torch.cos(x)
            try:
                x = torch.sin(x)
                gn()
                x = torch.sin(x)
            except Exception:
                x = torch.sigmoid(x)

            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_dynamo_undo_kw_names(self):
        # [FUNCTION DOC] Regression-style check that a keyword-argument call raising an
        # exception does not leave stale keyword names that break a later dict items loop;
        # compiled fullgraph result must equal eager.
        def g(x, k=None):
            # [FUNCTION DOC] HELPER: raises TypeError when k is truthy, else returns sin of x.
            if k:
                raise TypeError("error")
            return x.sin()

        def fn(x):
            # [FUNCTION DOC] HELPER: calls g with k true inside try; on error sums the tensor
            # values of a one-entry dict and returns the sum.
            d = {"a": x}
            try:
                g(x, k=True)
            except Exception:
                y = 0
                for _, b in d.items():  # noqa: PERF102
                    y += b.sum()
            return y

        x = torch.randn(2, 3)
        expected = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        got = opt_fn(x)
        self.assertEqual(expected, got)

    def test_raise_custom_exception(self):
        # [FUNCTION DOC] Raises a locally defined Exception subclass in fullgraph compiled code
        # and checks the specific handler (not the generic one) runs, giving sin.
        class Exc(Exception):
            # [CLASS DOC] HELPER: local Exception subclass.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises Exc; the Exc handler returns sin, the generic
            # handler would return cos.
            try:
                raise Exc
            except Exc:
                return t.sin()
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    def test_raise_custom_exception_with_args(self):
        # [FUNCTION DOC] Raises a local Exception subclass with two args and checks the handler
        # reads them from the args tuple; result must be sin plus both args.
        class Exc(Exception):
            # [CLASS DOC] HELPER: local Exception subclass.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises Exc with an int and a float; handler returns sin
            # plus the two args.
            try:
                raise Exc(1, 2.0)
            except Exc as e:
                return t.sin() + e.args[0] + e.args[1]
            except Exception:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin() + 1 + 2.0)

    def test_nn_module_getattr(self):
        # [FUNCTION DOC] Attribute lookups that fall through custom getattr hooks with
        # AttributeError caught in a subclass work in compiled fullgraph code; result equals eager.
        class A:
            # [CLASS DOC] HELPER: base class with attribute _b and a getattr hook that resolves
            # a missing name through the underscore-prefixed attribute, else raises AttributeError.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: sets _b to twenty.
                self._b = 20

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: returns the instance attribute with an underscore
                # prefixed name if present; otherwise raises AttributeError.
                fixed_name = "_" + name
                if fixed_name in self.__dict__:
                    return self.__dict__[fixed_name]
                raise AttributeError(f"{name} absent")

        class B(A):
            # [CLASS DOC] HELPER: subclass of A with attribute a whose getattr hook falls back
            # to thirty when the parent hook raises AttributeError.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: sets a to ten (the parent init is not called, so
                # the attribute _b is absent).
                self.a = 10

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: delegates to the parent hook, returning thirty if
                # that raises AttributeError.
                try:
                    return super().__getattr__(name)
                except AttributeError:
                    return 30

        obj = B()

        def fn(x):
            # [FUNCTION DOC] HELPER: returns x times attributes a, b and c of obj.
            return x * obj.a * obj.b * obj.c

        x = torch.ones(4)
        ref = fn(x)
        print(ref)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_custom_getattr_on_module_exception(self):
        # [FUNCTION DOC] A module overriding getattr to alias a_copy to parameter a (re-raising
        # for other names) must give the same output compiled fullgraph as in eager.
        class Foo(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a parameter a (all twos) and a getattr hook
            # exposing a_copy as that parameter.
            def __init__(self, a=3):
                # [FUNCTION DOC] HELPER: registers parameter a of four twos;
                # the argument is unused.
                super().__init__()
                self.register_parameter("a", torch.nn.Parameter(torch.ones(4) * 2))

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: defers to the module lookup; on AttributeError returns
                # parameter a for a_copy, otherwise re-raises.
                try:
                    return super().__getattr__(name)  # defer to nn.Module's logic
                except AttributeError:
                    if name == "a_copy":
                        return self.a
                    raise

            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns x times a times a_copy.
                return x * self.a * self.a_copy

        mod = Foo()
        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)

        x = torch.ones(4)
        self.assertEqual(mod(x), opt_mod(x))

    def test_attribute_error_from_getattr(self):
        # [FUNCTION DOC] The builtin hasattr on an object whose getattr hook raises
        # AttributeError for unknown names returns false in compiled fullgraph code; result
        # equals eager.
        class Mock:
            # [CLASS DOC] HELPER: object with attribute a and a getattr hook that raises
            # AttributeError for any other name.
            def __init__(self):
                # [FUNCTION DOC] HELPER: sets a to five.
                self.a = 5

            def __getattr__(self, name):
                # [FUNCTION DOC] HELPER: raises AttributeError unless name is a, in which case
                # returns the stored value.
                if name != "a":
                    raise AttributeError("missing")
                return self.__dict__["a"]

        mock = Mock()

        def fn(x):
            # [FUNCTION DOC] HELPER: cos x if mock has attribute b, else sin x.
            if hasattr(mock, "b"):
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_stop_iteration(self):
        # [FUNCTION DOC] Compiles (fullgraph) a hand-written zip_longest that relies on catching
        # StopIteration from next, called on a four and a three element list; result must
        # equal eager.
        def zip_longest(*iterables, fillvalue=None):
            # [FUNCTION DOC] HELPER: builds iterators of all inputs; loops taking one element
            # from each, and when any is exhausted appends the fill value and returns the list
            # (so it stops at the shortest input, unlike the real function).
            # Get the iterators for each iterable
            iterators = [iter(it) for it in iterables]

            result = []
            while True:
                for it in iterators:
                    try:
                        value = next(it)
                    except StopIteration:
                        result.append(fillvalue)
                        return result
                    result.append(value)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: computes a throwaway cos of a random tensor, then returns
            # the zip_longest result as a tuple.
            torch.cos(torch.randn(4))
            return tuple(zip_longest(x, y))

        x = [1, 2, 3, 4]
        y = [10, 11, 12]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x, y)
        res = opt_fn(x, y)
        self.assertEqual(ref, res)

    def test_nn_reraise(self):
        # [FUNCTION DOC] A module whose forward raises ValueError and has a forward pre hook is
        # called through compile; the ValueError must propagate and the first recorded
        # compilation metric's fail reason must mention an observed exception. Clears the
        # metrics first.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward always raises ValueError.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: raises ValueError (the following return is dead code).
                raise ValueError("woof")
                return x + 2

        m = M()
        m.register_forward_pre_hook(lambda m, go: None)

        torch._dynamo.utils.clear_compilation_metrics()
        opt_call = torch.compile(lambda x: m(x), backend="eager")
        self.assertRaises(ValueError, lambda: opt_call(torch.randn(3)))
        metrics = torch._dynamo.utils.get_compilation_metrics()
        self.assertIn("Observed exception", metrics[0].fail_reason)

    def test_observed_exception_formats_fstring_message(self):
        # [FUNCTION DOC] A tree map callback raises ValueError with a formatted message for a
        # non-tensor leaf; under fullgraph compile Unsupported is raised and its text must show
        # the rendered message with the key path and must not leak internal variable repr names.
        from torch.utils._pytree import tree_map_with_path

        def check_tensor(path, x):
            # [FUNCTION DOC] HELPER: raises ValueError naming the path for non-tensors, else
            # returns the tensor doubled.
            if not isinstance(x, torch.Tensor):
                raise ValueError(f"Expected Tensor at {path=}")
            return x * 2

        def fn(tree):
            # [FUNCTION DOC] HELPER: applies check_tensor over every leaf with its path.
            return tree_map_with_path(check_tensor, tree)

        tree = {"a": torch.randn(10), "b": 5}

        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        with self.assertRaises(Unsupported) as compiled_ctx:
            compiled_fn(tree)

        exc_str = str(compiled_ctx.exception)
        self.assertIn("Observed exception", exc_str)
        self.assertIn("Expected Tensor at path=(MappingKey(key='b'),)", exc_str)
        self.assertNotIn("Failed to trace builtin operator", exc_str)
        self.assertNotIn("StringFormatVariable", exc_str)
        self.assertNotIn("ConstantVariable(", exc_str)

    def test_key_error(self):
        # [FUNCTION DOC] A missing dict key lookup inside try/except KeyError is handled in
        # compiled fullgraph code; result equals eager.
        def fn(x, d):
            # [FUNCTION DOC] HELPER: scales x by d's value for b, or by 2 if the key is missing.
            try:
                a = d["b"]
            except KeyError:
                a = 2
            return x * a

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        d = {"a": 1}
        ref = fn(x, d)
        res = opt_fn(x, d)
        self.assertEqual(ref, res)

    def test_atrribute_error(self):
        # [FUNCTION DOC] Reading a missing attribute of a user object inside try/except
        # AttributeError works in compiled code (graph breaks allowed); result equals eager.
        class Mock:
            # [CLASS DOC] HELPER: object with a single attribute a.
            def __init__(self):
                # [FUNCTION DOC] HELPER: sets attribute a to one.
                self.a = 1

        mock = Mock()

        def fn(x):
            # [FUNCTION DOC] HELPER: reads missing attribute b; on AttributeError uses factor 3
            # (else 2) and returns sin of x times the factor.
            try:
                c = 2
                mock.b
            except AttributeError:
                c = 3
            return torch.sin(x) * c

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_tensor_attribute_error_in_try_except(self):
        # [FUNCTION DOC] A module forward that reads a nonexistent tensor attribute and handles
        # AttributeError by scaling with a parameter; compiled output must equal eager.
        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module holding a one-element scale parameter initialized to two.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the scale parameter.
                super().__init__()
                self.scale = torch.nn.Parameter(torch.tensor([2.0]))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: tries a missing tensor attribute; on AttributeError
                # returns x times scale.
                try:
                    return x.this_attribute_does_not_exist
                except AttributeError:
                    return x * self.scale

        m = M()
        opt_m = torch.compile(m, backend="eager")
        x = torch.randn(4, 4)
        ref = m(x)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_runtime_error_in_try_except(self):
        # [FUNCTION DOC] Inverting a non-square matrix raises RuntimeError in eager; compiled
        # fullgraph code must catch it the same way and return x plus one.
        def fn(x):
            # [FUNCTION DOC] HELPER: tries a matrix inverse on a 2x3 input; on RuntimeError
            # returns x plus one.
            try:
                torch.linalg.inv(x)
            except RuntimeError:
                return x + 1
            return x

        opt_m = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        ref = fn(x)
        res = opt_m(x)
        self.assertEqual(ref, res)

    def test_runtime_error_graph_break(self):
        # [FUNCTION DOC] An uncaught RuntimeError from inverting a non-square matrix propagates
        # to the caller, and the counting backend saw zero compiled frames.
        cnt = CompileCounter()

        def fn(x):
            # [FUNCTION DOC] HELPER: returns the matrix inverse of x.
            return torch.linalg.inv(x)

        opt_m = torch.compile(fn, backend=cnt)
        x = torch.randn(2, 3)
        with self.assertRaises(RuntimeError):
            opt_m(x)
        self.assertEqual(cnt.frame_count, 0)

    def test_raise_from_None(self):
        # [FUNCTION DOC] A mapping-like object re-raises KeyError "from None"; compiled fullgraph
        # code catching it through a helper returns the default for a missing key; results
        # equal eager.
        # Inspired from os.environ
        class MyMapping:
            # [CLASS DOC] HELPER: wraps a dict and exposes item access that re-raises a fresh
            # KeyError without context.
            def __init__(self, d):
                # [FUNCTION DOC] HELPER: stores the dict in _d.
                self._d = d

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: returns the value for key; on KeyError raises a new
                # KeyError with the key, suppressing the original.
                try:
                    value = self._d[key]
                except KeyError:
                    raise KeyError(key) from None
                return value

        d = MyMapping({"a": 10, "b": 20})

        def mapping_get(obj, key, value=None):
            # [FUNCTION DOC] HELPER: returns obj item for key, or value if KeyError is raised.
            try:
                return obj.__getitem__(key)
            except KeyError:
                return value

        def fn(x, d, key):
            # [FUNCTION DOC] HELPER: returns sin of x plus one together with mapping_get result.
            x = torch.sin(x + 1)
            return x, mapping_get(d, key)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.rand(2, 3)
        ref = fn(x, d, "m")
        res = opt_fn(x, d, "m")
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    @make_dynamo_test
    def test_raise_from_None_2(self):
        # [FUNCTION DOC] Runs under the dynamo harness: raising TypeError from None inside a
        # handler yields an exception with no cause and suppressed context.
        def fn():
            # [FUNCTION DOC] HELPER: raises ValueError, catches it, raises TypeError from None.
            try:
                raise ValueError
            except Exception:
                raise TypeError from None

        try:
            fn()
        except TypeError as e:
            assert e.__cause__ is None  # noqa: S101
            assert e.__suppress_context__ is True  # noqa: S101

    @make_dynamo_test
    def test_raise_from_other(self):
        # [FUNCTION DOC] Runs under the dynamo harness: raising TypeError from a caught
        # ValueError sets that ValueError as cause and marks the context as suppressed.
        def fn():
            # [FUNCTION DOC] HELPER: raises ValueError, catches it as e, raises TypeError from e.
            try:
                raise ValueError
            except Exception as e:
                raise TypeError from e

        try:
            fn()
        except TypeError as e:
            assert isinstance(e.__cause__, ValueError)  # noqa: S101
            assert e.__suppress_context__ is True  # noqa: S101

    @make_dynamo_test
    def test_reraise_first_exc(self):
        # [FUNCTION DOC] Runs under the dynamo harness: a bare re-raise after a nested caught
        # exception re-raises the original ZeroDivisionError, and afterwards no exception is
        # active.
        def fn():
            # [FUNCTION DOC] HELPER: handles ZeroDivisionError, runs a nested raise/catch of
            # ValueError, then re-raises the original.
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                try:
                    raise ValueError
                except ValueError:
                    pass
                raise

        try:
            fn()
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_ensure_exception_is_active_after_try_except_block(self):
        # [FUNCTION DOC] Runs under the dynamo harness: after nested try/except blocks complete
        # inside a handler, a bare re-raise still raises the outer ZeroDivisionError, and no
        # exception remains active at the end.
        try:
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                for exc in (KeyError, IndexError):
                    try:
                        raise exc
                    except exc:
                        pass
                raise
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_ensure_exception_is_active_inside_try_except_block(self):
        # [FUNCTION DOC] Runs under the dynamo harness: exceptions raised in a loop inside a
        # handler have the outer ZeroDivisionError as their context; then the outer one is
        # re-raised and no exception remains active.
        try:
            try:
                raise ZeroDivisionError
            except ZeroDivisionError:
                for exc in (KeyError, IndexError):
                    try:
                        raise exc
                    except exc as e:
                        assert isinstance(e.__context__, ZeroDivisionError)  # noqa: S101
                raise
        except ZeroDivisionError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_handle_all_exceptions(self):
        # [FUNCTION DOC] Runs under the dynamo harness: throwing ValueError into a generator
        # that raises TypeError from its handler surfaces TypeError, leaving no active exception.
        def cm():
            # [FUNCTION DOC] HELPER: generator yielding one; on ValueError thrown in, raises
            # TypeError (through a no-op finally).
            try:
                yield 1
            except ValueError:
                try:
                    raise TypeError
                finally:
                    pass

        try:
            gen = cm()
            next(gen)
            gen.throw(ValueError)
        except TypeError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_reraise(self):
        # [FUNCTION DOC] Runs under the dynamo harness: a bare re-raise propagates ValueError
        # to the outer handler and no exception stays active afterwards.
        try:
            try:
                raise ValueError
            except ValueError:  # noqa: TRY203
                raise
        except ValueError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    @make_dynamo_test
    def test_raise_finally_simple(self):
        # [FUNCTION DOC] Runs under the dynamo harness: raising TypeError inside a handler with
        # a no-op finally propagates TypeError; no exception remains active.
        def fn():
            # [FUNCTION DOC] HELPER: raises ValueError, and in its handler raises TypeError.
            try:
                raise ValueError
            except ValueError:
                try:
                    raise TypeError
                finally:
                    pass

        try:
            fn()
        except TypeError:
            pass
        assert sys.exc_info()[0] is None  # noqa: S101

    def test_reconstruct___context__(self):
        # [FUNCTION DOC] An exception built in compiled code with context and cause assigned is
        # returned; checks types of the reconstructed context and cause and that the
        # context-suppressed flag is true.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: builds a ValueError with a TypeError context and
            # RuntimeError cause; returns sin of t and the exception.
            v = ValueError(1, 2, 3)
            v.__context__ = TypeError()
            v.__cause__ = RuntimeError()
            return t.sin(), v

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, ValueError)
        self.assertIsInstance(v.__context__, TypeError)
        self.assertIsInstance(v.__cause__, RuntimeError)
        self.assertTrue(v.__suppress_context__)

    def test_reconstruct_exception_2(self):
        # [FUNCTION DOC] An exception raised from None inside a handler, then given a new
        # cause, is returned from compiled code; checks it is a TypeError with a ValueError
        # context and RuntimeError cause.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: nested raise/catch producing a TypeError, sets its cause
            # to a RuntimeError and returns sin of t with it.
            try:
                raise ValueError(1, 2, 3)
            except Exception:
                try:
                    raise TypeError(4, 5) from None
                except Exception as e:
                    e.__cause__ = RuntimeError(6, 7)
                    return t.sin(), e

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, TypeError)
        self.assertIsInstance(v.__context__, ValueError)
        self.assertIsInstance(v.__cause__, RuntimeError)

    def test_reconstruct_AttributeError(self):
        # [FUNCTION DOC] An AttributeError built with name and obj keywords in compiled code is
        # returned with args, name and obj preserved (obj by identity).
        sentinel = object()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: returns sin of t and an AttributeError with name and obj.
            return t.sin(), AttributeError("boom", name="myattr", obj=sentinel)

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, AttributeError)
        self.assertEqual(v.args, ("boom",))
        self.assertEqual(v.name, "myattr")
        self.assertIs(v.obj, sentinel)

    def test_reconstruct_AttributeError_from_getattr(self):
        # [FUNCTION DOC] An AttributeError raised by reading a missing attribute in compiled
        # code and returned to the caller must carry the attribute name.
        class Foo:
            # [CLASS DOC] HELPER: class with a single class attribute bar.
            bar = 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: reads a missing attribute of a Foo and returns sin of t
            # with the caught error.
            obj = Foo()
            try:
                obj.missing
            except AttributeError as e:
                return t.sin(), e

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, AttributeError)
        self.assertEqual(v.name, "missing")

    def test_reconstruct_NameError(self):
        # [FUNCTION DOC] A NameError built with a name keyword in compiled code is returned with
        # args and name preserved.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: returns sin of t and a NameError carrying a name.
            return t.sin(), NameError("boom", name="myvar")

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, NameError)
        self.assertEqual(v.args, ("boom",))
        self.assertEqual(v.name, "myvar")

    def test_reconstruct_NameError_from_undefined(self):
        # [FUNCTION DOC] A NameError raised by using an undefined global in compiled code and
        # returned to the caller must carry the missing name.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: references an undefined name, catching NameError, and
            # returns sin of t with the error.
            try:
                undefined_name
            except NameError as e:
                return t.sin(), e

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, NameError)
        self.assertEqual(v.name, "undefined_name")

    def test_reconstruct_StopIteration(self):
        # [FUNCTION DOC] A StopIteration built with a value in compiled code is returned with
        # args and value preserved.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: returns sin of t and a StopIteration carrying 42.
            return t.sin(), StopIteration(42)

        t = torch.randn(2)
        y, v = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsInstance(v, StopIteration)
        self.assertEqual(v.args, (42,))
        self.assertEqual(v.value, 42)

    def test_raise_GeneratorExit(self):
        # [FUNCTION DOC] Raising GeneratorExit in compiled code must skip the Exception handler
        # and be caught by the BaseException handler, giving cos.
        # GeneratorExit does not inherit from Exception
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises GeneratorExit; Exception handler returns sin,
            # BaseException handler returns cos.
            try:
                raise GeneratorExit
            except Exception:
                return t.sin()
            except BaseException:
                return t.cos()

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.cos())

    def test_speculation_exception(self):
        # [FUNCTION DOC] Records one entry in a SpeculationLog, restarts it, and checks that
        # replaying a different entry raises SpeculationLogDivergence.
        log = SpeculationLog()
        log.next("fake", 555, "fake", Instruction(1, "fake", 1, 1))
        log.restart()
        with self.assertRaises(SpeculationLogDivergence):
            log.next("bad", 58, "bad", Instruction(2, "different", 2, 2))

    def test_dict_pop(self):
        # [FUNCTION DOC] Popping a missing key from a dict inside try/except KeyError (with an
        # else branch) works in compiled fullgraph code for dicts with and without the key.
        # Pattern from inspect.bind
        def fn(dt, x):
            # [FUNCTION DOC] HELPER: pops key b; returns sin x if missing, cos x if present.
            try:
                dt.pop("b")
            except KeyError:
                return torch.sin(x)
            else:
                return torch.cos(x)

        d = {"a": 1}
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)
        self.assertEqual(fn(d, x), opt_fn(d, x))
        self.assertEqual(fn({"a": 1, "b": 2}, x), opt_fn({"a": 1, "b": 2}, x))

    def test_block_stack_cleanup(self):
        # [FUNCTION DOC] A loop with try/except KeyError inside, with some iterations raising
        # and some not, must compile fullgraph and match eager (checks block stack cleanup).
        params = {
            "a": 3,
            "b": 4,
            "c": 5,
        }

        dt = {
            "c": 5,
        }

        def fn(x):
            # [FUNCTION DOC] HELPER: for each name in params multiplies x by dt's entry, or by
            # sin x if missing.
            for name in params:
                try:
                    x = x * dt[name]
                except KeyError:
                    x = x * torch.sin(x)
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_set_cause_with_arg(self):
        # [FUNCTION DOC] Setting the cause of an exception passed into compiled code mutates the
        # caller's exception object: afterwards its cause is a ValueError.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t, err):
            # [FUNCTION DOC] HELPER: sets err cause to a new ValueError, returns sin of t.
            err.__cause__ = ValueError()
            return t.sin()

        t = torch.randn(2)
        e = TypeError("abcd")
        fn(t, e)
        self.assertIsInstance(e.__cause__, ValueError)

    def test_set_cause_with_arg_error(self):
        # [FUNCTION DOC] Assigning a non-exception (an int) as the cause of an exception in
        # compiled code must raise TypeError saying the cause must be an exception or None.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t, err):
            # [FUNCTION DOC] HELPER: sets err cause to the int two, returns sin of t.
            err.__cause__ = 2
            return t.sin()

        t = torch.randn(2)
        e = TypeError("abcd")
        with self.assertRaisesRegex(TypeError, "exception cause must be"):
            fn(t, e)

    @parametrize(
        "ex",
        [TypeError, CustomException],
        name_fn=lambda x: x.__name__,
    )
    @make_dynamo_test
    def test_set___cause__(self, ex):
        # [FUNCTION DOC] Parametrized over exception class ex (TypeError, CustomException) and
        # run under the dynamo harness: raising TypeError from None inside a handler for ex
        # leaves ex as context, no cause, and suppressed context.
        def fn():
            # [FUNCTION DOC] HELPER: raises ex, catches it, raises TypeError from None.
            try:
                raise ex
            except ex:
                raise TypeError from None

        try:
            fn()
        except TypeError as e:
            assert isinstance(e.__context__, ex)  # noqa: S101
            assert e.__cause__ is None  # noqa: S101
            assert e.__suppress_context__ is True  # noqa: S101

    @parametrize(
        "ex",
        [RuntimeError, CustomException],
        name_fn=lambda x: x.__name__,
    )
    @make_dynamo_test
    def test_set___cause___error(self, ex):
        # [FUNCTION DOC] Parametrized over exception class ex (RuntimeError, CustomException)
        # and run under the dynamo harness: assigning a non-exception cause then re-raising
        # raises TypeError whose single arg is the "exception cause must be None or derive from
        # BaseException" message; any other exception fails the test.
        def fn():
            # [FUNCTION DOC] HELPER: raises ex, sets the cause to the int two, re-raises.
            try:
                raise ex
            except Exception as e:
                e.__cause__ = 2
                raise

        z = 0

        try:
            fn()
        except TypeError as e:
            z = 1
            assert e.args == (  # noqa: S101
                "exception cause must be None or derive from BaseException",
            )
        except Exception:
            raise AssertionError from None

        assert z == 1  # noqa: S101

    def test_user_defined_exception_variable(self):
        # [FUNCTION DOC] Catching a user-defined exception class and taking the type of the
        # caught instance works in compiled fullgraph code; result equals eager.
        def fn(t):
            # [FUNCTION DOC] HELPER: raises CustomException; its handler checks the type of
            # the instance's class is plain type, adds one to t if so, and asserts the right
            # handler ran; returns sin t.
            z = 0
            try:
                raise CustomException
            except ValueError:
                z = 1
            except CustomException as e:
                # trying to call python_type on the
                # UserDefinedExceptionClassVariable
                cls = type(e)
                if type(cls) is type:
                    t = t + 1
                z = 2
            assert z == 2  # noqa: S101
            return t.sin()

        t = torch.randn(2)
        fn(t)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(t), opt_fn(t))

    def test_user_defined_exception_with_args(self):
        # [FUNCTION DOC] Raising a user-defined exception whose init takes a positional and a
        # keyword argument works in compiled fullgraph code and is matched by the right handler.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises CustomExceptionWithArgs and asserts the matching
            # handler ran; returns None.
            z = 0
            try:
                raise CustomExceptionWithArgs(2, b=3)
            except ValueError:
                z = 1
            except CustomExceptionWithArgs:
                z = 2
            assert z == 2  # noqa: S101

        t = torch.randn(2)
        fn(t)

    @make_dynamo_test
    def test_raise_set___context__(self):
        # [FUNCTION DOC] Runs under the dynamo harness: exceptions raised and caught in separate
        # top-level try blocks have no context set.
        try:
            raise TypeError
        except TypeError as e:
            exc = e

        assert exc.__context__ is None  # noqa: S101

        try:
            raise ValueError
        except ValueError as e:
            exc2 = e

        assert exc2.__context__ is None  # noqa: S101

    def test_exception_kwargs(self):
        # [FUNCTION DOC] Raising an AttributeError constructed with a keyword argument under
        # fullgraph compile is not supported: calling raises Unsupported.
        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: raises AttributeError with the name keyword.
            raise AttributeError(name="a")

        self.assertRaises(Unsupported, fn)

    def test_stack_trace_from_observed_exception(self):
        # [FUNCTION DOC] Exporting a module whose forward reads a missing submodule attribute
        # via the functional export graph capture must raise an exception whose message includes
        # the offending source line.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: module with one 16 by 16 linear layer whose forward reads a
            # nonexistent attribute of that layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(16, 16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: reads the missing weight attribute (raises) then
                # would apply linear.
                # no attribute w on self.linear
                weight = self.linear.w
                return torch.nn.functional.linear(x, weight)

        x = (torch.randn(4, 16, requires_grad=True),)

        with self.assertRaisesRegex(Exception, "weight = self.linear.w"):
            torch._dynamo.functional_export.dynamo_graph_capture_for_export(Model())(x)

    def test_context_manager_preserves_exception_stack(self):
        # [FUNCTION DOC] Fullgraph compile of a function that calls an asserting helper inside
        # no_grad must raise Unsupported whose text points at the failing assert in the helper,
        # not at the later return statement.
        # Regression test for https://github.com/pytorch/pytorch/issues/167900
        # When an exception is raised inside a context manager and the context manager
        # doesn't suppress it, the error message should point to the original raise
        # location, not the context manager cleanup code.
        def g():
            # [FUNCTION DOC] HELPER: always fails an assertion.
            assert False  # noqa: B011, S101

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: calls g inside no_grad, then returns x.
            with torch.no_grad():
                g()
            return x

        with self.assertRaises(Unsupported) as ctx:
            f(torch.randn(1))

        # The error should point to "assert False" in g(), not "return x"
        self.assertIn("in g", str(ctx.exception))
        self.assertIn("assert False", str(ctx.exception))
        self.assertNotIn("return x", str(ctx.exception))

    def test_reraise_preserves_exception_stack(self):
        # [FUNCTION DOC] Fullgraph compile of a function that catches and bare re-raises an
        # exception from a helper must raise Unsupported whose text points at the original raise
        # in the helper rather than the re-raise.
        # Regression test for https://github.com/pytorch/pytorch/issues/167900
        # When an exception is caught and re-raised, the error message should
        # point to the original raise location, not the reraise location.
        def g():
            # [FUNCTION DOC] HELPER: always raises a generic Exception with message Invalid.
            raise Exception("Invalid")  # noqa: TRY002

        @torch.compile(backend="eager", fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: calls g in try, re-raises any Exception, else returns x.
            try:
                g()
            except Exception:  # noqa: TRY203
                raise
            return x

        with self.assertRaises(Unsupported) as ctx:
            f(torch.randn(1))

        # The error should point to 'raise Exception("Invalid")' in g()
        self.assertIn("in g", str(ctx.exception))
        self.assertIn('raise Exception("Invalid")', str(ctx.exception))

    def test_str_repr_exception_no_args(self):
        # [FUNCTION DOC] str and repr of an arg-less ValueError in compiled code give the empty
        # string and the bare class call text.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises/catches ValueError; returns sin t, str and repr.
            try:
                raise ValueError
            except ValueError as e:
                return t.sin(), str(e), repr(e)

        t = torch.randn(2)
        y, s, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, "")
        self.assertEqual(r, "ValueError()")

    def test_str_repr_exception_single_arg(self):
        # [FUNCTION DOC] str of a one-arg ValueError is the argument text and repr shows the
        # quoted argument, in compiled code.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises/catches ValueError with a message; returns sin t,
            # str and repr.
            try:
                raise ValueError("test error")
            except ValueError as e:
                return t.sin(), str(e), repr(e)

        t = torch.randn(2)
        y, s, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, "test error")
        self.assertEqual(r, "ValueError('test error')")

    def test_str_repr_exception_multi_args(self):
        # [FUNCTION DOC] str of a two-arg ValueError is the text of the args tuple and repr
        # lists both args, in compiled code.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: raises/catches ValueError with a string and an int;
            # returns sin t, str and repr.
            try:
                raise ValueError("hello", 42)
            except ValueError as e:
                return t.sin(), str(e), repr(e)

        t = torch.randn(2)
        y, s, r = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(s, str(("hello", 42)))
        self.assertEqual(r, "ValueError('hello', 42)")

    def test_frozen_dataclass_setattr_raises(self):
        # [FUNCTION DOC] Assigning a field of a frozen dataclass inside compiled code raises
        # FrozenInstanceError, caught by its specific handler: result is t plus one.
        @dataclasses.dataclass(frozen=True)
        class TestDataClass:
            # [CLASS DOC] HELPER: frozen dataclass with one int field x.
            x: int

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: tries to mutate a frozen instance; returns t plus 1 on
            # FrozenInstanceError, t plus 2 on other errors, else t plus x.
            dc = TestDataClass(1)
            try:
                dc.x = 2
            except dataclasses.FrozenInstanceError:
                return t + 1
            except Exception:
                return t + 2
            return t + dc.x

        self.assertEqual(fn(torch.zeros(1)), 1)

    def test_exception_traceback_access(self):
        # [FUNCTION DOC] Reading the traceback attribute of a caught exception works in
        # compiled fullgraph code; result equals eager.
        # Test that __traceback__ is accessible after raising/catching an exception
        def fn(x):
            # [FUNCTION DOC] HELPER: adds one to x if the caught exception has a traceback.
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_tb_next(self):
        # [FUNCTION DOC] Reading tb_next of a caught exception's traceback works in compiled
        # fullgraph code; result equals eager.
        # Test that tb_next can be accessed on a traceback
        def fn(x):
            # [FUNCTION DOC] HELPER: adds one to x when the traceback's next entry is None.
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    # tb_next is None for a single-frame traceback
                    if tb.tb_next is None:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_tb_lineno(self):
        # [FUNCTION DOC] Reading tb_lineno of a caught exception's traceback works in compiled
        # fullgraph code; result equals eager.
        # Test that tb_lineno is accessible on a traceback
        def fn(x):
            # [FUNCTION DOC] HELPER: adds one to x when the traceback line number is positive.
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None and tb.tb_lineno > 0:
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_with_traceback_method(self):
        # [FUNCTION DOC] Using the with_traceback method to attach a traceback to a new
        # exception works in compiled fullgraph code; result equals eager.
        # Test the with_traceback() method
        def fn(x):
            # [FUNCTION DOC] HELPER: attaches the first exception's traceback to a second one
            # and adds one to x if it then has a traceback.
            try:
                raise ValueError("first")
            except ValueError as e:
                tb = e.__traceback__
                try:
                    raise RuntimeError("second").with_traceback(tb) from None
                except RuntimeError as e2:
                    if e2.__traceback__ is not None:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_set_traceback(self):
        # [FUNCTION DOC] Assigning the traceback attribute of an exception works in compiled
        # fullgraph code; result equals eager.
        # Test assigning __traceback__ on an exception
        def fn(x):
            # [FUNCTION DOC] HELPER: copies one exception's traceback to another; adds one to x.
            try:
                raise ValueError("first")
            except ValueError as e:
                tb = e.__traceback__
                try:
                    raise RuntimeError("second") from None
                except RuntimeError as e2:
                    e2.__traceback__ = tb
                    if e2.__traceback__ is not None:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_set_traceback_none(self):
        # [FUNCTION DOC] Clearing the traceback attribute of a caught exception works in compiled
        # fullgraph code; result equals eager.
        # Test assigning None to __traceback__
        def fn(x):
            # [FUNCTION DOC] HELPER: sets the traceback to None; adds one to x if it reads None.
            try:
                raise ValueError("oops")
            except ValueError as e:
                e.__traceback__ = None
                if e.__traceback__ is None:
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_tb_lasti_graph_break(self):
        # [FUNCTION DOC] Reading tb_lasti is unsupported in tracing so it graph breaks; compiled
        # (non-fullgraph) result must still equal eager.
        # Accessing tb_lasti should cause a graph break
        def fn(x):
            # [FUNCTION DOC] HELPER: reads the traceback's last instruction index then adds
            # one to x.
            try:
                raise ValueError("oops")
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    _ = tb.tb_lasti
                    x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        # Should graph break but still produce correct results
        opt_fn = torch.compile(fn, backend="eager")
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_set_tb_next(self):
        # [FUNCTION DOC] Assigning tb_next of a traceback works in compiled fullgraph code;
        # result equals eager.
        # Test setting tb_next on a traceback
        def fn(x):
            # [FUNCTION DOC] HELPER: collects two exceptions' tracebacks, clears the second's
            # next link and adds one to x.
            try:
                raise ValueError("first")
            except ValueError as e:
                tb1 = e.__traceback__
                try:
                    raise RuntimeError("second") from None
                except RuntimeError as e2:
                    tb2 = e2.__traceback__
                    if tb2 is not None and tb1 is not None:
                        tb2.tb_next = None
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_exception_traceback_chained(self):
        # [FUNCTION DOC] Walking the next links of a traceback of an exception raised in a callee
        # works in compiled fullgraph code; result equals eager.
        # Test traceback chaining through multiple frames
        def inner():
            # [FUNCTION DOC] HELPER: always raises ValueError.
            raise ValueError("inner")

        def fn(x):
            # [FUNCTION DOC] HELPER: catches inner's error, adds one to x, walks the traceback
            # chain counting depth and adds one more if the depth is positive.
            try:
                inner()
            except ValueError as e:
                tb = e.__traceback__
                if tb is not None:
                    x = x + 1
                    # Walk the traceback chain
                    depth = 0
                    curr = tb
                    while curr is not None:
                        depth += 1
                        curr = curr.tb_next
                    if depth > 0:
                        x = x + 1
            return x

        x = torch.randn(4)
        ref = fn(x)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    @parametrize(
        "exc_type_1,exc_type_2",
        [
            (ValueError, TypeError),
            (CustomException, ValueError),
        ],
        name_fn=lambda exc1, exc2: f"{exc1.__name__}_to_{exc2.__name__}",
    )
    def test_exception_set_context(self, exc_type_1, exc_type_2):
        # [FUNCTION DOC] Parametrized over two exception classes: assigning one instance as the
        # context of another in compiled fullgraph code is visible on read; expects sin.
        # Test explicitly assigning to __context__ attribute (reaches ExceptionVariable.__context__ assignment)
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: builds two exceptions, links context, returns sin t if the
            # link reads back identical else cos t.
            exc1 = exc_type_1("first")
            exc2 = exc_type_2("second")

            # This explicitly sets __context__ via call_setattr
            exc2.__context__ = exc1

            # Verify it was set correctly
            if exc2.__context__ is exc1:
                return t.sin()
            else:
                return t.cos()

        t = torch.randn(2)
        ref_result = t.sin()
        result = fn(t)
        self.assertEqual(result, ref_result)

    @parametrize(
        "exc_type_1,exc_type_2,exc_type_3",
        [
            (ValueError, TypeError, RuntimeError),
            (CustomException, ValueError, TypeError),
        ],
        name_fn=lambda exc1, exc2, exc3: (
            f"{exc1.__name__}_chain_{exc2.__name__}_{exc3.__name__}"
        ),
    )
    def test_exception_context_chain(self, exc_type_1, exc_type_2, exc_type_3):
        # [FUNCTION DOC] Parametrized over three exception classes: a context chain of three
        # exceptions built by assignment in compiled fullgraph code reads back with the right
        # types; expects sin.
        # Test chaining contexts through multiple exceptions
        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: links three exceptions by context and returns sin t if the
            # chain has the expected types, else cos t.
            exc1 = exc_type_1("first")
            exc2 = exc_type_2("second")
            exc3 = exc_type_3("third")

            exc2.__context__ = exc1
            exc3.__context__ = exc2

            # Verify the chain
            if isinstance(exc3.__context__, exc_type_2) and isinstance(
                exc3.__context__.__context__, exc_type_1
            ):
                return t.sin()
            else:
                return t.cos()

        t = torch.randn(2)
        ref_result = t.sin()
        result = fn(t)
        self.assertEqual(result, ref_result)

    @make_dynamo_test
    def test_exception_custom_attribute(self):
        # [FUNCTION DOC] Runs under the dynamo harness: a new attribute set on an exception
        # instance reads back.
        e = RuntimeError("boom")
        e.foo = 42
        assert e.foo == 42  # noqa: S101

    @make_dynamo_test
    def test_exception_set_args_from_iterable(self):
        # [FUNCTION DOC] Runs under the dynamo harness: assigning a list to the args attribute
        # of an exception stores it as a tuple.
        e = RuntimeError("boom")
        e.args = [1, 2, 3]
        assert e.args == (1, 2, 3)  # noqa: S101

    @make_dynamo_test
    def test_exception_set_args_not_iterable(self):
        # [FUNCTION DOC] Runs under the dynamo harness: assigning a non-iterable to args raises
        # TypeError mentioning the object is not iterable; fails if no error occurs.
        e = RuntimeError("boom")
        try:
            e.args = 2
        except TypeError as exc:
            assert "object is not iterable" in str(exc)  # noqa: S101
        else:
            raise AssertionError

    @make_dynamo_test
    def test_exception_setstate_dict(self):
        # [FUNCTION DOC] Runs under the dynamo harness: restoring state from a mapping sets
        # that entry as an attribute of the exception.
        e = RuntimeError("boom")
        e.__setstate__({"foo": 7})
        assert e.foo == 7  # noqa: S101

    @make_dynamo_test
    def test_exception_setstate_none_noop(self):
        # [FUNCTION DOC] Runs under the dynamo harness: restoring state from None does nothing
        # and returns None.
        e = RuntimeError("boom")
        assert e.__setstate__(None) is None  # noqa: S101

    @make_dynamo_test
    def test_exception_setstate_not_dict(self):
        # [FUNCTION DOC] Runs under the dynamo harness: restoring state from a non-mapping
        # raises TypeError saying the state is not a dictionary; fails if no error occurs.
        e = RuntimeError("boom")
        try:
            e.__setstate__(2)
        except TypeError as exc:
            assert "state is not a dictionary" in str(exc)  # noqa: S101
        else:
            raise AssertionError

    def test_exception_custom_attribute_side_effect_replayed(self):
        # [FUNCTION DOC] An exception with a custom attribute and restored state is returned
        # from fullgraph compiled code; checks the returned real object has both attributes
        # replayed (42 and 7) and the tensor output is x plus one.
        # The exception escapes the compiled region, so the custom attributes
        # set during tracing must be replayed onto the real object handed back
        # to eager (exercises the side-effects codegen, not just tracing).
        def fn(x):
            # [FUNCTION DOC] HELPER: builds a RuntimeError, sets attribute foo and restores state
            # with bar; returns it with x plus one.
            e = RuntimeError("boom")
            e.foo = 42
            e.__setstate__({"bar": 7})
            return e, x + 1

        x = torch.randn(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        e, y = opt_fn(x)
        self.assertIsInstance(e, RuntimeError)
        self.assertEqual(e.foo, 42)
        self.assertEqual(e.bar, 7)
        self.assertEqual(y, x + 1)

    def test_exception_setstate_non_string_key_diverges_from_eager(self):
        # [FUNCTION DOC] Documents a divergence: restoring exception state with an integer key
        # raises TypeError in eager, but the fullgraph compiled run does not raise.
        # KNOWN ISSUE: the compiled path silently accepts a non-string attribute name.
        # Eager's BaseException.__setattr__ requires string names, so
        # __setstate__ with a non-string key raises TypeError. Dynamo currently
        # stores non-string constant keys in the side-effect dict without error
        # -- a known divergence from eager, to be fixed alongside tp_setattro.
        def fn(x):
            # [FUNCTION DOC] HELPER: restores state with an int key on a RuntimeError,
            # returns x plus one.
            e = RuntimeError("boom")
            e.__setstate__({1: 2})
            return x + 1

        x = torch.randn(4)
        with self.assertRaisesRegex(TypeError, "attribute name must be string"):
            fn(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        opt_fn(x)  # diverges: Dynamo does not raise


instantiate_parametrized_tests(ExceptionTests)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
