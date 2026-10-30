# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for the TorchScript Await type (torch.jit._awaitable, _awaitable_wait and
# _awaitable_nowait). Covers eager Python behavior, scripting with class arguments and returns,
# nested awaits, implicit wait on attribute access, tracing, and save/load round trips.

import io
from typing import List, Optional, Tuple

import torch
from torch import Tensor
from torch._awaits import _Await as Await
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


class TestAwait(JitTestCase):
    # [CLASS DOC] Test case holding the Await tests; each scripted test compares scripted output
    # with eager output and a hand-computed expected tensor.
    def test_await_python(self):
        # [FUNCTION DOC] In eager mode an awaitable wraps a function and arguments; checks that
        # calling the stored function on the stored args equals the wait result, and that a nowait
        # awaitable reports is_nowait and keeps its argument tuple.
        def foo(x: int) -> int:
            # [FUNCTION DOC] HELPER: Adds thirteen to an integer.
            return x + 13

        aw: Await[int] = torch.jit._awaitable(foo, 13)
        self.assertTrue(aw.fn()(*aw.args()) == torch.jit._awaitable_wait(aw))
        nw = torch.jit._awaitable_nowait(33)
        self.assertTrue(nw.is_nowait())
        self.assertTrue(nw.args() == (33,))

    def test_await_type_python(self):
        # [FUNCTION DOC] Only checks that an awaitable can be created and appended to an annotated
        # list of Await of Tensor; never calls the function (which would fail).
        def foo() -> Tensor:
            # [FUNCTION DOC] HELPER: Body that would call randn without a size; never executed by
            # the test.
            return torch.randn()

        awaits = torch.jit.annotate(List[Await[Tensor]], [])
        awaits.append(torch.jit._awaitable(foo))

    def test_script(self):
        # [FUNCTION DOC] Scripts a function that awaits a delayed integer addition while computing
        # an identity matrix, and checks the result equals eye plus 102 and matches eager.
        def delayed(z: int) -> int:
            # [FUNCTION DOC] HELPER: Adds three to an integer.
            return z + 3

        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Creates an awaitable of delayed on 99, builds eye, waits,
            # and sums.
            aw: Await[int] = torch.jit._awaitable(delayed, 99)
            a = torch.eye(2)
            b = torch.jit._awaitable_wait(aw)
            return a + b + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 102, script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_nowait(self):
        # [FUNCTION DOC] Scripts a function using a nowait awaitable of 13; checks result is eye
        # plus 13 and equal to eager.
        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Wraps 13 with nowait, waits on it, adds to eye and the input.
            aw = torch.jit._awaitable_nowait(13)
            a = torch.eye(2)
            b = torch.jit._awaitable_wait(aw)
            return a + b + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 13, script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_nowait_class(self):
        # [FUNCTION DOC] Scripts a function that wraps a user class instance in a nowait awaitable
        # and reads a method after waiting; expects eye as the result since the class holds zeros.
        class C:
            # [CLASS DOC] HELPER: Plain class storing two tensors with an accessor for the first.
            def __init__(self, a: Tensor, b: Tensor):
                # [FUNCTION DOC] HELPER: Stores both tensors in private-style attributes.
                self._a = a
                self._b = b

            def a(self) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the first stored tensor.
                return self._a

        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Nowait-wraps a class instance, waits, and combines with eye
            # and x.
            aw = torch.jit._awaitable_nowait(C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c = torch.jit._awaitable_wait(aw)
            return _a + c.a() + x

        make_global(C)
        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_class_arg(self):
        # [FUNCTION DOC] Scripts a function passing a class instance as the argument to an
        # awaitable function; result should be eye (instance holds zeros) and match eager.
        class C:
            # [CLASS DOC] HELPER: Plain class storing two tensors in name-mangled attributes.
            def __init__(self, a: Tensor, b: Tensor):
                # [FUNCTION DOC] HELPER: Stores both tensors.
                self.__a = a
                self.__b = b

            def a(self) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the first stored tensor.
                return self.__a

        make_global(C)

        def delayed(c: C) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns the first tensor of the given class instance.
            return c.a()

        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Builds an instance, awaits delayed on it, waits, sums with
            # eye and x.
            c = C(torch.zeros(2), torch.ones(2))
            aw = torch.jit._awaitable(delayed, c)
            _a = torch.eye(2)
            c2_t = torch.jit._awaitable_wait(aw)
            return _a + c2_t + x

        inp = torch.zeros(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_awaitable_to_await(self):
        # [FUNCTION DOC] Uses a method-like function defined outside the class as the awaited
        # function on a class with slots; result is eye plus twice ones for input ones.
        class C:
            # [CLASS DOC] HELPER: Slotted class storing two tensors.
            __slots__ = ["_a", "_b"]

            def __init__(self, a: Tensor, b: Tensor):
                # [FUNCTION DOC] HELPER: Stores both tensors.
                self._a = a
                self._b = b

        make_global(C)

        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C):
            # [FUNCTION DOC] HELPER: Defined outside the class because scripting cannot handle a
            # recursive annotation of self; returns the sum of the two stored tensors.
            return self._a + self._b

        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Awaits the wait-impl function on an instance, waits, sums
            # with eye and x.
            aw = torch.jit._awaitable(C_wait_impl, C(torch.zeros(2), torch.ones(2)))
            _a = torch.eye(2)
            c_wait_impl_res = torch.jit._awaitable_wait(aw)
            return _a + c_wait_impl_res + x

        inp = torch.ones(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 2 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_class_return(self):
        # [FUNCTION DOC] Awaitable whose function returns a class instance, consumed by another
        # scripted function; checks value (eye plus six times ones) and exactly one
        # awaitable_wait node.
        class C:
            # [CLASS DOC] HELPER: Slotted class with two public tensor attributes.
            __slots__ = ["a", "b"]

            def __init__(self, a: Tensor, b: Tensor):
                # [FUNCTION DOC] HELPER: Stores both tensors.
                self.a = a
                self.b = b

        make_global(C)

        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C) -> C:
            # [FUNCTION DOC] HELPER: Returns a new instance with first scaled by two and second
            # by three.
            return C(self.a * 2, self.b * 3)

        def fn_arg_C(x: C) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns the sum of the two tensors of the instance.
            return x.a + x.b

        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Awaits the wait-impl, passes the waited result to the
            # consumer, adds eye and x.
            aw: Await[C] = torch.jit._awaitable(C_wait_impl, C(x, x))
            _a = torch.eye(2)
            y = fn_arg_C(torch.jit._awaitable_wait(aw))
            return _a + y + x

        inp = torch.ones(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 6 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))
        self.assertGraphContainsExactly(
            sm.graph, kind="prim::awaitable_wait", num_kind_nodes=1
        )

    def test_await_getattr_implicit_convertion(self):
        # [FUNCTION DOC] Attribute and method access directly on an awaitable of a class instance
        # implicitly waits; checks result (eye plus seven times ones) and two
        # awaitable_wait nodes.
        class C:
            # [CLASS DOC] HELPER: Class with two tensor attributes and an accessor for the second.
            def __init__(self, a: Tensor, b: Tensor):
                # [FUNCTION DOC] HELPER: Stores both tensors.
                self._a = a
                self._b = b

            def b(self):
                # [FUNCTION DOC] HELPER: Returns the second stored tensor.
                return self._b

        make_global(C)

        # Can not stay in the class as Jit does not support Recursive annotations
        # (self in wait_impl can not be annotated as C as C is not defined by this time)
        def C_wait_impl(self: C) -> C:
            # [FUNCTION DOC] HELPER: Returns a new instance with first scaled by two and second
            # by three.
            return C(self._a * 2, self._b * 3)

        def fn_arg_C(x: C) -> Tensor:
            # [FUNCTION DOC] HELPER: Unused consumer returning the sum of the instance tensors.
            return x._a + x._b

        def fn(x: Tensor):
            # [FUNCTION DOC] HELPER: Reads an attribute and a method result straight from an
            # awaitable, plus a fresh instance's values, and sums them with eye and x.
            aw: Await[C] = torch.jit._awaitable(C_wait_impl, C(x, x))
            _a = torch.eye(2)
            ai = aw._a
            awb = aw.b()  # noqa: F841
            c = C(2 * x, 2 * x)
            return _a + ai + x + c._a + c.b()

        inp = torch.ones(2)

        sm = torch.jit.script(fn)
        out = fn(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(torch.eye(2) + 7 * torch.ones(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))
        self.assertGraphContainsExactly(
            sm.graph, kind="prim::awaitable_wait", num_kind_nodes=2
        )

    def test_await_nested(self):
        # [FUNCTION DOC] Awaitable returning another awaitable; main waits twice and the result is
        # six times eye, equal to eager.
        class C:
            # [CLASS DOC] HELPER: Class storing two tensors in name-mangled attributes.
            def __init__(self, a: Tensor, b: Tensor):
                # [FUNCTION DOC] HELPER: Stores both tensors.
                self.__a = a
                self.__b = b

            def a(self) -> Tensor:
                # [FUNCTION DOC] HELPER: Returns the first stored tensor.
                return self.__a

        make_global(C)

        def delayed(c: C) -> Await[Tensor]:
            # [FUNCTION DOC] HELPER: Returns a nowait awaitable of three times the first tensor.
            return torch.jit._awaitable_nowait(3 * c.a())

        def fn(x: Tensor) -> Await[Await[Tensor]]:
            # [FUNCTION DOC] HELPER: Returns an awaitable of an awaitable built from delayed.
            return torch.jit._awaitable(delayed, C(2 * x, x))

        def main(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Waits twice on the nested awaitable and returns the tensor.
            awaw = fn(x)
            return torch.jit._awaitable_wait(torch.jit._awaitable_wait(awaw))

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        self.assertTrue(torch.allclose(6 * torch.eye(2), script_out))
        self.assertTrue(torch.allclose(script_out, out))

    def test_eager_await_non_scriptable(self):
        # [FUNCTION DOC] In eager mode an awaitable can wrap a function on a self-referential type
        # that cannot be scripted; waiting yields the incremented value three.
        # Tree type can not be compiled (Recursive type)
        class Tree:
            # [CLASS DOC] HELPER: Recursive tree-like class with an optional parent and a value.
            def __init__(self, v):
                # [FUNCTION DOC] HELPER: Sets parent to none (annotated optional of the class) and
                # stores v.
                self.parent = torch.jit.annotate(Optional[Tree], None)
                self.v = v

        make_global(Tree)

        def delayed(t: Tree):
            # [FUNCTION DOC] HELPER: Increments the value and returns the same tree.
            t.v = t.v + 1
            return t

        aw = torch.jit._awaitable(delayed, Tree(2))
        t = torch.jit._awaitable_wait(aw)
        self.assertTrue(t.v == 3)

    def test_await_isinstance(self):
        # [FUNCTION DOC] Checks that while scripting the awaitable is an instance of the jit Await
        # type and that results match eager (two times eye plus ones).
        def delayed(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns twice the input plus one.
            return 2 * (x + 1)

        def main(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Creates an awaitable, asserts its type only when
            # scripting, waits.
            aw = torch.jit._awaitable(delayed, x)
            if torch.jit.is_scripting():
                assert isinstance(aw, torch.jit._Await)  # noqa: S101
            return torch.jit._awaitable_wait(aw)

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        self.assertTrue(
            torch.allclose(2 * torch.eye(2) + 2 * torch.ones(2), script_out)
        )
        self.assertTrue(torch.allclose(script_out, out))

    def test_await_eager_lazy(self):
        # [FUNCTION DOC] Eagerly created awaitable is an instance of the C-level Await class and
        # reports the same dtype as the wrapped result tensor.
        def delayed(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns twice the input plus one.
            return 2 * (x + 1)

        t = torch.ones(2, dtype=torch.int64)
        aw = torch.jit._awaitable(delayed, t)
        self.assertTrue(isinstance(aw, torch._C._Await))
        self.assertTrue(t.dtype == aw.dtype)

    def test_await_out_of_interpreter(self):
        # [FUNCTION DOC] A scripted function may return an awaitable to Python; waiting on it
        # outside the interpreter yields the same values as eager.
        def delayed(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns twice the input plus one.
            return 2 * (x + 1)

        def main(x: Tensor) -> Await[Tensor]:
            # [FUNCTION DOC] HELPER: Returns an awaitable of the delayed function without waiting.
            aw = torch.jit._awaitable(delayed, x)
            return aw

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out_aw = main(inp)
        out = torch.jit._awaitable_wait(out_aw)

        script_out_aw = sm(inp)
        script_out = torch.jit._awaitable_wait(script_out_aw)
        self.assertTrue(
            torch.allclose(2 * torch.eye(2) + 2 * torch.ones(2), script_out)
        )
        self.assertTrue(torch.allclose(script_out, out))

    def test_jit_trace(self):
        # [FUNCTION DOC] Traces a function using an awaitable and checks the traced module
        # reproduces the eager output on a new input.
        def gap(x: Tensor):
            # [FUNCTION DOC] HELPER: Returns relu plus sin of the input.
            return torch.relu(x) + torch.sin(x)

        def delayed(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns two times cosine plus one.
            return 2 * (torch.cos(x) + 1)

        def main(x: Tensor, y: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Starts awaitable, computes unrelated gap on y, then waits and
            # adds to y.
            aw = torch.jit._awaitable(delayed, x)
            z = gap(y)  # noqa: F841
            k = torch.jit._awaitable_wait(aw)
            return y + k

        inp = torch.randn(2)
        tm = torch.jit.trace(main, (inp, inp))
        inp_check = torch.ones(2)
        self.assertEqual(main(inp_check, inp_check), tm(inp_check, inp_check))

    def test_await_multiout_save(self):
        # [FUNCTION DOC] Awaitable returning a tuple with a list is scripted, compared with eager
        # and a hand expected value, then saved to a bytes buffer and reloaded with the
        # same result.
        def gap(x: Tensor):
            # [FUNCTION DOC] HELPER: Returns relu plus sin of the input.
            return torch.relu(x) + torch.sin(x)

        def delayed(x: Tensor) -> Tuple[Tensor, List[Tensor]]:
            # [FUNCTION DOC] HELPER: Returns a tuple of the scaled input and a list of
            # five multiples.
            l = [x * i for i in range(5)]
            return (100 * x, l)

        def main(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Starts awaitable, computes gap, waits, and returns the fourth
            # list element plus gap.
            aw = torch.jit._awaitable(delayed, x)
            z = gap(x)
            (_, l) = torch.jit._awaitable_wait(aw)
            return l[3] + z

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        expected = 4.8415 * torch.eye(2)
        self.assertTrue(torch.allclose(expected, script_out))
        self.assertTrue(torch.allclose(script_out, out))

        iofile = io.BytesIO()
        torch.jit.save(sm, iofile)
        iofile.seek(0)
        sm = torch.jit.load(iofile)
        script_out_load = sm(inp)
        self.assertTrue(torch.allclose(expected, script_out_load))

    def test_await_func_arg(self):
        # [FUNCTION DOC] Passes an awaitable as an argument to a scripted function; checks the
        # value against eager and expected, then a save/load round trip in a bytes buffer.
        def gap(x: Tensor):
            # [FUNCTION DOC] HELPER: Returns relu plus sin of the input.
            return torch.relu(x) + torch.sin(x)

        def delayed(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Returns the negated input.
            return -1 * x

        def fn(aw: Await[Tensor]) -> Tensor:
            # [FUNCTION DOC] HELPER: Waits on the awaitable argument and triples the result.
            return 3 * torch.jit._awaitable_wait(aw)

        def main(x: Tensor) -> Tensor:
            # [FUNCTION DOC] HELPER: Starts awaitable, computes gap, passes the awaitable to fn,
            # adds x.
            aw = torch.jit._awaitable(delayed, x)
            z = gap(x)  # noqa: F841
            y = fn(aw)
            return y + x

        inp = torch.eye(2)

        sm = torch.jit.script(main)
        out = main(inp)
        script_out = sm(inp)
        expected = -2 * torch.eye(2)
        self.assertTrue(torch.allclose(expected, script_out))
        self.assertTrue(torch.allclose(script_out, out))

        iofile = io.BytesIO()
        torch.jit.save(sm, iofile)
        iofile.seek(0)
        sm = torch.jit.load(iofile)
        script_out_load = sm(inp)
        self.assertTrue(torch.allclose(expected, script_out_load))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
