# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo skips (produces no ops for) frames that do no tensor work,
# using op counts from a counting backend with the assert-style optimizer, plus a test that
# functions calling the compiling check still get committed so side effects match eager. Defines
# module globals used as counters and small helpers: a function and a generator returning the
# compiling flag, and a module whose forward behavior depends on an integer mode.
from unittest.mock import patch

import torch
import torch._dynamo
import torch._dynamo.test_case
from torch._dynamo.testing import CompileCounter


_variable = 0
_variable_2 = 0


def user_function():
    # [FUNCTION DOC] HELPER: Returns whether the compiler is currently tracing, as reported by
    # the compiler is_compiling API.
    return torch.compiler.is_compiling()


def user_generator():
    # [FUNCTION DOC] HELPER: Generator that yields the compiling flag exactly once, then
    # returns.
    for _ in range(1):
        yield torch.compiler.is_compiling()
    return


class MyModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Test module whose behavior is selected by an integer mode. Each mode
    # (1 to 6) records through module globals whether code runs under compilation, via a direct
    # check, a helper function, a lambda, a generator, a pre-forward hook, or a graph break.
    # State: mode attribute. Registers a forward pre-hook at construction.
    def __init__(self, mode: int):
        # [FUNCTION DOC] HELPER: Stores the mode and registers the pre_forward method as a
        # forward pre-hook that receives keyword arguments.
        super().__init__()
        self.mode = mode
        self.register_forward_pre_hook(self.pre_forward, with_kwargs=True)

    def pre_forward(self, module, args, kwargs):
        # [FUNCTION DOC] HELPER: Forward pre-hook. Only in mode 5, if the helper function
        # reports compiling, increments the global counter _variable. Always returns the
        # arguments and keyword arguments unchanged.
        if self.mode == 5:
            if user_function():
                global _variable
                _variable += 1
        return args, kwargs

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Runs the mode specific side effect on module globals. Mode 1
        # increments _variable if compiling else _variable_2; modes 2, 3 and 4 increment
        # _variable if compiling is detected via a helper, a lambda or a generator; mode 5 adds
        # one in place to x; mode 6 graph-breaks and then increments _variable if the helper
        # reports compiling. Returns x.
        global _variable, _variable_2

        if self.mode == 1:
            if torch.compiler.is_compiling():
                _variable += 1
            else:
                _variable_2 += 1
        elif self.mode == 2:
            if user_function():
                _variable += 1
        elif self.mode == 3:
            lambda_f = lambda: torch.compiler.is_compiling()  # noqa: E731
            if lambda_f():
                _variable += 1
        elif self.mode == 4:
            for cond in user_generator():
                if cond:
                    _variable += 1
        elif self.mode == 5:
            x += 1
        elif self.mode == 6:
            if user_function():
                torch._dynamo.graph_break()
                _variable += 1
        return x


class SkipNonTensorTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case checking that frames with no tensor ops compile to zero ops, and
    # that such frames still run when they have observable side effects.
    def test_add_tensor1(self):
        # [FUNCTION DOC] Compiles a function adding a tensor and a Python int under the assert-
        # style optimizer with a counting backend, calls it, and raises an error unless exactly
        # one op was counted.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return a + b

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn(x, y)

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_tensor2(self):
        # [FUNCTION DOC] Like the first add test but uses the torch add function; calls with a
        # tensor and an int and raises an error unless exactly one op was counted.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns torch add of its two arguments.
            return torch.add(a, b)

        counter = CompileCounter()

        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn(x, y)

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_tensor_list(self):
        # [FUNCTION DOC] Passes a list holding a tensor and an int; the function adds the two
        # elements. Raises an error unless exactly one op was counted.
        def fn(lst):
            # [FUNCTION DOC] HELPER: Returns the sum of the first two elements of the list.
            return lst[0] + lst[1]

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn([x, y])

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_tensor_dict(self):
        # [FUNCTION DOC] Passes a dictionary holding a tensor and an int; the function adds the
        # entries a and b. Raises an error unless exactly one op was counted.
        def fn(dt):
            # [FUNCTION DOC] HELPER: Returns the sum of the dictionary values stored under keys
            # a and b.
            return dt["a"] + dt["b"]

        counter = CompileCounter()
        x = torch.randn(4)
        y = 5
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        opt_fn({"a": x, "b": y})

        if counter.op_count != 1:
            raise AssertionError(f"Expected op_count 1, got {counter.op_count}")

    def test_add_skip(self):
        # [FUNCTION DOC] Calls a compiled addition of two plain ints and raises an error unless
        # zero ops were counted, i.e. the frame is skipped.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of its two arguments.
            return a + b

        counter = CompileCounter()
        opt_fn = torch._dynamo.optimize_assert(counter)(fn)
        x = 4
        y = 5
        opt_fn(x, y)

        if counter.op_count != 0:
            raise AssertionError(f"Expected op_count 0, got {counter.op_count}")

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_recursive_list(self):
        # [FUNCTION DOC] Runs an identity function on a self-referential list inside the assert-
        # style optimizer context manager, with the context manager usage error disabled by a
        # config patch. Raises an error unless zero ops were counted.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns its argument unchanged.
            return x

        counter = CompileCounter()

        x = []
        x.append(x)
        with torch._dynamo.optimize_assert(counter):
            fn(x)

        if counter.op_count != 0:
            raise AssertionError(f"Expected op_count 0, got {counter.op_count}")

    @patch.object(torch._dynamo.config, "raise_on_ctx_manager_usage", False)
    def test_custom_list(self):
        # [FUNCTION DOC] Runs a function indexing the first two elements of a list subclass
        # whose iteration and length raise errors, filled with two tensors, under the context
        # manager form with the usage error patched off. Raises an error unless zero ops were
        # counted, so the frame is skipped.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the sum of elements at index zero and one of x.
            return x[0] + x[1]

        counter = CompileCounter()

        class Foo(list):
            # [CLASS DOC] HELPER: List subclass whose iteration and length operations always
            # raise a generic exception, to check Dynamo does not try to inspect it.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Always raises a generic exception.
                raise Exception  # noqa: TRY002

            def __len__(self):
                # [FUNCTION DOC] HELPER: Always raises a generic exception.
                raise Exception  # noqa: TRY002

        x = Foo()
        x.append(torch.randn(4))
        x.append(torch.randn(4))
        with torch._dynamo.optimize_assert(counter):
            fn(x)

        if counter.op_count != 0:
            raise AssertionError(f"Expected op_count 0, got {counter.op_count}")

    def test_do_not_skip_side_effects(self):
        # [FUNCTION DOC] Regression test for issue 110765. For each module mode from 1 to 6 it
        # resets Dynamo, zeroes both globals, builds the mode module and compiles it with the
        # eager backend (full graph except mode 6), asserts nothing ran at compile time, then
        # calls it twice with a one-element tensor. After each call it asserts the first counter
        # has grown by one and the second counter stays zero, so compile-time checks ran as
        # compiling code and side effects were not skipped. Raises an error on mismatch.
        # https://github.com/pytorch/pytorch/issues/110765

        # By invoking torch.compiler.is_compiling(),
        # there may be side-effects inconsistent with eager when
        # compiling. Thus we force dynamo to commit the graph,
        # even if it does not perform any tensor operation
        global _variable, _variable_2

        for mode in range(1, 7):
            torch._dynamo.reset()

            _variable = 0
            _variable_2 = 0

            mod = MyModule(mode=mode)
            model = torch.compile(mod, backend="eager", fullgraph=mode != 6)
            if _variable != 0:
                raise AssertionError(f"Expected _variable 0, got {_variable}")
            if _variable_2 != 0:
                raise AssertionError(f"Expected _variable_2 0, got {_variable_2}")

            model(torch.tensor([1]))
            if _variable != 1:
                raise AssertionError(f"Expected _variable 1, got {_variable}")
            if _variable_2 != 0:
                raise AssertionError(f"Expected _variable_2 0, got {_variable_2}")

            model(torch.tensor([1]))
            if _variable != 2:
                raise AssertionError(f"Expected _variable 2, got {_variable}")
            if _variable_2 != 0:
                raise AssertionError(f"Expected _variable_2 0, got {_variable_2}")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
