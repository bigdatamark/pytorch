# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that dynamo's bytecode transformation machinery keeps simple functions
# correct when two NOP instructions are inserted at the start of every frame. Defines a few
# sample functions (plain arithmetic, a closure with nonlocal, a generator), a decorator
# built from the debug_insert_nops callback, and one test class. At import time it builds
# the decorator with no hooks and reads a module global used by the sample functions.
import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo import eval_frame
from torch._dynamo.hooks import Hooks


c = 10


def fn1(a, b):
    # [FUNCTION DOC] HELPER: Returns the sum of a and b minus the module global c.
    return a + b - c


def fn2(a, b):
    # [FUNCTION DOC] HELPER: Runs a nested closure twice that adds a, b and global c into
    # a nonlocal accumulator, then returns the accumulator plus one.
    x = 0
    y = 1

    def modify():
        # [FUNCTION DOC] HELPER: Adds a + b + c to the enclosing nonlocal accumulator x.
        nonlocal x
        x += a + b + c

    for _ in range(2):
        modify()

    return x + y


def fn3():
    # [FUNCTION DOC] HELPER: Generator that yields 1 then 2 and finishes.
    yield 1
    yield 2


with_debug_nops = eval_frame._optimize_catch_errors(
    torch._dynamo.testing.debug_insert_nops, Hooks(None, None)
)


class NopTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Runs sample functions under the NOP-inserting frame callback and checks
    # results are unchanged. Stateless; each test is independent.
    @with_debug_nops
    def test1(self):
        # [FUNCTION DOC] Calls fn1 twice (second call exercises the cache) and checks -7.
        self.assertEqual(fn1(1, 2), -7)
        self.assertEqual(fn1(1, 2), -7)

    @with_debug_nops
    def test2(self):
        # [FUNCTION DOC] Calls the closure-based fn2 twice and checks both results are 27.
        self.assertEqual(fn2(1, 2), 27)
        self.assertEqual(fn2(1, 2), 27)

    @with_debug_nops
    def test3(self):
        # [FUNCTION DOC] Iterates the generator fn3 and checks it yields 1, 2, then raises
        # StopIteration.
        t = fn3()
        self.assertEqual(next(t), 1)
        self.assertEqual(next(t), 2)
        self.assertRaises(StopIteration, lambda: next(t))

    def test_extended_args(self):
        # [FUNCTION DOC] Builds a lambda via eval whose body is a very long chain of
        # additions (512 operands per branch) so jump offsets need EXTENDED_ARG, wraps it
        # with the NOP decorator, and checks the sum over ones tensors equals 513.
        too_many_adds = "+".join(["a", "b"] * 256)
        source = (
            f"lambda a, b: ({too_many_adds}+a if a.sum() > 0 else {too_many_adds} - b)"
        )
        fn = eval(source)
        a = torch.ones(1)
        b = torch.ones(1)
        fn = with_debug_nops(fn)
        self.assertEqual(fn(a, b).sum(), 513)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
