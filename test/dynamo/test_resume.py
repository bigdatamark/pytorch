# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo resume functions (created after a graph break) keep the right
# free variables when the original function is a closure.

import torch
import torch._dynamo.test_case


def fn_creator():
    # [FUNCTION DOC] HELPER: Build and return a closure fn that captures one outer variable. The
    # inner function adds one to its input, calls graph_break, adds the captured value, defines
    # an unused nested function capturing a local, and returns the tensor. Used as test input.
    var1 = 1

    def fn(x):
        # [FUNCTION DOC] HELPER: Function under test. Adds 1 to x, graph-breaks, then adds the
        # captured variable var1 and returns the result. The local var2 becomes a cell variable
        # because the unused nested function inner_fn captures it.
        x = x + 1
        var2 = 1
        torch._dynamo.graph_break()
        x = x + var1

        def inner_fn():
            # [FUNCTION DOC] HELPER: Never called; exists only to make var2 a cell variable of
            # fn. Returns the captured var2.
            return var2

        return x

    return fn


class ResumeFunctionTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case checking the shape of resume functions that Dynamo installs into the
    # module globals after a graph break.
    def test_freevars(self):
        # [FUNCTION DOC] Compiles a closure with the eager backend and runs it once on a random
        # tensor, then finds the globals whose names start with the resume prefix. Asserts there
        # is exactly one, unwraps its factory closure to get the code object, and asserts the
        # resume code's free variables are var1 and var2, i.e. the sorted union of the original
        # function's free and cell variables.
        fn = fn_creator()
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(torch.randn(10))
        entries = [v for k, v in list(globals().items()) if k.startswith("__resume_at")]
        self.assertEqual(len(entries), 1)
        # When freevars are present, install_resume_function_global stores a
        # factory that closes over the code object (first closure cell).
        code = entries[0].__closure__[0].cell_contents
        # co_freevars of resume functions, are sorted concatenation of the original function's co_freevars and co_cellvars
        self.assertEqual(code.co_freevars, ("var1", "var2"))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
