# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for contextvars.ContextVar support under torch.compile. Covers get
# with/without defaults, argument errors, guards and recompilation on value change (including
# copy_context and threads), graph breaks on set/reset, and values of various types. No import-
# time side effects.
import contextvars
import threading

import torch
import torch._dynamo
import torch._dynamo.testing
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter


class TestContextVars(TestCase):
    # [CLASS DOC] Test case holding all ContextVar tracing tests. Each test creates its own
    # ContextVar, compiles a function reading it with the eager or a CompileCounter backend, and
    # checks results and frame counts.
    def test_get_with_constructor_default(self):
        # [FUNCTION DOC] get() with a constructor default returns that default under fullgraph;
        # result is x + 1.
        cv = contextvars.ContextVar("precision", default="fp32")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: read the variable with get() and add 1 or 2 depending on
            # its value.
            p = cv.get()
            if p == "fp32":
                return x + 1
            return x + 2

        x = torch.randn(4)
        ref = x + 1
        self.assertEqual(fn(x), ref)

    def test_get_with_explicit_default(self):
        # [FUNCTION DOC] get() with an explicit positional fallback and no value set returns the
        # fallback; result is x * 2.
        cv = contextvars.ContextVar("test_explicit")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: get() with a positional fallback; double x when the
            # fallback is returned, else triple.
            val = cv.get("fallback")
            if val == "fallback":
                return x * 2
            return x * 3

        x = torch.randn(4)
        self.assertEqual(fn(x), x * 2)

    def test_get_rejects_keyword_default(self):
        # [FUNCTION DOC] get() with a keyword default argument raises TypeError when the
        # compiled function runs.
        cv = contextvars.ContextVar("test_keyword")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: call get() with a keyword default (expected to raise
            # TypeError) and return x.
            cv.get(default="kw_fallback")
            return x

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            fn(x)

    def test_get_rejects_extra_positional_args(self):
        # [FUNCTION DOC] get() with two positional arguments raises TypeError when the compiled
        # function runs.
        cv = contextvars.ContextVar("test_extra_args")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: call get() with two positional arguments (expected to raise
            # TypeError) and return x.
            cv.get(1, 2)
            return x

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            fn(x)

    def test_get_with_set_value(self):
        # [FUNCTION DOC] get() sees a value set before compiling; the value is reset in a
        # finally block.
        cv = contextvars.ContextVar("test_set", default="default")
        token = cv.set("active")

        try:

            @torch.compile(backend="eager", fullgraph=True)
            def fn(x):
                # [FUNCTION DOC] HELPER: return x + 10 if the variable reads as active, else x.
                val = cv.get()
                if val == "active":
                    return x + 10
                return x

            x = torch.randn(4)
            self.assertEqual(fn(x), x + 10)
        finally:
            cv.reset(token)

    def test_get_no_default_no_value(self):
        # [FUNCTION DOC] get() on a variable without default or value raises LookupError from
        # the compiled function.
        cv = contextvars.ContextVar("no_default")

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: call get() on an unset variable without default (expected
            # to raise LookupError).
            cv.get()
            return x

        x = torch.randn(4)
        with self.assertRaises(LookupError):
            fn(x)

    def test_recompilation_on_value_change(self):
        # [FUNCTION DOC] changing the variable value forces a recompile: frame_count goes from 1
        # to 2.
        cv = contextvars.ContextVar("recompile_test", default="a")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x + 1 if the variable reads a, else x + 2.
            val = cv.get()
            if val == "a":
                return x + 1
            return x + 2

        x = torch.randn(4)

        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)

        token = cv.set("b")
        try:
            self.assertEqual(fn(x), x + 2)
            self.assertEqual(cnt.frame_count, 2)
        finally:
            cv.reset(token)

    def test_no_recompilation_same_value(self):
        # [FUNCTION DOC] repeated calls with an unchanged value compile once:
        # frame_count stays 1.
        cv = contextvars.ContextVar("no_recompile", default="stable")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: read the variable and return x + 1.
            cv.get()
            return x + 1

        x = torch.randn(4)
        fn(x)
        fn(x)
        fn(x)
        self.assertEqual(cnt.frame_count, 1)

    def test_multiple_cvs(self):
        # [FUNCTION DOC] two ContextVars read in one fullgraph function are both honoured.
        cv1 = contextvars.ContextVar("cv1", default=1)
        cv2 = contextvars.ContextVar("cv2", default=2)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x plus the values of both variables.
            a = cv1.get()
            b = cv2.get()
            return x + a + b

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 3)

    def test_fullgraph_get_only(self):
        # [FUNCTION DOC] a function that only calls get() compiles with fullgraph.
        cv = contextvars.ContextVar("fullgraph_ok", default="yes")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: read the variable and return x + 1.
            cv.get()
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_fullgraph_set_errors(self):
        # [FUNCTION DOC] calling set() under fullgraph raises torch._dynamo.exc.Unsupported.
        cv = contextvars.ContextVar("fullgraph_set", default="val")

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: call set() on the variable and return x.
            cv.set("new")
            return x

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            fn(x)

    def test_set_graph_breaks(self):
        # [FUNCTION DOC] set() causes a graph break: two frames are compiled.
        cv = contextvars.ContextVar("set_breaks", default="old")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: add 1, call set() (graph break), then add 2.
            x = x + 1
            cv.set("new")
            return x + 2

        x = torch.randn(4)
        fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_reset_graph_breaks(self):
        # [FUNCTION DOC] reset() with a token causes a graph break: two frames are compiled.
        cv = contextvars.ContextVar("reset_breaks", default="val")
        token = cv.set("tmp")

        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: add 1, call reset() with the outer token (graph break),
            # then add 2.
            x = x + 1
            cv.reset(token)
            return x + 2

        x = torch.randn(4)
        fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_cv_name_attribute(self):
        # [FUNCTION DOC] the name attribute of a ContextVar is readable under fullgraph.
        cv = contextvars.ContextVar("my_var_name", default=0)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x + 1 if the variable name equals
            # my_var_name, else x.
            name = cv.name
            if name == "my_var_name":
                return x + 1
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_nn_module_forward(self):
        # [FUNCTION DOC] a ContextVar read inside an nn.Module forward compiled with fullgraph.
        cv = contextvars.ContextVar("module_cv", default="train")

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward reads a ContextVar to choose behaviour.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: return x * 2 when the variable reads train, else x.
                mode = cv.get()
                if mode == "train":
                    return x * 2
                return x

        mod = MyModule()
        compiled = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), x * 2)

    def test_cv_across_graph_break(self):
        # [FUNCTION DOC] values read before and after a graph break (a print) are consistent;
        # two frames compiled.
        cv = contextvars.ContextVar("graph_break_cv", default="before")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: read the variable, add 1 or 2 by value, print (graph
            # break), read again and add accordingly.
            val = cv.get()
            if val == "before":
                x = x + 1
            print("graph break")
            val2 = cv.get()
            if val2 == "before":
                x = x + 2
            return x

        x = torch.randn(4)
        result = fn(x)
        self.assertEqual(result, x + 3)
        self.assertEqual(cnt.frame_count, 2)

    def test_cv_as_function_arg(self):
        # [FUNCTION DOC] a ContextVar passed as argument to an inlined helper is read correctly.
        cv = contextvars.ContextVar("arg_cv", default="hello")

        def read_cv(c):
            # [FUNCTION DOC] HELPER: return the current value of the ContextVar c via get().
            return c.get()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: read the variable via read_cv and add 5 when it
            # equals hello.
            val = read_cv(cv)
            if val == "hello":
                return x + 5
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 5)

    def test_cv_in_closure(self):
        # [FUNCTION DOC] a ContextVar captured by a closure of the compiled function
        # is read correctly.
        cv = contextvars.ContextVar("closure_cv", default="closed")

        def make_fn():
            # [FUNCTION DOC] HELPER: build and return an inner function that closes over
            # the ContextVar.
            def fn(x):
                # [FUNCTION DOC] HELPER: return x - 1 when the closed-over variable reads
                # closed, else x.
                val = cv.get()
                if val == "closed":
                    return x - 1
                return x

            return fn

        compiled = torch.compile(make_fn(), backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(compiled(x), x - 1)

    def test_tensor_value(self):
        # [FUNCTION DOC] a tensor default value is usable in tensor arithmetic under fullgraph.
        t = torch.tensor([1.0, 2.0, 3.0])
        cv = contextvars.ContextVar("tensor_cv", default=t)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: add the tensor value of the variable to x.
            bias = cv.get()
            return x + bias

        x = torch.randn(3)
        self.assertEqual(fn(x), x + t)

    def test_guard_none_value(self):
        # [FUNCTION DOC] a None default value is read and guarded correctly.
        cv = contextvars.ContextVar("none_cv", default=None)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x + 1 if the value is None, else x + 2.
            val = cv.get()
            if val is None:
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)

    def test_list_value(self):
        # [FUNCTION DOC] a list default value can be read and measured with len under fullgraph.
        cv = contextvars.ContextVar("list_cv", default=[10, 20])  # noqa: B039

        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x plus the length of the list value.
            items = cv.get()
            return x + len(items)

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 2)

    def test_list_default_graph_breaks(self):
        # [FUNCTION DOC] passing a fresh list as the get() default causes a graph break: two
        # frames compiled.
        cv = contextvars.ContextVar("list_default")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: add 1, then read the variable with a list default and add
            # its length.
            x = x + 1
            val = cv.get([1, 2, 3])
            return x + len(val)

        x = torch.randn(4)
        fn(x)
        self.assertEqual(cnt.frame_count, 2)

    def test_recompilation_on_context_run(self):
        # [FUNCTION DOC] running the compiled function in a copied context where the variable
        # was changed recompiles (frame_count 2).
        cv = contextvars.ContextVar("ctx_run", default="original")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x + 1 if the variable is original, else x + 2.
            val = cv.get()
            if val == "original":
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)

        ctx = contextvars.copy_context()
        ctx.run(cv.set, "changed")
        result = ctx.run(fn, x)
        self.assertEqual(result, x + 2)
        self.assertEqual(cnt.frame_count, 2)

    def test_recompilation_across_threads(self):
        # [FUNCTION DOC] a different thread setting the variable triggers a recompile; the
        # thread result and any error are collected and checked.
        cv = contextvars.ContextVar("thread_cv", default="main")
        cnt = CompileCounter()

        @torch.compile(backend=cnt)
        def fn(x):
            # [FUNCTION DOC] HELPER: return x + 1 if the variable is main, else x + 2.
            val = cv.get()
            if val == "main":
                return x + 1
            return x + 2

        x = torch.randn(4)
        self.assertEqual(fn(x), x + 1)
        self.assertEqual(cnt.frame_count, 1)

        results = []
        errors = []

        def thread_fn():
            # [FUNCTION DOC] HELPER: runs in a new thread: set the variable to thread, call the
            # compiled function and append the result to results, or append any
            # exception to errors.
            try:
                cv.set("thread")
                results.append(fn(x))
            except Exception as e:
                errors.append(e)

        t = threading.Thread(target=thread_fn)
        t.start()
        t.join()
        self.assertEqual(len(errors), 0, f"Thread raised: {errors}")
        self.assertEqual(results[0], x + 2)
        self.assertEqual(cnt.frame_count, 2)


if __name__ == "__main__":
    run_tests()
