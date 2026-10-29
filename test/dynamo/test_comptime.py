# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for torch._dynamo.comptime (trace-time inspection hooks such
# as print, print_graph, print_disas, print_locals, print_bt, print_guards,
# get_local, graph_break and sleep). Defines the globals FILE and SELF used to
# pass data into comptime blocks and the ComptimeTests class. Import has no side
# effects beyond those globals; a __main__ guard runs the suite.

import collections
import re
import sys
import time
import typing
from io import StringIO

import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.comptime import comptime


# Because we don't support free variables in comptime at the moment,
# we have to communicate via globals.  This also means these tests cannot
# be run in parallel in a single process (not that you'd... ever want
# to do that?)
FILE = None
SELF = None


class ComptimeTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for the comptime facility of torch._dynamo.comptime: code run
    # at Dynamo trace time that can inspect locals, graph, bytecode, value stack,
    # guards and force graph breaks. Tests exchange data with comptime blocks through
    # the module globals FILE (a StringIO capture) and SELF (the test case), because
    # comptime blocks cannot use free variables. Not safe to run in parallel in
    # one process.
    def test_print_single(self):
        # [FUNCTION DOC] Prints many kinds of values at trace time (tensor, int, list,
        # tuple, dict, range, namedtuple, list subclass, sets, symbolic size) into the
        # global FILE and compares to an expected-inline string (symbolic size s77 under
        # dynamic=True). Then prints a defaultdict and only checks its name appears.
        # Asserts one compiled frame. Side effect: replaces the global FILE.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def comptime_print(e):
            # [FUNCTION DOC] HELPER: Wrapper that runs a comptime block printing the local e
            # into FILE through the context's print method.
            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints the traced variable for local e to
                # the global FILE.
                ctx.print(ctx.get_local("e"), file=FILE)

        class Employee(typing.NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with fields name and id, used as a print sample.
            name: object
            id: object

        class mylist(list):
            # [CLASS DOC] HELPER: Empty list subclass; printed as a user-defined list variable.
            pass

        @torch.compile(backend=cnt, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body (dynamic shapes): doubles x, then prints a
            # tensor, constants, containers, a range, a namedtuple, a list subclass, empty
            # and non-empty sets, and x.size(0); returns y plus 3.
            y = x * 2
            comptime_print(y)
            comptime_print(2)
            comptime_print([y, 2])
            comptime_print((y, 2))
            comptime_print({"foo": y})
            comptime_print(range(1, 3))
            comptime_print(Employee("foo", 2))
            comptime_print(mylist([1, 2]))
            comptime_print(set())
            comptime_print({"a", "b"})
            comptime_print(x.size(0))
            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
Tensor(shape=(s77,), dtype=torch.float32)
2
[Tensor(shape=(s77,), dtype=torch.float32), 2]
(Tensor(shape=(s77,), dtype=torch.float32), 2)
{'foo': Tensor(shape=(s77,), dtype=torch.float32)}
range(1, 3)
Employee(name='foo', id=2)
UserDefinedListVariable(mylist)
set()
{'a', 'b'}
s77""",
        )

        FILE = StringIO()

        @torch.compile(backend=cnt, dynamic=True)
        def g(x):
            # [FUNCTION DOC] HELPER: Compiled body: prints a defaultdict via comptime_print.
            comptime_print(collections.defaultdict(lambda: None))

        g(torch.randn(2))

        # it seems different pythons in CI change the
        # function str repr. Since this doesn't seem that
        # important, we just be very lenient
        self.assertIn(
            "defaultdict",
            FILE.getvalue().strip(),
        )

    def test_print_graph(self):
        # [FUNCTION DOC] Verifies print_graph (non-verbose, to FILE) shows the graph with
        # just the multiply by 2 captured so far, and that the compact comptime
        # print_graph form neither errors nor breaks the graph. One compiled frame.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, comptime block printing the graph,
            # the compact print_graph call, returns y plus 3.
            y = x * 2

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints the current graph without verbose
                # detail to FILE.
                ctx.print_graph(verbose=False, file=FILE)

            # Test the compact notation doesn't error or graph break;
            # you'll have to visually inspect to see that it printed
            comptime.print_graph()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = y = None""",
        )

    def test_print_disas(self):
        # [FUNCTION DOC] Prints the bytecode disassembly to FILE and checks it marks the
        # current instruction (the arrow marker), includes STORE_FAST, and has the
        # multiply op (BINARY_MULTIPLY before 3.11, else BINARY_OP). One compiled frame.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, comptime block printing the
            # disassembly, compact print_disas call, returns y plus 3.
            y = x * 2

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints the disassembly to FILE.
                ctx.print_disas(file=FILE)

            comptime.print_disas()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        out = FILE.getvalue()
        # Check that the instruction offset is working
        self.assertIn("-->", out)
        # Check that the bytecode resembles what we expect
        self.assertIn("STORE_FAST", out)
        if sys.version_info < (3, 11):
            self.assertIn("BINARY_MULTIPLY", out)
        else:
            self.assertIn("BINARY_OP", out)

    def test_print_value_stack(self):
        # [FUNCTION DOC] Checks that print_value_stack with stacklevel 1 shows the caller's
        # value stack (one tensor entry) when inlined function g is traced, and that
        # print_value_stack_and_return works. One frame, exact output compared.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def g(x):
            # [FUNCTION DOC] HELPER: Inlined callee: comptime block printing the stack one
            # level up, then returns x.
            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints the caller's value stack to FILE.
                ctx.print_value_stack(file=FILE, stacklevel=1)

            return x

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x + g(x); returns y plus the value of
            # print_value_stack_and_return on y*2.
            y = x + g(x)

            return y + comptime.print_value_stack_and_return(y * 2)

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue(),
            """\
- Tensor(shape=(2,), dtype=torch.float32)
""",
        )

    def test_print_locals(self):
        # [FUNCTION DOC] Prints the locals (x and y tensors) to FILE at trace time and
        # compares the exact text. One compiled frame.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, comptime block printing locals,
            # compact print_locals call, returns y plus 3.
            y = x * 2

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints locals to FILE.
                ctx.print_locals(file=FILE)

            comptime.print_locals()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue(),
            """\
x = Tensor(shape=(2,), dtype=torch.float32)
y = Tensor(shape=(2,), dtype=torch.float32)
""",
        )

    # Just make sure it doesn't crash
    def test_print_direct(self):
        # [FUNCTION DOC] Smoke test: comptime print on a captured variable z (also
        # referenced by a lambda so it is a closure cell) must not crash. No assertions.
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x, z):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, unused lambda capturing z,
            # comptime print of z, returns y plus 3.
            y = x * 2
            lambda: z
            comptime.print(z)
            return y + 3

        f(torch.randn(2), torch.randn(2))

    def test_sleep(self):
        # [FUNCTION DOC] Checks comptime.sleep delays compilation: runs once without
        # sleep and once with a 5 second sleep, asserts the
        # slow run exceeds 5 seconds and differs from the baseline by under 3 seconds.
        # Timing based, may be flaky.
        sleep_time = 5
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x, z, should_sleep):
            # [FUNCTION DOC] HELPER: Compiled body: optionally calls comptime sleep for the
            # captured sleep_time, then y = x*2, returns y plus 3.
            if should_sleep:
                comptime.sleep(sleep_time)
            y = x * 2
            return y + 3

        start = time.time()
        f(torch.randn(2), torch.randn(2), False)
        total_no_sleep = time.time() - start

        start = time.time()
        f(torch.randn(2), torch.randn(2), True)
        total_with_sleep = time.time() - start

        self.assertTrue(total_with_sleep > sleep_time)
        # Hopefully this won't be flaky
        self.assertTrue(abs(total_with_sleep - sleep_time - total_no_sleep) < 3)

    # Just make sure it doesn't crash
    def test_get_local_closure_variable(self):
        # [FUNCTION DOC] Smoke test: get_local can read z, a closure variable of the
        # enclosing traced function, from a comptime block in an inlined function; its
        # repr must be "3". Sets the global SELF to the test case.
        global SELF
        SELF = self
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: z = 3, calls g, returns y plus 3.
            z = 3

            def g():
                # [FUNCTION DOC] HELPER: Inlined callee: comptime block, prints z, returns 2.
                @comptime
                def _(ctx):
                    # [FUNCTION DOC] HELPER: comptime block: asserts the repr of local z is "3".
                    r = ctx.get_local("z")
                    SELF.assertEqual(repr(r), "3")

                comptime.print(z)
                return 2

            y = x * g()
            return y + 3

        f(torch.randn(2))

    def test_print_bt(self):
        # [FUNCTION DOC] Prints the trace-time backtrace to FILE and checks that its text
        # contains the source line calling g. One compiled frame.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        def g(x):
            # [FUNCTION DOC] HELPER: Inlined callee: comptime block printing the backtrace to
            # FILE, compact print_bt call, returns x plus 3.
            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints the backtrace to FILE.
                ctx.print_bt(file=FILE)

            comptime.print_bt()

            return x + 3

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, y = g(y), returns y plus 3.
            y = x * 2
            y = g(y)
            return y + 3

        def munge_filenames(s):
            # [FUNCTION DOC] HELPER: Normalizes File "..." line N entries to a fixed form;
            # defined but never called in this test.
            return re.sub(r'File "[^"]+", line \d+', 'File "X", line X', s)

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        bt = FILE.getvalue()
        self.assertIn("y = g(y)", bt)

    def test_print_guards(self):
        # [FUNCTION DOC] Prints the active guards to FILE and compares to an expected
        # text (tensor match for x plus global-state and shape-env guards) after
        # stripping trailing whitespace. One compiled frame.
        global FILE
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, comptime block printing guards,
            # compact print_guards call, returns y plus 3.
            y = x * 2

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: prints guards to FILE.
                ctx.print_guards(file=FILE)

            comptime.print_guards()

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            re.sub(r"\s+$", "", FILE.getvalue().rstrip(), flags=re.MULTILINE),
            """\

        local "L['x']" TENSOR_MATCH
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' AUTOGRAD_SAVED_TENSORS_HOOKS
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' GRAD_MODE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' DETERMINISTIC_ALGORITHMS
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' GLOBAL_STATE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' TORCH_FUNCTION_STATE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        global '' DEFAULT_DEVICE
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }
        shape_env '' SHAPE_ENV
        {
            'guard_types': None,
            'code': None,
            'obj_weakref': None
            'guarded_class': None
        }""",
        )

    def test_graph_break(self):
        # [FUNCTION DOC] A comptime block that does nothing yields one frame. One with
        # ctx.graph_break plus a compact comptime graph_break call yields three frames
        # (the counter is reset in between).
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body with an empty comptime block, returns y plus 3.
            y = x * 2

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: Empty comptime block (no effect).
                pass

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        cnt.frame_count = 0

        @torch.compile(backend=cnt)
        def g(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, break via comptime block, y+2,
            # compact graph_break, returns y*3.
            y = x * 2

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: forces a graph break.
                ctx.graph_break()

            y = y + 2

            comptime.graph_break()

            return y * 3

        g(torch.randn(2))
        self.assertEqual(cnt.frame_count, 3)

    def test_get_local(self):
        # [FUNCTION DOC] Exercises ComptimeVar from get_local: fake tensor size, size,
        # as_proxy (adds a real graph node), python_type, and as_python_constant for an
        # int local. Checks the printed graph includes the extra add node. One frame.
        # Sets the global SELF and FILE.
        global SELF, FILE
        SELF = self
        FILE = StringIO()
        cnt = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnt)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled body: y = x*2, lit = 2, comptime block, returns y
            # plus 3.
            y = x * 2
            lit = 2  # noqa: F841

            @comptime
            def _(ctx):
                # [FUNCTION DOC] HELPER: comptime block: asserts size, python type and constant
                # value of locals y and lit, adds 4 to y's proxy, prints the graph to FILE.
                y = ctx.get_local("y")
                SELF.assertEqual(y.as_fake().size(0), 2)
                SELF.assertEqual(y.size(0), 2)
                # Trigger a graph write (TODO: this is not so
                # useful right now as there's no way to make use
                # of the output proxy; maybe it's useful for inserting
                # side-effectful operations into the graph)
                y.as_proxy() + 4
                ctx.print_graph(verbose=False, file=FILE)
                SELF.assertIs(y.python_type(), torch.Tensor)
                lit = ctx.get_local("lit")
                SELF.assertEqual(lit.as_python_constant(), 2)

            return y + 3

        f(torch.randn(2))
        self.assertEqual(cnt.frame_count, 1)
        self.assertExpectedInline(
            FILE.getvalue().strip(),
            """\
def forward(self, L_x_ : torch.Tensor):
    l_x_ = L_x_
    y = l_x_ * 2;  l_x_ = None
    add = y + 4;  y = add = None""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
