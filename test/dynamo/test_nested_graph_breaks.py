# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for nested graph breaks in dynamo, where a graph break inside an inlined
# callee is handled by resuming the whole inlined call stack rather than skipping the caller
# frames. Covers cells, globals, context managers, generators, loops, try blocks, class
# construction, pytree mapping and error-on-break. Defines a module-level global tensor tuple
# and a context manager helper class, and imports a sibling helper module (relative first,
# then absolute).
import sys
import unittest

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.utils._pytree as python_pytree


try:
    from . import _test_nested_graph_breaks_helper
except ImportError:
    import _test_nested_graph_breaks_helper


# for use in test_side_effects_globals
global1, global2, global3, global4 = (torch.zeros(3),) * 4


class CustomizedCtxManager:
    # [CLASS DOC] HELPER: Context manager that stores its argument and forces a graph break in
    # its constructor. Enter and exit do nothing.
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the argument on the instance and calls graph_break, so
        # the break happens before the context is entered.
        self.x = x
        torch._dynamo.graph_break()

    def __enter__(self):
        # [FUNCTION DOC] HELPER: Does nothing.
        pass

    def __exit__(self, exc_type, exc_value, traceback):
        # [FUNCTION DOC] HELPER: Does nothing; does not suppress exceptions.
        pass


class NestedGraphBreakTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for nested graph breaks, where a graph break inside an inlined callee
    # splits the caller into resume frames instead of skipping the caller. Each test builds a
    # chain of small functions, compiles the outermost with a counting backend, and checks
    # results against eager along with frame and op counts. Stateless apart from global helper
    # names declared by some tests.
    def test_single_graph_break(self):
        # [FUNCTION DOC] Chain of three functions with a graph break in the innermost; the helper
        # names are declared global so they are not closure variables. Expects equal results, 2
        # frames and 6 ops.
        # NOTE marking f1, f2, f3 as global
        # prevents them from being freevars
        global f1, f2, f3

        def f1(x1):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x1 = x1 + 1
            torch._dynamo.graph_break()
            return x1 + 2

        def f2(x2):
            # [FUNCTION DOC] HELPER: Calls f1 on the input plus four and adds eight.
            return f1(x2 + 4) + 8

        def f3(x3):
            # [FUNCTION DOC] HELPER: Calls f2 on the input plus sixteen and adds thirty-two.
            return f2(x3 + 16) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_single_graph_break_repeat(self):
        # [FUNCTION DOC] Like the single break test but the middle function calls the breaking
        # function twice (second result shifted left by four) and the outer adds constants, using
        # long tensors. Expects 3 frames and 10 ops.
        global f1, f2, f3

        def f1(x1):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x1 = x1 + 1
            torch._dynamo.graph_break()
            return x1 + 2

        def f2(x2):
            # [FUNCTION DOC] HELPER: Calls f1 twice on shifted inputs, shifts the second result
            # left by four and returns the sum.
            tmp1 = f1(x2 + 4)
            tmp2 = f1(x2 + 8) << 4
            return tmp1 + tmp2

        def f3(x3):
            # [FUNCTION DOC] HELPER: Calls f2 on the input plus 256 and adds 512.
            return f2(x3 + 256) + 512

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3, dtype=torch.long)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 10)

    def test_doubly_nested_graph_break(self):
        # [FUNCTION DOC] Graph breaks at two nesting levels (f1 and f2). Expects equal results, 3
        # frames and 7 ops.
        global f1, f2, f3

        def f1(x1):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x1 = x1 + 1
            torch._dynamo.graph_break()
            return x1 + 2

        def f2(x2):
            # [FUNCTION DOC] HELPER: Adds four, forces a graph break, then calls f1 on the input
            # plus eight and adds sixteen.
            x2 = x2 + 4
            torch._dynamo.graph_break()
            return f1(x2 + 8) + 16

        def f3(x3):
            # [FUNCTION DOC] HELPER: Calls f2 on the input plus 32 and adds 64.
            return f2(x3 + 32) + 64

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 7)

    def test_differing_arg_nums(self):
        # [FUNCTION DOC] Chain of four functions with differing argument counts (2, 4, 2, 1)
        # where the innermost breaks; checks the resume functions pass arguments correctly.
        # Expects 2 frames and 10 ops.
        global f1, f2, f3, f4

        def f1(x1, x2):
            # [FUNCTION DOC] HELPER: Adds its two inputs, forces a graph break, then adds one.
            x = x1 + x2
            torch._dynamo.graph_break()
            return x + 1

        def f2(x3, x4, x5, x6):
            # [FUNCTION DOC] HELPER: Calls f1 with the pair sums of its four inputs and adds two.
            return f1(x3 + x4, x5 + x6) + 2

        def f3(x7, x8):
            # [FUNCTION DOC] HELPER: Calls f2 with four derived arguments built from its two
            # inputs and adds 16.
            return f2(x7, x7 + 4, x8, x8 + 8) + 16

        def f4(x9):
            # [FUNCTION DOC] HELPER: Calls f3 with the input and the input plus 32 and adds 64.
            return f3(x9, x9 + 32) + 64

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f4)
        x = torch.zeros(3)
        res = f4(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 10)

    def test_differing_locals_nums(self):
        # [FUNCTION DOC] Chain of three functions with different numbers of local variables (1,
        # 3, 5) where the innermost breaks; checks locals survive in resume frames. Expects 2
        # frames and 14 ops.
        global f1, f2, f3

        def f1(x1):
            # [FUNCTION DOC] HELPER: Stores a local sum, forces a graph break, then adds two to
            # the saved local.
            loc1 = x1 + 1
            torch._dynamo.graph_break()
            return loc1 + 2

        def f2(x2):
            # [FUNCTION DOC] HELPER: Defines two locals, calls f1 and returns the sum of
            # all three.
            loc1 = x2 + 4
            loc2 = x2 + 8
            return f1(x2) + loc1 + loc2

        def f3(x3):
            # [FUNCTION DOC] HELPER: Defines four locals, calls f2 and returns the sum of
            # all five.
            loc1 = x3 + 16
            loc2 = x3 + 32
            loc3 = x3 + 64
            loc4 = x3 + 128
            return f2(x3) + loc1 + loc2 + loc3 + loc4

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 14)

    def test_counters(self):
        # [FUNCTION DOC] Checks the graph_break counter has exactly two entries after calling a
        # compiled function whose callee chain (including a loop with one iteration) hits a graph
        # break, and checks the numeric result 255.
        global f1, f2, f3, f4

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 on the input plus four and adds eight.
            return f1(x + 4) + 8

        def f3(x):
            # [FUNCTION DOC] HELPER: Adds sixteen, calls f2 inside a one-iteration loop, then
            # adds thirty-two.
            x = x + 16
            for _ in range(1):
                x = f2(x)
            return x + 32

        @torch.compile(backend="eager")
        def f4(x):
            # [FUNCTION DOC] HELPER: Compiled with the eager backend; calls f3 on the input plus
            # 64 and adds 128.
            return f3(x + 64) + 128

        self.assertEqual(f4(torch.zeros(3)), torch.zeros(3) + 255)
        self.assertEqual(len(torch._dynamo.utils.counters["graph_break"]), 2)

    def test_supported_ctx_manager(self):
        # [FUNCTION DOC] Checks grad-mode context managers (no_grad and enable_grad) nested
        # across calls with a graph break-causing disabled function inside each; verifies
        # grad mode inside every call. Expects 4 frames and 14 ops (including grad-mode
        # switching ops).
        global check, check_disabled, f1, f2, f3

        @torch._dynamo.disable
        def check_disabled(value):
            # [FUNCTION DOC] HELPER: Disabled from dynamo; raises an assertion error unless the
            # global grad mode equals the given value.
            assert torch.is_grad_enabled() == value  # noqa: S101

        def check(value):
            # [FUNCTION DOC] HELPER: Raises an assertion error unless the global grad mode equals
            # the given value.
            assert torch.is_grad_enabled() == value  # noqa: S101

        def f1(x):
            # [FUNCTION DOC] HELPER: Inside no_grad adds one, checks grad is off several times
            # (once via the disabled function), and adds two.
            with torch.no_grad():
                x = x + 1
                check(False)
                check_disabled(False)
                check(False)
                return x + 2

        def f2(x):
            # [FUNCTION DOC] HELPER: Inside enable_grad adds four, checks grad is on, calls f1
            # and adds eight.
            with torch.enable_grad():
                x = x + 4
                check(True)
                check_disabled(True)
                check(True)
                return f1(x) + 8

        def f3(x):
            # [FUNCTION DOC] HELPER: Inside no_grad adds sixteen, checks grad is off, calls f2
            # and adds thirty-two.
            with torch.no_grad():
                x = x + 16
                check(False)
                check_disabled(False)
                check(False)
                return f2(x) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 4)
        # includes set_grad_enabled ops
        self.assertEqual(cnts.op_count, 14)

    def test_inactive_ctx_manager(self):
        # [FUNCTION DOC] Checks context manager objects that are created but not yet entered
        # survive nested graph breaks: some are created before a break, one as a stack value at
        # the break, and one passed as an argument. All are entered later and must see grad
        # disabled. Expects 3 frames and 7 ops.
        global check, f1, f2, f3

        def check(value):
            # [FUNCTION DOC] HELPER: Raises an assertion error unless the global grad mode equals
            # the given value.
            assert torch.is_grad_enabled() == value  # noqa: S101

        def f1(x, ctx1):
            # [FUNCTION DOC] HELPER: Creates no_grad managers around a graph break (one on
            # the evaluation stack), then enters the passed and created managers and checks
            # grad is off.
            x = x + 1
            ctx2 = torch.no_grad()
            # torch.no_grad() is a stack value at the time of graph break
            ctx3 = (torch.no_grad(), torch._dynamo.graph_break())[0]
            x = x + 64
            torch._dynamo.graph_break()
            with ctx1:
                check(False)
            with ctx2:
                check(False)
            with ctx3:
                check(False)
            return x + 2

        def f2(x, ctx1):
            # [FUNCTION DOC] HELPER: Creates a no_grad manager, calls f1 with another, then
            # enters both and checks grad is off.
            x = x + 4
            ctx2 = torch.no_grad()
            x = f1(x, torch.no_grad())
            with ctx1:
                check(False)
            with ctx2:
                check(False)
            return x + 8

        def f3(x):
            # [FUNCTION DOC] HELPER: Creates a no_grad manager, calls f2 with another, enters the
            # first and checks grad is off.
            x = x + 16
            ctx = torch.no_grad()
            x = f2(x, torch.no_grad())
            with ctx:
                check(False)
            return x + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 7)

    def test_ctx_manager_nested_step_graph_break(self):
        # [FUNCTION DOC] Checks a step-unsupported (skip rest of frame) event in a callee invoked
        # inside a no_grad block; expects 2 frames and 5 ops.
        global f1, f2

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one, calls step_unsupported, then adds two.
            x = x + 1
            torch._dynamo.step_unsupported()
            return x + 2

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds four, calls f1 inside no_grad, then adds eight.
            x = x + 4
            with torch.no_grad():
                x = f1(x)
            return x + 8

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f2)
        x = torch.zeros(3)
        res = f2(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 5)

    def test_ctx_manager_active_in_nested_call(self):
        # [FUNCTION DOC] Checks that a callee disabled from dynamo, called under no_grad inside
        # an enable_grad caller, sees grad disabled (the inner context wins). Expects 2 frames
        # and 8 ops.
        @torch._dynamo.disable()
        def f1(x):
            # [FUNCTION DOC] HELPER: Disabled from dynamo; asserts grad mode is off, then
            # adds one.
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Inside no_grad calls f1 on the input plus two and adds four.
            # order matters! should cancel out the enable_grad
            with torch.no_grad():
                return f1(x + 2) + 4

        def f3(x):
            # [FUNCTION DOC] HELPER: Inside enable_grad calls f2 on the input plus eight and
            # adds sixteen.
            with torch.enable_grad():
                return f2(x + 8) + 16

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 8)

    def test_ctx_manager_active_in_nested_step_graph_break(self):
        # [FUNCTION DOC] Checks a step-unsupported event in a callee while the caller holds an
        # active no_grad block; grad must remain disabled in the resumed code. Expects 2 frames
        # and 8 ops.
        def f1(x):
            # [FUNCTION DOC] HELPER: Calls step_unsupported, asserts grad mode is off, then
            # adds one.
            torch._dynamo.step_unsupported()
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 on the input plus two and adds four.
            return f1(x + 2) + 4

        def f3(x):
            # [FUNCTION DOC] HELPER: Inside no_grad calls f2 on the input plus eight and
            # adds sixteen.
            with torch.no_grad():
                return f2(x + 8) + 16

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 8)

    @torch._dynamo.config.patch(recompile_limit=1, fail_on_recompile_limit_hit=True)
    def test_no_recompiles(self):
        # [FUNCTION DOC] With a recompile limit of one and failure on hitting it, checks that two
        # levels of graph breaks do not cause recompilation. Checks result and 3 frames.
        global f1, f2, f3

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds four, calls f1, forces a graph break, then adds eight.
            x = x + 4
            x = f1(x)
            torch._dynamo.graph_break()
            return x + 8

        def f3(x):
            # [FUNCTION DOC] HELPER: Adds sixteen, calls f2 and adds thirty-two.
            x = x + 16
            return f2(x) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)

    def test_cells(self):
        # [FUNCTION DOC] Checks closures with cells shared across three nesting levels (nonlocal
        # modifications at each level) survive a graph break in the innermost function. Expects
        # equal results (including the returned cell values), 2 frames and 13 ops.
        def f1(x1):
            # [FUNCTION DOC] HELPER: Defines two cell variables and a nested function, calls it,
            # and returns its result tuple concatenated with the two final cell values.
            cell1 = x1 + 1
            cell2 = x1 + 2

            def f2(x2, x3):
                # [FUNCTION DOC] HELPER: Modifies the outer cell, creates its own cell, and
                # defines a further nested function; returns the nested result and its own cell.
                nonlocal cell1
                cell3 = x2 + x3 + 4
                cell1 += 8

                def f3(x4):
                    # [FUNCTION DOC] HELPER: Modifies two nonlocal cells, forces a graph break,
                    # and returns the input plus the three cell values.
                    nonlocal cell2, cell3
                    cell2 += 16
                    cell3 += 32
                    torch._dynamo.graph_break()
                    return x4 + cell1 + cell2 + cell3

                return f3(x2 + x3), cell3

            return f2(x1 + 64, x1 + 128) + (cell1, cell2)

        def outer(x):
            # [FUNCTION DOC] HELPER: Calls f1.
            return f1(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 13)

    def test_dead_nested_cells(self):
        # [FUNCTION DOC] Checks that a cell captured by an inner closure but dead in the outer
        # function does not add graph breaks when codegen runs for resume functions. Expects 2
        # frames and 6 ops.
        global f1, f2, f3

        def f3(x, cell1):
            # [FUNCTION DOC] HELPER: Increments the cell argument, adds it to the input, forces a
            # graph break, then adds it again.
            cell1 += 2
            x = x + cell1
            torch._dynamo.graph_break()
            return x + cell1

        def f1(cell1=0):
            # [FUNCTION DOC] HELPER: Returns a closure over its default-valued argument; the
            # closure adds four to the input, calls f3 with the cell and adds eight.
            def inner(x):
                # [FUNCTION DOC] HELPER: Adds four, calls f3 with the captured cell value,
                # adds eight.
                x += 4
                x = f3(x, cell1)
                return x + 8

            return inner

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls the closure returned from f1 on the input plus sixteen
            # and adds thirty-two.
            return f1()(x + 16) + 32

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f2)
        x = torch.zeros(3)
        res = f2(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # If we don't handle dead cells in nested functions correctly,
        # frame_count will increase since we also
        # graph break when we attempt to codegen inner.
        # The exact issue was that side_effects was failing to codegen inner's cell's creation.
        # So when we try to codegen cells for resume functions, we end up trying to codegen
        # a CellVariable without a source, which leads to a graph break we can't resume from.
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_cells_double_graph_break(self):
        # [FUNCTION DOC] Checks a function with a cell modified via nonlocal in a callee that has
        # two consecutive graph breaks. Expects 2 frames and 4 ops.
        def f1(x1):
            # [FUNCTION DOC] HELPER: Creates a cell and a nested function, calls it, and returns
            # its result along with the final cell value.
            cell1 = x1 + 1

            def f2(x2):
                # [FUNCTION DOC] HELPER: Increments the nonlocal cell, forces two graph breaks,
                # and returns the input plus the cell.
                nonlocal cell1
                cell1 += 2
                torch._dynamo.graph_break()
                torch._dynamo.graph_break()
                return x2 + cell1

            return f2(x1 + 4), cell1

        def outer(x):
            # [FUNCTION DOC] HELPER: Calls f1.
            return f1(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

    def test_side_effects_cells(self):
        # [FUNCTION DOC] Checks nonlocal mutation of closure cells (tensors modified in place)
        # before a graph break, comparing the returned value and the final tensors against an
        # eager run on freshly reset cells. Expects 2 frames and 5 ops.
        cell1, cell2, cell3, cell4 = (torch.zeros(3),) * 4

        def f1():
            # [FUNCTION DOC] HELPER: Increments a nonlocal tensor in place, forces a graph break,
            # and returns the sum of two cell tensors.
            nonlocal cell1
            cell1 += 1
            torch._dynamo.graph_break()
            return cell1 + cell2

        def f2():
            # [FUNCTION DOC] HELPER: Increments another nonlocal tensor in place by two and
            # returns f1 plus two cell tensors.
            nonlocal cell3
            cell3 += 2
            return f1() + cell3 + cell4

        def f3():
            # [FUNCTION DOC] HELPER: Returns f2.
            return f2()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)

        cell1 = torch.zeros(3)
        cell2 = torch.zeros(3) + 4
        cell3 = torch.zeros(3)
        cell4 = torch.zeros(3) + 8
        res = f3()
        res = (res,) + tuple(x.clone() for x in (cell1, cell2, cell3, cell4))

        cell1 = torch.zeros(3)
        cell2 = torch.zeros(3) + 4
        cell3 = torch.zeros(3)
        cell4 = torch.zeros(3) + 8
        ref = opt_fn()
        ref = (ref,) + tuple(x.clone() for x in (cell1, cell2, cell3, cell4))

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 5)

    def test_side_effects_globals(self):
        # [FUNCTION DOC] Same idea as the cell test but with module globals: mutates global
        # tensors across a graph break and compares the results and final global values between
        # eager and compiled runs after resetting them. Expects 2 frames and 6 ops.
        global f1, f2, f3
        global global1, global2, global3, global4

        def f1():
            # [FUNCTION DOC] HELPER: Increments a global tensor in place, forces a graph break,
            # returns the sum of two globals.
            global global1
            global1 += 1
            torch._dynamo.graph_break()
            return global1 + global2

        def f2():
            # [FUNCTION DOC] HELPER: Increments another global tensor by two and returns f1 plus
            # two globals.
            global global3
            global3 += 2
            return f1() + global3 + global4

        def f3(x):
            # [FUNCTION DOC] HELPER: Returns the input plus f2.
            return x + f2()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.ones(3)

        global1 = torch.zeros(3)
        global2 = torch.zeros(3) + 4
        global3 = torch.zeros(3)
        global4 = torch.zeros(3) + 8
        res = (f3(x), global1.clone(), global2, global3.clone(), global4)

        global1 = torch.zeros(3)
        global2 = torch.zeros(3) + 4
        global3 = torch.zeros(3)
        global4 = torch.zeros(3) + 8
        ref = (opt_fn(x), global1.clone(), global2, global3.clone(), global4)

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_side_effects_globals_different_module(self):
        # [FUNCTION DOC] Checks that a function from another module (the helper module) mutating
        # that module global state, calling back into a function with a graph break, behaves like
        # eager; state is reset before each run. Expects 2 frames and 7 ops.
        global f1, f2

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds one.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds one, calls the helper module function with f1 as
            # callback, and adds one.
            x = x + 1
            x = _test_nested_graph_breaks_helper.fn(x, f1)
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f2)

        _test_nested_graph_breaks_helper.reset_state()
        x = torch.zeros(3)
        res = (f2(x), _test_nested_graph_breaks_helper.global1.clone())

        _test_nested_graph_breaks_helper.reset_state()
        ref = (opt_fn(x), _test_nested_graph_breaks_helper.global1.clone())

        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 7)

    def test_nested_graph_break_in_loop(self):
        # [FUNCTION DOC] Checks a graph break that occurs only on one iteration (value five) of a
        # loop in a nested callee, using dynamic shapes to avoid recompiles. Expects 5 frames, 12
        # ops and 6 total frames compiled counter.
        global f1, f2, f3, f4, f5

        def f1(x, i):
            # [FUNCTION DOC] HELPER: Adds one, breaks the graph only when the index equals five,
            # and adds one.
            x = x + 1
            if i == 5:
                torch._dynamo.graph_break()
            return x + 1

        def f2(x, i):
            # [FUNCTION DOC] HELPER: Adds one, calls f1, and adds one.
            x = x + 1
            x = f1(x, i)
            return x + 1

        def f3(x):
            # [FUNCTION DOC] HELPER: Runs f2 for eight iterations with the loop index.
            for i in range(8):
                x = f2(x, i)
            return x

        def f4(x):
            # [FUNCTION DOC] HELPER: Adds one, calls f3, adds one.
            x = x + 1
            x = f3(x)
            return x + 1

        def f5(x):
            # [FUNCTION DOC] HELPER: Adds one, calls f4, adds one.
            x = x + 1
            x = f4(x)
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        # dynamic=True to prevent unnecessary recompiles
        opt_fn = torch._dynamo.optimize(backend=cnts, dynamic=True)(f5)
        x = torch.zeros(3)
        res = f5(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # skip frame due to nested graph break in for loop
        # 2 frames from f5+f4, 2 frames from f2+f1 (i == 5), 1 frame from f2+f1 (i != 5)
        self.assertEqual(cnts.frame_count, 5)
        # 4 additions from f5+f4, 2 x 4 additions from f2+f1 (i == 5, i != 5)
        self.assertEqual(cnts.op_count, 12)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], 6)

    def test_nested_graph_break_in_try_block(self):
        # [FUNCTION DOC] Checks a graph break inside a try block within a nested call chain (also
        # covers a nested step graph break). The frame holding the try is skipped from the break.
        # Expects 4 frames, 9 ops and 4 frames in the counter.
        # NOTE: this also tests nested step_graph_break
        global f1, f2, f3, f4, f5

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds one.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds one, calls f1, adds one.
            x = x + 1
            x = f1(x)
            return x + 1

        def f3(x):
            # [FUNCTION DOC] HELPER: Adds one, then inside a try with a finally block adds one,
            # calls f2 and adds one, then adds one after the block.
            x = x + 1
            try:
                x = x + 1
                x = f2(x)
                x = x + 1
            finally:
                pass
            return x + 1

        def f4(x):
            # [FUNCTION DOC] HELPER: Adds one, calls f3, adds one.
            x = x + 1
            x = f3(x)
            return x + 1

        def f5(x):
            # [FUNCTION DOC] HELPER: Adds one, calls f4, adds one.
            x = x + 1
            x = f4(x)
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f5)
        x = torch.zeros(3)
        res = f5(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # skip frame due to graph break in try block
        # 2 frames from f5+f4+(first part of f3), 2 frames from f2+f1
        self.assertEqual(cnts.frame_count, 4)
        # 5 additions from f5+f4+(first part of f3), 4 additions from f2+f1
        self.assertEqual(cnts.op_count, 9)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], 4)

    def test_nested_step_unsupported(self):
        # [FUNCTION DOC] Checks a step-unsupported event in a middle function: the start of the
        # outer pair, the callee f1 and the end of the outer function should become separate
        # frames. Expects 3 frames, 4 ops and 3 in the frames counter.
        global f1, f2, f3

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one.
            return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Adds two, calls step_unsupported, calls f1 and adds four.
            x = x + 2
            torch._dynamo.step_unsupported()
            return f1(x) + 4

        def f3(x):
            # [FUNCTION DOC] HELPER: Adds eight, calls f2 and adds sixteen.
            x = x + 8
            return f2(x) + 16

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # 1 frame from start of f3 + start of f2, 1 frame from f1, 1 frame from the end of f3
        self.assertEqual(cnts.frame_count, 3)
        # all ops except + 4
        self.assertEqual(cnts.op_count, 4)
        self.assertEqual(torch._dynamo.utils.counters["frames"]["total"], 3)

    def test_nested_step_graph_break_diff_args(self):
        # [FUNCTION DOC] Checks a step graph break in a callee with two arguments called while a
        # user object is alive in the caller. Expects 2 frames and 3 ops.
        global inner, outer

        def inner(x1, x2):
            # [FUNCTION DOC] HELPER: Calls step_unsupported then returns the sum of the
            # two inputs.
            torch._dynamo.step_unsupported()
            return x1 + x2

        class Foo:
            # [CLASS DOC] HELPER: Simple object with a single attribute.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the attribute to one.
                self.attr = 1

        def outer(x):
            # [FUNCTION DOC] HELPER: Creates a Foo object, calls inner with two shifted inputs,
            # and adds the Foo attribute.
            z = Foo()
            y = inner(x + 1, x + 2)
            y = y + z.attr
            return y

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 3)

    def test_generator_nested_graph_break(self):
        # [FUNCTION DOC] Checks that a generator with a graph break inside is not given nested
        # graph breaks: the caller is skipped (0 frames) and results match. Then compiles an
        # outer function calling it and checks 2 frames and 2 ops.
        def gen(x):
            # [FUNCTION DOC] HELPER: Generator that yields the input plus one, forces a graph
            # break, then yields the input plus two.
            yield x + 1
            torch._dynamo.graph_break()
            yield x + 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds four and returns the list of generator values.
            x = x + 4
            return list(gen(x))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        res = fn(x)
        # NOTE: if we enable nested graph breaks on inlined generators, we expect
        # some sort of internal dynamo failure
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # fn should be skipped
        self.assertEqual(cnts.frame_count, 0)

        def outer(x):
            # [FUNCTION DOC] HELPER: Adds eight, calls fn and returns the first element
            # plus sixteen.
            x = x + 8
            return fn(x)[0] + 16

        cnts.clear()
        torch.compiler.reset()

        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # only outer should be traced
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_return_after_graph_break_nested(self):
        # [FUNCTION DOC] Checks that returning right after a nested graph break does not skip the
        # rest of the top-level function. Tries a regular break, a step-unsupported break, a
        # break on attribute store created via manually assembled bytecode (replacing the code of
        # a helper), and a chain where every function returns. Verifies numerical results and the
        # graph_break counter.
        # With improper implementation, returning immediately after a nested graph
        # break may skip the rest of the top-level frame.
        def f2(inner, x):
            # [FUNCTION DOC] HELPER: Adds two to the input in place and returns the result of the
            # passed callable.
            x += 2
            return inner(x)

        @torch.compile(backend="eager")
        def f3(inner, x):
            # [FUNCTION DOC] HELPER: Compiled with the eager backend; calls f2, adds four, and
            # adds the result if it is not None.
            result = f2(inner, x)
            x += 4
            if result is not None:
                x += result
            return x

        # test normal graph break
        x = torch.zeros(3)

        def inner1(x):
            # [FUNCTION DOC] HELPER: Adds one in place and returns the result of graph_break.
            x += 1
            return torch._dynamo.graph_break()

        ref = f3(inner1, x)
        self.assertEqual(ref, torch.zeros(3) + 7)

        # test step graph break
        x = torch.zeros(3)

        def inner2(x):
            # [FUNCTION DOC] HELPER: Adds one in place and returns the result
            # of step_unsupported.
            x += 1
            return torch._dynamo.step_unsupported()

        ref = f3(inner2, x)
        self.assertEqual(ref, torch.zeros(3) + 7)

        # test store attr graph break
        # NOTE: we do this manual bytecode generation hack since the only RETURN_*
        # instruction that can follow STORE_ATTR is RETURN_CONST, which was removed in 3.14+.

        # make sure inner3's code options are compatible with the instructions below
        global y

        # y must cause a graph break on STORE_ATTR. A torch-internal class
        # (UDCV with __module__ starting with "torch.") declines mutation.
        class y:
            # [CLASS DOC] HELPER: Global class marked with a torch module name so dynamo declines
            # mutating its attributes.
            __module__ = "torch._dynamo._test_marker"

        def inner3(x):
            # [FUNCTION DOC] HELPER: Placeholder whose code is replaced later by assembled
            # bytecode that sets attributes on the argument and the global class and returns.
            x.attr = 1000
            y.attr = 2000

        new_inst = torch._dynamo.bytecode_transformation.create_instruction
        insts = [
            new_inst("LOAD_CONST", argval=1000),
            new_inst("LOAD_CONST", argval=2000),
            new_inst("LOAD_GLOBAL", argval="y"),
            # NOTE: this should cause a graph break - change y if it doesn't work!
            new_inst("STORE_ATTR", argval="attr"),
            new_inst("RETURN_VALUE"),
        ]
        if sys.version_info >= (3, 11):
            insts = [new_inst("RESUME", arg=0)] + insts
        code_keys = torch._dynamo.bytecode_transformation.get_code_keys()
        code_options = {k: getattr(inner3.__code__, k) for k in code_keys}
        _, inner3_code = (
            torch._dynamo.bytecode_transformation.clean_and_assemble_instructions(
                insts, code_keys, code_options
            )
        )
        inner3.__code__ = inner3_code

        torch._dynamo.utils.counters.clear()
        x = torch.zeros(3)
        ref = f3(inner3, x)
        self.assertEqual(ref, torch.zeros(3) + 1006)
        # make sure we're actually STORE_ATTR graph breaking
        self.assertEqual(len(torch._dynamo.utils.counters["graph_break"]), 1)

        # dynamic branching is harder to test - the other tests should be enough cover

        # test every function returning
        @torch.compiler.disable
        def inner5(x):
            # [FUNCTION DOC] HELPER: Disabled from dynamo; adds eight in place and returns it.
            x += 8
            return x

        def inner4(x):
            # [FUNCTION DOC] HELPER: Adds one in place and returns the result of inner5.
            x += 1
            return inner5(x)

        @torch.compile(backend="eager")
        def f4(x):
            # [FUNCTION DOC] HELPER: Compiled with the eager backend; adds four and returns f2
            # called with inner4.
            x += 4
            return f2(inner4, x)

        x = torch.zeros(3)
        ref = f4(x)
        self.assertEqual(ref, torch.zeros(3) + 15)

    def test_return_after_graph_break_deep_nested(self):
        # [FUNCTION DOC] Checks that in a deep chain of eight functions with one disabled
        # function at the bottom, only two resume functions are created (for f4 and f7), via the
        # resumes counter and the numeric result.
        @torch.compiler.disable
        def f1(x):
            # [FUNCTION DOC] HELPER: Disabled from dynamo; adds one.
            return x + 1

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 on the input plus two.
            return f1(x + 2)

        def f3(x):
            # [FUNCTION DOC] HELPER: Calls f2 on the input plus four.
            return f2(x + 4)

        def f4(x):
            # [FUNCTION DOC] HELPER: Calls f3 on the input plus eight and adds sixteen.
            x = f3(x + 8)
            return x + 16

        def f5(x):
            # [FUNCTION DOC] HELPER: Calls f4 on the input plus 32.
            return f4(x + 32)

        def f6(x):
            # [FUNCTION DOC] HELPER: Calls f5 on the input plus 64.
            return f5(x + 64)

        def f7(x):
            # [FUNCTION DOC] HELPER: Calls f6 on the input plus 128 and adds 256.
            x = f6(x + 128)
            return x + 256

        @torch.compile(backend="eager")
        def f8(x):
            # [FUNCTION DOC] HELPER: Compiled with the eager backend; calls f7 on the input
            # plus 512.
            return f7(x + 512)

        x = torch.zeros(3)
        ref = f8(x)
        self.assertEqual(ref, torch.zeros(3) + 1023)

        # check that only 2 resume functions are created
        self.assertEqual(len(torch._dynamo.utils.counters["resumes"]), 2)
        for name in ("resume_in_f4", "resume_in_f7"):
            self.assertTrue(
                any(name in key for key in torch._dynamo.utils.counters["resumes"])
            )

    def test_disable_nested_graph_breaks(self):
        # [FUNCTION DOC] Checks the decorator that disables nested graph breaks for one function
        # in a chain: with f3 decorated, the frames are split as if there were no nested breaks
        # inside it. Expects 8 frames and 10 ops.
        global f1, f2, f3, f4, f5

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 on the input plus four and adds eight.
            return f1(x + 4) + 8

        # NOTE since the disable_nested_graph_breaks decorator is implemented as a
        # context manager, we don't need to separately test context manager usage.
        @torch._dynamo.disable_nested_graph_breaks
        def f3(x):
            # [FUNCTION DOC] HELPER: Decorated to disable nested graph breaks; calls f2 on the
            # input plus sixteen and adds thirty-two.
            return f2(x + 16) + 32

        def f4(x):
            # [FUNCTION DOC] HELPER: Calls f3 on the input plus 64 and adds 128.
            return f3(x + 64) + 128

        def f5(x):
            # [FUNCTION DOC] HELPER: Calls f4 on the input plus 256 and adds 512.
            return f4(x + 256) + 512

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f5)
        x = torch.zeros(3)
        res = f5(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # 2 frames from each of f5+f4, f3, f2, f1
        self.assertEqual(cnts.frame_count, 8)
        self.assertEqual(cnts.op_count, 10)

    def test_disable_nested_graph_breaks_context_manager(self):
        # [FUNCTION DOC] Same as the decorator test but uses the disable function as a context
        # manager around the call inside f3. Expects 6 frames.
        """disable_nested_graph_breaks as a context manager."""
        global f1, f2, f3

        def f1(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        def f2(x):
            # [FUNCTION DOC] HELPER: Calls f1 on the input plus four and adds eight.
            return f1(x + 4) + 8

        def f3(x):
            # [FUNCTION DOC] HELPER: Calls f2 inside the disable-nested-graph-breaks context and
            # then adds sixty-four.
            with torch._dynamo.disable_nested_graph_breaks():
                x = f2(x + 16) + 32
            return x + 64

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f3)
        x = torch.zeros(3)
        res = f3(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 6)

    def test_recursive_compiled_fn_with_graph_break(self):
        # [FUNCTION DOC] Checks a recursive compiled function does not treat its own self-call as
        # a nested compile (which would disable nested graph breaks) by comparing eager and
        # compiled results.
        """Recursive compiled functions should not disable NGB on self-calls.

        When a compiled function recursively calls itself, the inner call
        should NOT be treated as a nested torch.compile (which would disable
        NGB). Only calls to a *different* torch.compile wrapper should.
        """

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Recursive: returns the input if the counter is non-
            # positive; otherwise forces a graph break and recurses with input plus one and
            # counter minus one.
            if n <= 0:
                return x
            torch._dynamo.graph_break()
            return fn(x + 1, n - 1)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        ref = fn(x, 3)
        res = opt_fn(x, 3)
        self.assertEqual(ref, res)

    def test_nested_store_attr_graph_break(self):
        # [FUNCTION DOC] Checks a graph break inside a user-defined attribute-setting method
        # triggered by an attribute store in a compiled function; the method must be traced in
        # the resumed frame. Expects the value stored to be plus 3, 2 frames and 3 ops.
        class Foo:
            # [CLASS DOC] HELPER: Object whose attribute setter forces a graph break, raises
            # RuntimeError if not being compiled, and stores the value plus one.
            def __setattr__(self, name, value):
                # [FUNCTION DOC] HELPER: Forces a graph break; raises RuntimeError if
                # not compiling; then sets the attribute to the value plus one via
                # the parent implementation.
                torch._dynamo.graph_break()
                if not torch.compiler.is_compiling():
                    raise RuntimeError("Expected this to be traced")
                super().__setattr__(name, value + 1)

        def fn(foo, x):
            # [FUNCTION DOC] HELPER: Stores the input plus two on the object attribute and
            # returns the input plus four.
            foo.attr = x + 2
            return x + 4

        foo = Foo()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        ref = opt_fn(foo, x)
        self.assertEqual(ref, x + 4)
        self.assertEqual(foo.attr, x + 3)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 3)

    def test_nested_store_subscr_graph_break(self):
        # [FUNCTION DOC] Like the store-attribute test but through the item-set operator on the
        # object. Expects the stored attribute to be input plus 3, 2 frames and 3 ops.
        class Foo:
            # [CLASS DOC] HELPER: Object whose item setter forces a graph break, raises
            # RuntimeError if not compiling, and stores the value plus one as an attribute.
            def __setitem__(self, name, value):
                # [FUNCTION DOC] HELPER: Forces a graph break; raises RuntimeError if not
                # compiling; then sets an attribute to the value plus one.
                torch._dynamo.graph_break()
                if not torch.compiler.is_compiling():
                    raise RuntimeError("Expected this to be traced")
                super().__setattr__(name, value + 1)

        def fn(foo, x):
            # [FUNCTION DOC] HELPER: Sets an item on the object to the input plus two and returns
            # the input plus four.
            foo["attr"] = x + 2
            return x + 4

        foo = Foo()

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(fn)
        x = torch.zeros(3)
        ref = opt_fn(foo, x)
        self.assertEqual(ref, x + 4)
        self.assertEqual(foo.attr, x + 3)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 3)

    def test_functorch_with_nested_graph_break(self):
        # [FUNCTION DOC] Checks a nested graph break under a functorch gradient transform using
        # an aot_eager counting backend, with full-graph disabled; compares results with eager
        # and checks one graph break, 2 frames and 4 ops.
        def f1(x):
            # [FUNCTION DOC] HELPER: Multiplies by two, forces a graph break, multiplies by four.
            x = x * 2
            torch._dynamo.graph_break()
            return x * 4

        def f2(x):
            # [FUNCTION DOC] HELPER: Returns the sum of f1 applied to the input times eight,
            # times sixteen.
            return (f1(x * 8) * 16).sum()

        def f3(x):
            # [FUNCTION DOC] HELPER: Returns the gradient of f2 at the input times 32, times 64.
            return torch.func.grad(f2)(x * 32) * 64

        def f4(x):
            # [FUNCTION DOC] HELPER: Returns f3 of the input times 128, times 256.
            return f3(x * 128) * 256

        cnts = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        x = torch.randn(3)
        actual = f4(x)
        expected = torch.compile(f4, backend=cnts, fullgraph=False)(x)
        self.assertEqual(actual, expected)
        self.assertEqual(len(torch._dynamo.utils.counters["graph_break"]), 1)
        # f4 + f3, f3 end + f4 end
        self.assertEqual(cnts.frame_count, 2)
        # multiplication by 32, 64, 128, 256
        self.assertEqual(cnts.op_count, 4)

    def test_resume_closure_different_module_globals(self):
        # [FUNCTION DOC] Checks resume functions for a closure from another module use that
        # module globals; compares eager and compiled results and 2 frames.
        # Tests that resume functions with freevars (closures) from inlined
        # frames get the correct f_globals. Without the factory fix,
        # MAKE_FUNCTION inherits the root frame's globals, so the resume
        # function for `inner` would not find HELPER_CONSTANT.
        def outer(x):
            # [FUNCTION DOC] HELPER: Calls a helper module function with a closure that breaks
            # the graph and adds two.
            return _test_nested_graph_breaks_helper.closure_with_graph_break(x) + 2

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(outer)
        x = torch.zeros(3)
        res = outer(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

    def test_error_on_graph_break_nested(self):
        # [FUNCTION DOC] Checks the error-on-graph-break setting in nested frames: a decorated
        # inner function turned off inside a function with it on is allowed (2 frames), as is a
        # context manager turning it off; but turning it on inside a frame with it off, or on in
        # the outer frame, raises Unsupported.
        # error_on_graph_break in a nested frame
        cnts = torch._dynamo.testing.CompileCounter()

        @torch._dynamo.error_on_graph_break(False)
        def inner_f5(x):
            # [FUNCTION DOC] HELPER: Error-on-break turned off; adds two, forces a graph break,
            # adds four.
            x = x + 2
            torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f5(x):
            # [FUNCTION DOC] HELPER: Compiled with error-on-break on; adds one and
            # calls inner_f5.
            x = x + 1
            return inner_f5(x)

        inp = torch.ones(3)
        self.assertEqual(f5(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner_f6(x):
            # [FUNCTION DOC] HELPER: Adds two, forces a graph break inside a context turning
            # error-on-break off, adds four.
            x = x + 2
            with torch._dynamo.error_on_graph_break(False):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f6(x):
            # [FUNCTION DOC] HELPER: Compiled with error-on-break on; adds one and
            # calls inner_f6.
            x = x + 1
            return inner_f6(x)

        cnts.clear()
        self.assertEqual(f6(inp), inp + 7)
        self.assertEqual(cnts.frame_count, 2)

        def inner_f7(x):
            # [FUNCTION DOC] HELPER: Adds two, forces a graph break inside a context turning
            # error-on-break on, adds four.
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(False)
        @torch.compile(backend=cnts)
        def f7(x):
            # [FUNCTION DOC] HELPER: Compiled with error-on-break off; adds one and
            # calls inner_f7.
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f7(inp)

        def inner_f8(x):
            # [FUNCTION DOC] HELPER: Adds two, forces a graph break inside a context turning
            # error-on-break on, adds four. Defined but not called.
            x = x + 2
            with torch._dynamo.error_on_graph_break(True):
                torch._dynamo.graph_break()
            return x + 4

        @torch._dynamo.error_on_graph_break(True)
        @torch.compile(backend=cnts)
        def f8(x):
            # [FUNCTION DOC] HELPER: Compiled with error-on-break on; adds one and calls inner_f7
            # (not inner_f8).
            # KNOWN ISSUE: calls inner_f7 rather than the inner_f8 defined just above, so
            # inner_f8 is unused.
            x = x + 1
            return inner_f7(x)

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            f8(inp)

    def test_graph_break_in_wrapped_user_function(self):
        # [FUNCTION DOC] Checks a graph break inside a user function wrapped with no_grad and
        # called from a compiled function: the resumed code must be compiled with grad disabled
        # inside the callee and enabled again afterwards. Expects 2 frames.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, forces a graph break, asserts compiling and grad
            # disabled, adds two.
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 2

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls the wrapped function, then asserts compiling
            # and grad enabled.
            x = torch.no_grad()(fn)(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

    def test_graph_break_in_nested_wrapped_user_function(self):
        # [FUNCTION DOC] Same as the wrapped function test but with a chain of nested grad-mode
        # decorators (no_grad outermost, innermost no_grad), so the effective mode inside is
        # disabled. Expects 2 frames.
        # no_grad wraps enable_grad wraps no_grad wraps fn
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, forces a graph break, asserts compiling and grad
            # disabled, adds two.
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 2

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; builds the nested wrapped function, calls it,
            # asserts compiling and grad enabled.
            wrapped = torch.no_grad()(torch.enable_grad()(torch.no_grad()(fn)))
            x = wrapped(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

    def test_graph_break_in_wrapped_user_function_with_multiple_breaks(self):
        # [FUNCTION DOC] Like the wrapped function test but the callee has two graph breaks.
        # Expects the result plus 6 and 3 frames.
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, breaks, checks state, adds two, breaks, checks
            # state, adds three.
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            x = x + 2
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 3

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls the no_grad-wrapped callee and asserts
            # compiling and grad enabled afterwards.
            x = torch.no_grad()(fn)(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 6)
        self.assertEqual(cnts.frame_count, 3)

    def test_graph_break_in_sequential_wrapped_user_functions(self):
        # [FUNCTION DOC] Calls two no_grad-wrapped functions in sequence, each with a graph
        # break, and checks grad state is restored in between. Expects 3 frames and result
        # plus 10.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Adds one, breaks, checks state, adds two.
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 2

        def fn2(x):
            # [FUNCTION DOC] HELPER: Adds three, breaks, checks state, adds four.
            x = x + 3
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 4

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls both wrapped callees in turn, checking grad
            # is enabled after each.
            x = torch.no_grad()(fn1)(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            x = torch.no_grad()(fn2)(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 10)
        # 3 frames: before fn1's break, after fn1's break through fn2's break,
        # after fn2's break to end
        self.assertEqual(cnts.frame_count, 3)

    def test_graph_break_in_wrapped_enable_grad(self):
        # [FUNCTION DOC] Checks a no_grad-decorated function that internally uses an enable_grad-
        # wrapped lambda and then breaks; grad should be disabled after the lambda and results
        # equal input plus 6.
        @torch.no_grad()
        def fn(x):
            # [FUNCTION DOC] HELPER: Decorated with no_grad; adds one, calls an enable_grad-
            # wrapped lambda adding two, breaks, asserts compiling and grad disabled, adds three.
            x = x + 1
            # enable_grad inside no_grad
            x = torch.enable_grad()(lambda y: y + 2)(x)
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 3

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls fn.
            return fn(x)

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 6)

    def test_graph_break_in_wrapped_user_method(self):
        # [FUNCTION DOC] Checks a graph break in a method that is wrapped with no_grad and stored
        # back on the instance inside the compiled function. Expects result plus 3 and 2 frames.
        class Foo:
            # [CLASS DOC] HELPER: Object with two integer attributes and a method that adds one
            # attribute, breaks, checks compile and grad state, and adds the other.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the two attributes to one and two.
                self.a = 1
                self.b = 2

            def fn(self, x):
                # [FUNCTION DOC] HELPER: Adds the first attribute, forces a graph break, asserts
                # compiling and grad disabled, adds the second attribute.
                x = x + self.a
                torch._dynamo.graph_break()
                assert torch.compiler.is_compiling()  # noqa: S101
                assert not torch.is_grad_enabled()  # noqa: S101
                return x + self.b

        obj = Foo()

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; replaces the method on the instance with a
            # no_grad-wrapped version, calls it, asserts compiling and grad enabled.
            obj.fn = torch.no_grad()(obj.fn)
            x = obj.fn(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

    def test_graph_break_in_wrapped_nested_function(self):
        # [FUNCTION DOC] Checks a no_grad-decorated nested function defined with closure
        # variables inside a compiled function, containing a graph break. Expects result plus 3
        # and 2 frames.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; defines closure variables and a decorated nested
            # function, calls it, asserts compiling and grad enabled.
            a = 1
            b = 2

            @torch.no_grad()
            def fn(x):
                # [FUNCTION DOC] HELPER: Decorated with no_grad; adds the captured a, breaks,
                # checks state, adds the captured b.
                x = x + a
                torch._dynamo.graph_break()
                assert torch.compiler.is_compiling()  # noqa: S101
                assert not torch.is_grad_enabled()  # noqa: S101
                return x + b

            x = fn(x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

    def test_graph_break_in_wrapped_skipped_function(self):
        # [FUNCTION DOC] Checks a no_grad-wrapped skipped helper function (from the testing
        # module) that calls back into a function containing a graph break. Expects result plus 4
        # and 3 frames.
        from torch._dynamo.testing import _skipped_function_for_test_reconstruct

        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, forces a graph break, asserts compiling and grad
            # disabled, adds two.
            x = x + 1
            torch._dynamo.graph_break()
            assert torch.compiler.is_compiling()  # noqa: S101
            assert not torch.is_grad_enabled()  # noqa: S101
            return x + 2

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def gn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls the wrapped skipped helper with fn and the
            # input, asserts state, adds one.
            x = torch.no_grad()(_skipped_function_for_test_reconstruct)(fn, x)
            assert torch.compiler.is_compiling()  # noqa: S101
            assert torch.is_grad_enabled()  # noqa: S101
            return x + 1

        inp = torch.randn(3)
        self.assertEqual(gn(inp), inp + 4)
        self.assertEqual(cnts.frame_count, 3)

    def test_step_graph_break_frame_values_not_corrupted(self):
        # [FUNCTION DOC] Regression test: a step-unsupported callee called while the caller
        # operand stack is non-empty (building a tuple) must not corrupt the caller locals.
        # Checks results and 1 frame and 3 ops.
        """Bytecode generation bug in step_graph_break corrupted parent frame
        locals when the parent had a non-empty operand stack (num_stack > 0).
        """

        def inner(x):
            # [FUNCTION DOC] HELPER: Adds one twice, calls step_unsupported and returns
            # the value.
            x = x + 1
            x = x + 1
            torch._dynamo.step_unsupported()
            return x

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; adds one, builds a tuple with the input and inner
            # result, returns both.
            x = x + 1
            y = (x, inner(x))
            return x, y

        x = torch.tensor([1.0, 2.0])
        result = fn(x)
        self.assertEqual(result[0], torch.tensor([2.0, 3.0]))
        self.assertEqual(
            result[1], (torch.tensor([2.0, 3.0]), torch.tensor([4.0, 5.0]))
        )
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 3)

    def test_contextmanager_graph_break_in_init(self):
        # [FUNCTION DOC] Checks a graph break during construction of a generator-based context
        # manager whose generator function is excluded from dynamo (DDP pattern). Expects result
        # plus 6 and 2 frames.
        """Graph break in _GeneratorContextManager.__init__ when the generator
        function is @torch._disable_dynamo (the DDP pattern)."""
        from contextlib import contextmanager

        @contextmanager
        @torch._disable_dynamo(recursive=False)
        def my_ctx():
            # [FUNCTION DOC] HELPER: Generator context manager body that simply yields; the
            # underlying function is excluded from dynamo.
            yield

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; adds one, adds two inside the context, adds three.
            x = x + 1
            with my_ctx():
                x = x + 2
            return x + 3

        inp = torch.randn(3)
        self.assertEqual(fn(inp), inp + 6)
        # `x + 1` compiles before the break in the ctx manager construction;
        # the resume reconstructs the ctx manager (now enterable, see
        # VariableBuilder.wrap_user_defined) and compiles the with-body `x + 2`
        # plus `x + 3`, giving a second frame.
        self.assertEqual(cnts.frame_count, 2)

    def test_graph_break_during_module_init(self):
        # [FUNCTION DOC] Checks a graph break inside a module constructor does not prevent
        # compiling the rest of the caller; constructs the module and calls it, expecting output
        # shape (10) and 1 frame and 1 op.
        """Graph break during nn.Module.__init__ should not prevent the caller
        from compiling the rest of the function."""

        class MyModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose constructor forces a graph break before creating a
            # linear layer; forward applies the layer.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Calls the parent constructor, forces a graph break,
                # creates a Linear layer of ten by ten.
                super().__init__()
                torch._dynamo.graph_break()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer.
                return self.linear(x)

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled; builds the module and applies it to the input.
            m = MyModule()
            return m(x)

        x = torch.randn(10)
        result = f(x)
        self.assertEqual(result.shape, torch.Size([10]))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_graph_break_during_user_class_init(self):
        # [FUNCTION DOC] Checks a graph break in a user class constructor called from compiled
        # code does not use nested graph breaks (which would leave a half-initialized object).
        # Expects the correct product, 1 frame and 1 op.
        """Graph break during user class __init__ via instantiate_user_defined_class_object.

        The polyfill does cls.__new__ then obj.__init__. If nested graph breaks
        were allowed in the polyfill, the resume would have a half-initialized
        object (created by __new__ but __init__ not yet complete), and accessing
        attributes set after the graph break would fail with AttributeError.
        """

        class Config:
            # [CLASS DOC] HELPER: Class whose constructor forces a graph break and then stores a
            # scale attribute.
            def __init__(self, scale):
                # [FUNCTION DOC] HELPER: Forces a graph break then stores the scale.
                torch._dynamo.graph_break()
                self.scale = scale

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled; creates the object and scales the input.
            cfg = Config(2)
            return x * cfg.scale

        x = torch.randn(10)
        result = f(x)
        self.assertEqual(result, x * 2)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_graph_break_during_user_class_new(self):
        # [FUNCTION DOC] Checks a graph break in the allocation method of a user class does not
        # use nested graph breaks. Expects the sum with the value 3, 1 frame and 1 op.
        """Graph break during __new__ via instantiate_user_defined_class_object.

        The polyfill calls cls.__new__ first. If nested graph breaks were
        allowed, the resume would not yet have the object on the stack at all
        (since __new__ creates it), so the subsequent __init__ call and any
        attribute access would operate on garbage.
        """

        class Wrapper:
            # [CLASS DOC] HELPER: Class whose allocation method forces a graph break, creates the
            # instance via the parent and stores a value attribute.
            def __new__(cls, val):
                # [FUNCTION DOC] HELPER: Forces a graph break, creates an instance via the
                # parent, sets the value attribute and returns it.
                torch._dynamo.graph_break()
                obj = super().__new__(cls)
                obj.val = val
                return obj

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled; creates the object and adds its value to
            # the input.
            w = Wrapper(3)
            return x + w.val

        x = torch.randn(10)
        result = f(x)
        self.assertEqual(result, x + 3)
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_graph_break_in_copy_deepcopy(self):
        # [FUNCTION DOC] Checks deepcopy of a Linear module inside compiled code does not use
        # nested graph breaks (loops and breaks inside the copy module). Expects shape (10), 2
        # frames and 2 ops.
        """copy.deepcopy is a BUILTIN_INLINE_WHEN_CALLED function that contains
        loops and graph-breaks internally. If nested graph breaks were allowed,
        deepcopy's internal loop would trigger raise_loop_graph_break which
        poisons the parent frame, and deepcopy's graph-breaks would produce
        unusable resume functions for stdlib frames."""
        import copy

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def f(x):
            # [FUNCTION DOC] HELPER: Compiled; adds one, deep-copies a new Linear layer and
            # applies the copy to the result.
            y = x + 1
            m = torch.nn.Linear(10, 10)
            m2 = copy.deepcopy(m)
            return m2(y)

        x = torch.randn(10)
        result = f(x)
        self.assertEqual(result.shape, torch.Size([10]))
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 2)

    def test_inlined_function_globals_across_graph_break(self):
        # [FUNCTION DOC] Checks resume functions for an inlined function from another module use
        # that module globals. Expects result plus 3 and 2 frames.
        """Module-level globals in inlined functions survive graph breaks.

        When a function from another module is inlined and hits a graph break,
        the resume function must use that module's globals (not the caller's).
        This is handled by install_resume_function_global().
        """
        fn_with_module_global = _test_nested_graph_breaks_helper.fn_with_module_global

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls the helper module function that contains a
            # graph break.
            return fn_with_module_global(x)

        inp = torch.randn(3)
        self.assertEqual(fn(inp), inp + 3)
        self.assertEqual(cnts.frame_count, 2)

    @unittest.expectedFailure
    def test_nested_decorated_function(self):
        # [FUNCTION DOC] Expected failure: a nested function decorated with an autocast context
        # containing a graph break; the comparison should hold with 2 frames and 6 ops.
        # KNOWN ISSUE: the test prints both results (a leftover debug print) and is
        # marked expectedFailure.
        # decorator must call ContextWrappingVariable.cleanup_assert to trigger this test
        def f(x):
            # [FUNCTION DOC] HELPER: Defines a nested function decorated with autocast and
            # calls it.
            @torch.autocast("cpu")
            def inner(y):
                # [FUNCTION DOC] HELPER: Adds one, forces a graph break, adds one.
                y = y + 1
                torch._dynamo.graph_break()
                return y + 1

            return inner(x)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f)
        x = torch.zeros(3)
        res = f(x)
        ref = opt_fn(x)
        print(ref, res)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 6)

    def test_nested_graph_break_in_custom_ctx_manager_init(self):
        # [FUNCTION DOC] Checks a graph break inside the constructor of a user context manager,
        # before it is entered: the pre-break graph is empty and only the resume is compiled.
        # Expects 1 frame and 1 op.
        def f(x):
            # [FUNCTION DOC] HELPER: Enters the custom context manager and returns the input
            # plus one.
            with CustomizedCtxManager(x):
                return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch._dynamo.optimize(backend=cnts)(f)
        x = torch.zeros(3)
        res = f(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)
        # The graph break happens inside __init__, i.e. before the context
        # manager is entered, so the pre-break graph is empty (not compiled).
        # Only the resume (enter + `x + 1` + exit) is compiled: 1 frame, 1 op.
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_ctx_manager_passed_as_sourced_arg(self):
        # [FUNCTION DOC] Checks a user context manager passed as an argument works like eager
        # both when entered and when only used as a plain object (attribute access). Compiles
        # each with fullgraph and expects 1 frame.
        # A ctx manager passed in as an argument is sourced, so it is wrapped
        # by VariableBuilder.wrap_user_defined -- which now produces a
        # GenericContextWrappingVariable for any type with __enter__/__exit__.
        # It must behave like eager both when entered in a `with` and when
        # merely used as a plain object (attribute access, never entered).
        class Ctx:
            # [CLASS DOC] HELPER: Context manager with one attribute; enter returns self and exit
            # returns false.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the argument as an attribute.
                self.x = x

            def __enter__(self):
                # [FUNCTION DOC] HELPER: Returns self.
                return self

            def __exit__(self, *args):
                # [FUNCTION DOC] HELPER: Returns false so exceptions propagate.
                return False

        def f_with(cm, x):
            # [FUNCTION DOC] HELPER: Enters the manager and returns the input plus its attribute.
            with cm:
                return x + cm.x

        def f_plain(cm, x):
            # [FUNCTION DOC] HELPER: Returns the input plus the manager attribute without
            # entering it.
            return x + cm.x

        cm = Ctx(torch.ones(3))
        x = torch.zeros(3)
        for fn in (f_with, f_plain):
            cnts = torch._dynamo.testing.CompileCounter()
            res = torch.compile(fn, backend=cnts, fullgraph=True)(cm, x)
            self.assertEqual(res, fn(cm, x))
            self.assertEqual(cnts.frame_count, 1)

    def test_ctx_manager_c_level_enter_graph_breaks(self):
        # [FUNCTION DOC] Checks context managers whose enter and exit are not plain bound
        # functions (thread locks, static-method dunders) graph-break to eager cleanly: expects
        # correct result and 0 frames for passed ones, and a correct result for one constructed
        # inside the trace.
        # A sourced context manager whose __enter__/__exit__ are not plain
        # bound-method functions (no `.__func__`) must cleanly graph-break to
        # eager when entered, NOT be wrapped as a GenericContextWrappingVariable
        # (whose enter/exit take `.__func__`) which would raise
        # InternalTorchDynamoError. Covers C-level builtins (threading locks)
        # and a staticmethod-defined dunder.
        import threading

        class StaticCtx:
            # [CLASS DOC] HELPER: Context manager whose enter and exit are static methods.
            @staticmethod
            def __enter__():
                # [FUNCTION DOC] HELPER: Static method; returns None.
                return None

            @staticmethod
            def __exit__(*args):
                # [FUNCTION DOC] HELPER: Static method; returns false.
                return False

        def f(cm, x):
            # [FUNCTION DOC] HELPER: Enters the passed manager and returns the input plus one.
            with cm:
                return x + 1

        x = torch.ones(3)
        # sourced (VariableBuilder.wrap_user_defined path)
        for cm in (threading.Lock(), threading.RLock(), StaticCtx()):
            cnts = torch._dynamo.testing.CompileCounter()
            res = torch.compile(f, backend=cnts)(cm, x)
            self.assertEqual(res, x + 1)
            self.assertEqual(cnts.frame_count, 0)  # skipped to eager, no crash

        # constructed in-trace (SideEffects.get_variable_cls path)
        def g(x):
            # [FUNCTION DOC] HELPER: Enters a newly constructed static context manager and
            # returns the input plus one.
            with StaticCtx():
                return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        self.assertEqual(torch.compile(g, backend=cnts)(x), x + 1)

    def test_ngb_suppressed_for_inline_module(self):
        # [FUNCTION DOC] Checks nested graph breaks are suppressed for functions whose file is
        # reported as suppressed (via a patched trace rules predicate), so the break propagates
        # to the parent giving more than 2 frames. Result must still equal input plus 6.
        """NGB should be suppressed for functions in NGB_SUPPRESS_INLINELIST."""
        from unittest.mock import patch

        def inner(x):
            # [FUNCTION DOC] HELPER: Adds one to the input, forces a graph break, then adds two.
            x = x + 1
            torch._dynamo.graph_break()
            return x + 2

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def outer(x):
            # [FUNCTION DOC] HELPER: Compiled; adds three to inner.
            return inner(x) + 3

        inp = torch.randn(3)
        inner_file = inner.__code__.co_filename
        with patch(
            "torch._dynamo.trace_rules.is_ngb_suppressed_inline",
            side_effect=lambda f: f == inner_file,
        ):
            self.assertEqual(outer(inp), inp + 6)

        # NGB normally handles an inlined graph break in 2 frames.
        # With suppression, the break propagates to the parent, producing more.
        self.assertGreater(cnts.frame_count, 2)

    def test_complex_nan_constant_across_graph_break(self):
        # [FUNCTION DOC] Regression test: constants with non-finite imaginary parts survive a
        # graph break in a loop (the generated code must use the complex constructor). Checks the
        # result is NaN.
        """Graph break with complex constants containing nan/inf imaginary parts.

        Regression test: repr(complex(x, nan)) produces "(x+nanj)" which Python
        parses as a single identifier, not nan*1j. The FX codegen must use the
        complex() constructor form for non-finite components.
        """
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; builds a list of three complex constants, and in a
            # loop forces a graph break and adds each constant to a running total.
            vals = [
                complex(1.0, float("nan")),
                complex(float("inf"), 2.0),
                complex(float("nan"), float("nan")),
            ]
            total = x
            for v in vals:
                torch._dynamo.graph_break()
                total = total + v
            return total

        inp = torch.tensor(0.0 + 0j)
        result = fn(inp)
        self.assertTrue(result.isnan())

    def test_fstring_graph_break_in_custom_str(self):
        # [FUNCTION DOC] Regression test: formatting a function signature in an f-string, which
        # involves a graph break while building a string, must re-raise rather than swallow the
        # error. Checks the result is input plus one.
        """f-string formatting of an object whose __str__ causes a graph break.

        Regression test: when generic_str inlines __str__ and compile_subgraph
        is called (setting should_exit=True) but fails with Unsupported,
        _format_value must re-raise rather than silently swallowing the error.
        """
        import inspect

        def target_fn(x: list[int]) -> int:
            # [FUNCTION DOC] HELPER: Sums a list of integers.
            return sum(x)

        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; gets a signature object, formats it in an
            # f-string, returns the input plus one.
            sig = inspect.signature(target_fn)
            f"{sig}"
            return x + 1

        inp = torch.randn(3)
        self.assertEqual(fn(inp), inp + 1)

    def test_exhausted_generator_across_graph_break(self):
        # [FUNCTION DOC] Regression test: reconstructing an exhausted generator that is still
        # a live local across a nested graph break must not crash. Checks the result is input
        # plus 3.
        """Reconstruct an exhausted generator after a graph break.

        Regression test: LocalGeneratorObjectVariable.reconstruct() crashed
        with AttributeError on 'remaining_items' when the generator was
        exhausted, because the field was only set inside a conditional.
        The generator must be fully consumed (exhausted) and still in locals
        when a graph break occurs inside a nested inlined function -- the NGB
        stack reconstruction includes outer locals, so the exhausted generator
        gets reconstruct()'d.
        """
        cnts = torch._dynamo.testing.CompileCounter()

        def my_generator(n):
            # [FUNCTION DOC] HELPER: Generator yielding integers from zero up to n.
            for i in range(n):  # noqa: UP028
                yield i

        def inner(x, val):
            # [FUNCTION DOC] HELPER: Forces a graph break and adds the value to the input.
            torch._dynamo.graph_break()
            return x + val

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; consumes a generator to a total, calls inner, then
            # touches the type of the generator so it stays live, and returns the result.
            gen = my_generator(3)
            total = 0
            for val in gen:
                total += val
            result = inner(x, total)
            # Reference gen after the graph break to keep it live in locals
            # during NGB stack reconstruction.
            type(gen)
            return result

        inp = torch.tensor(10.0)
        result = fn(inp)
        self.assertEqual(result, inp + 3)

    def test_generator_star_unpack_with_graph_break(self):
        # [FUNCTION DOC] Regression test: a generator unpacked as positional arguments with
        # keyword arguments must survive two codegen passes. Checks the result shape is one
        # by two.
        """Generator unpacked as *args must survive two-pass codegen.

        reconstruct() is called twice during compile_subgraph (pass1 for use
        counting, pass2 for actual codegen). The remaining items from the
        generator must be cached so the second call doesn't see an empty
        generator. CALL_FUNCTION_EX with *generator + **kwargs triggers a
        graph break, putting the unconsumed generator on the stack.
        """

        def my_generator(n):
            # [FUNCTION DOC] HELPER: Generator yielding the integers one to n.
            for i in range(n):
                yield i + 1

        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Compiled; unpacks a two-item generator as size arguments to
            # a random tensor with grad enabled.
            shape = my_generator(2)
            return torch.randn(*shape, requires_grad=True)

        result = fn()
        self.assertEqual(result.shape, torch.Size([1, 2]))

    def test_load_attr_method_variant_nested_graph_break(self):
        # [FUNCTION DOC] Regression test: two graph breaks inside a method called through an
        # object (resume into resume) must push the correct receiver for the call layout. Checks
        # result plus 2 and 2 frames.
        """Resume-into-resume with LOAD_ATTR method variant.

        Two graph breaks inside a method called via obj.method(x) test that the
        resume prefix correctly pushes NULL/self for the method call stack layout.
        """

        class Foo:
            # [CLASS DOC] HELPER: Object with a method that has two graph breaks around
            # an addition.
            def method(self, x):
                # [FUNCTION DOC] HELPER: Breaks, adds one, breaks again, adds one.
                torch._dynamo.graph_break()
                y = x + 1
                torch._dynamo.graph_break()
                return y + 1

        obj = Foo()
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Compiled; calls the method on the captured object.
            return obj.method(x)

        x = torch.randn(3)
        result = fn(x)
        self.assertEqual(result, x + 2)
        self.assertEqual(cnts.frame_count, 2)

    def test_tree_map_graph_break_preserves_structure(self):
        # [FUNCTION DOC] Checks tree mapping a function creating grad-requiring zeros over a list
        # preserves list type and leaf shape.
        from torch.utils._pytree import tree_map

        def fn():
            # [FUNCTION DOC] HELPER: Builds a one-tensor list and tree-maps zeros_like with grad
            # over it.
            inps = [torch.randn(3)]
            return tree_map(lambda x: torch.zeros_like(x, requires_grad=True), inps)

        cnts = torch._dynamo.testing.CompileCounter()
        result = torch.compile(fn, backend=cnts)()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 1)
        self.assertIsInstance(result[0], torch.Tensor)
        self.assertEqual(result[0].shape, (3,))
        self.assertTrue(result[0].requires_grad)

    def test_tree_map_graph_break_nested_structure(self):
        # [FUNCTION DOC] Same as the previous but with a nested dict, list and tuple structure;
        # checks types, lengths and shapes are preserved.
        from torch.utils._pytree import tree_map

        def fn():
            # [FUNCTION DOC] HELPER: Builds a nested structure of random tensors and tree-maps
            # zeros_like with grad over it.
            inps = {"a": [torch.randn(3)], "b": (torch.randn(2), torch.randn(4))}
            return tree_map(lambda x: torch.zeros_like(x, requires_grad=True), inps)

        result = torch.compile(fn, backend="eager")()
        self.assertIsInstance(result, dict)
        self.assertIsInstance(result["a"], list)
        self.assertEqual(len(result["a"]), 1)
        self.assertIsInstance(result["b"], tuple)
        self.assertEqual(len(result["b"]), 2)
        self.assertEqual(result["b"][0].shape, (2,))
        self.assertEqual(result["b"][1].shape, (4,))

    def test_tree_map_with_path_graph_break_preserves_structure(self):
        # [FUNCTION DOC] Same idea with the path-aware tree map over a list of two tensors;
        # checks list, length, shapes and grad flag.
        from torch.utils._pytree import tree_map_with_path

        def fn():
            # [FUNCTION DOC] HELPER: Builds two random tensors and tree-maps with path a
            # zeros_like with grad function.
            inps = [torch.randn(3), torch.randn(2)]
            return tree_map_with_path(
                lambda path, x: torch.zeros_like(x, requires_grad=True), inps
            )

        result = torch.compile(fn, backend="eager")()
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0].shape, (3,))
        self.assertEqual(result[1].shape, (2,))
        self.assertTrue(result[0].requires_grad)

    def test_tree_map_only_graph_break_preserves_structure(self):
        # [FUNCTION DOC] Checks the tree map restricted to tensors preserves list structure when
        # the mapped function contains a graph break.
        from torch.utils._pytree import tree_map_only

        def map_fn(x):
            # [FUNCTION DOC] HELPER: Forces a graph break and adds one.
            torch._dynamo.graph_break()
            return x + 1

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies map_fn to tensors in a list of the two inputs.
            return tree_map_only(torch.Tensor, map_fn, [x, y])

        x, y = torch.randn(3), torch.randn(2)
        result = torch.compile(fn, backend="eager")(x, y)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x + 1)
        self.assertEqual(result[1], y + 1)

    def test_tree_map_functools_partial_graph_break_preserves_structure(self):
        # [FUNCTION DOC] Checks tree map with a partially applied function containing a graph
        # break preserves the list structure and values.
        import functools

        from torch.utils._pytree import tree_map

        def add_bias(bias, x):
            # [FUNCTION DOC] HELPER: Forces a graph break and adds the bias to the input.
            torch._dynamo.graph_break()
            return x + bias

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Tree-maps the partially applied function over a list of the
            # two inputs.
            return tree_map(functools.partial(add_bias, 1), [x, y])

        x, y = torch.randn(3), torch.randn(2)
        result = torch.compile(fn, backend="eager")(x, y)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x + 1)
        self.assertEqual(result[1], y + 1)

    def test_tree_map_no_grad_wrapped_graph_break_preserves_structure(self):
        # [FUNCTION DOC] Checks tree map with a no_grad-wrapped function containing a graph break
        # preserves structure and values.
        from torch.utils._pytree import tree_map

        def map_fn(x):
            # [FUNCTION DOC] HELPER: Forces a graph break and adds one.
            torch._dynamo.graph_break()
            return x + 1

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Tree-maps the no_grad-wrapped function over a list of the
            # two inputs.
            return tree_map(torch.no_grad()(map_fn), [x, y])

        x, y = torch.randn(3), torch.randn(2)
        result = torch.compile(fn, backend="eager")(x, y)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x + 1)
        self.assertEqual(result[1], y + 1)

    def test_tree_map_no_grad_nested_fn_graph_break_preserves_structure(self):
        # [FUNCTION DOC] Same as the previous test but the mapped function is defined inside the
        # compiled function.
        from torch.utils._pytree import tree_map

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Defines a nested mapping function and tree-maps its no_grad
            # wrapped version over the two inputs.
            def map_fn(t):
                # [FUNCTION DOC] HELPER: Forces a graph break and adds one.
                torch._dynamo.graph_break()
                return t + 1

            return tree_map(torch.no_grad()(map_fn), [x, y])

        x, y = torch.randn(3), torch.randn(2)
        result = torch.compile(fn, backend="eager")(x, y)
        self.assertIsInstance(result, list)
        self.assertEqual(len(result), 2)
        self.assertEqual(result[0], x + 1)
        self.assertEqual(result[1], y + 1)

    def test_tree_map_graph_break_in_inlined_function(self):
        # [FUNCTION DOC] Checks nested graph breaks still work on the caller after tree map
        # suppression restores config: the function is compiled with a graph recording backend,
        # result checked, and exactly four graphs captured.
        """NGB should work on the caller frame after tree_map suppression restores config."""
        from torch.utils._pytree import tree_map

        def map_fn(t):
            # [FUNCTION DOC] HELPER: Forces a graph break and adds one.
            torch._dynamo.graph_break()
            return t + 1

        def inner(x, y):
            # [FUNCTION DOC] HELPER: Adds one, tree-maps the breaking function over a pair, and
            # returns the first element plus two.
            a = x + 1
            result = tree_map(map_fn, [a, y])
            return result[0] + 2

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Doubles the input, calls inner and triples the result.
            a = x * 2
            result = inner(a, y)
            return result * 3

        x, y = torch.randn(3), torch.randn(2)
        backend = torch._dynamo.testing.EagerAndRecordGraphs()
        result = torch.compile(fn, backend=backend)(x, y)
        self.assertEqual(result, (x * 2 + 1 + 1 + 2) * 3)
        # With NGB on the caller frame, fn's ops merge with inner's ops into
        # shared partial graphs (4 graphs). Without NGB, inner would run as a
        # separate frame and produce 6 isolated single-op graphs.
        self.assertEqual(len(backend.graphs), 4)

    @unittest.skipIf(not python_pytree._cxx_pytree_exists, "missing optree package")
    def test_cxx_pytree_treespec_leaf_namespace(self):
        # [FUNCTION DOC] Checks the C++ pytree tree flatten and leaf spec helpers work across a
        # graph break and that the spec equals the leaf spec. Skipped when the optional optree
        # package is missing.
        import torch.utils._cxx_pytree as cxx_pytree

        def fn():
            # [FUNCTION DOC] HELPER: Flattens an integer, builds the leaf spec, forces a graph
            # break, returns values and the equality of the spec with the leaf spec.
            values, treespec = cxx_pytree.tree_flatten(1)
            leaf = cxx_pytree.treespec_leaf()
            torch._dynamo.graph_break()
            return values, treespec == leaf

        cnts = torch._dynamo.testing.CompileCounter()
        values, specs_equal = torch.compile(fn, backend=cnts)()
        self.assertEqual(values, [1])
        self.assertTrue(specs_equal)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
