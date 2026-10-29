# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests that Dynamo splits functions into multiple subgraphs at graph breaks
# (data-dependent control flow, untraceable C calls, calls to helpers that break) and resumes
# correctly afterwards. Defines a global ReLU module and a helper that wraps the unsupported C
# function; the single test class compares compiled results with eager and checks the graph and
# operator counts. No import-time side effects beyond creating the module.
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.testing import unsupported
from torch._dynamo.utils import ifdynstaticdefault


globalmod = torch.nn.ReLU()


def indirectly_unsupported(a, b):
    # [FUNCTION DOC] HELPER: Test helper that adds its two tensor arguments, then passes the
    # original first argument and the sum to the untraceable C function, which returns its second
    # argument. Used to trigger a graph break from inside a nested call.
    c = a + b
    return unsupported(a, c)


class SubGraphTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case with one shared checker and many tests, each defining a small
    # function that is compiled with a counting backend. Tests assert that results match eager
    # and that the number of compiled frames and captured operators equals the values expected
    # for the pattern under test (control flow, unsupported calls, resume after a break,
    # iterators, dynamic shapes).
    def _common(self, fn, frame_count, op_count):
        # [FUNCTION DOC] HELPER: Shared checker for the tests. Resets Dynamo, runs the given
        # function eagerly on two ones/minus-two tensors in both argument orders, compiles it with
        # a compile counter backend, runs it in the same orders, and asserts the outputs match.
        # Then asserts the counter frame count equals frame_count and the operator count equals
        # op_count (frame mismatch message shows both values).
        torch._dynamo.reset()
        v1 = torch.ones(10)
        v2 = torch.ones(10) * -2.0
        correct1 = fn(v1, v2)
        correct2 = fn(v2, v1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        r1 = opt_fn(v1, v2)
        r2 = opt_fn(v2, v1)
        self.assertTrue(torch._dynamo.testing.same(r1, correct1))
        self.assertTrue(torch._dynamo.testing.same(r2, correct2))
        self.assertEqual(
            cnt.frame_count,
            frame_count,
            lambda msg: f"{msg}\nactual {cnt.frame_count} != expected {frame_count}",
        )
        self.assertEqual(cnt.op_count, op_count)

    def test_control_flow1(self):
        # [FUNCTION DOC] Compiles a function with a tensor-dependent branch where both branches
        # return a tensor; checks 1 frame and 5 ops via the shared checker.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes the two differences of its inputs and returns the
            # larger-summed one, branching on a tensor comparison.
            c1 = a - b
            c2 = b - a
            if c1.sum() > c2.sum():
                return c1
            else:
                return c2

        self._common(fn, 1, 5)

    def test_control_flow2(self):
        # [FUNCTION DOC] Compiles a function whose tensor-dependent branch returns Python integer
        # constants; checks 1 frame and 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the constant 1 or 2 depending on whether the sum of
            # the first argument exceeds the sum of the second.
            if a.sum() > b.sum():
                return 1
            else:
                return 2

        self._common(fn, 1, 3)

    def test_control_flow3(self):
        # [FUNCTION DOC] Like the first control-flow test but each branch applies the module-level
        # ReLU module; checks 3 frames and 7 ops (the branch plus module call cause extra graphs).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Takes the two differences of the inputs, picks one by
            # comparing sums, and passes it through the global ReLU module.
            c1 = a - b
            c2 = b - a
            m = globalmod
            if c1.sum() > c2.sum():
                return m(c1)
            else:
                return m(c2)

        self._common(fn, 3, 7)

    def test_control_flow4(self):
        # [FUNCTION DOC] Branches on a short-circuit conjunction of two tensor comparisons; checks
        # 3 frames and 5 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes a boolean from a conjunction of tensor comparisons
            # and returns constant 1 or 2 based on it.
            tmp1 = a.sum() > b.sum() and a.sum() > 0
            if tmp1:
                return 1
            else:
                return 2

        self._common(fn, 3, 5)

    def test_control_flow5(self):
        # [FUNCTION DOC] Two short-circuit boolean expressions (conjunction and disjunction) feed a
        # branch; checks 6 frames and 13 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds two boolean values from tensor sum comparisons and
            # returns a constant plus both booleans, chosen by their conjunction.
            tmp1 = a.sum() > b.sum() and a.sum() > 0
            tmp2 = a.sum() < b.sum() or b.sum() > 0
            if tmp1 and tmp2:
                return 1, tmp1, tmp2
            else:
                return 2, tmp1, tmp2

        self._common(fn, 6, 13)

    def test_capi_call1(self):
        # [FUNCTION DOC] Compiles a function that calls the untraceable C function on two computed
        # tensors; checks 1 frame and 2 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes both differences of the inputs and returns the result
            # of the untraceable C function applied to them.
            c1 = a - b
            c2 = b - a
            return unsupported(c1, c2)

        self._common(fn, 1, 2)

    def test_capi_call2(self):
        # [FUNCTION DOC] Untraceable C call nested inside an arithmetic expression, so work happens
        # both before and after the break; checks 2 frames and 4 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the first input minus the quantity second input minus
            # the untraceable call result on two differences.
            c1 = a - b
            c2 = b - a
            return a - (b - unsupported(c1, c2))

        self._common(fn, 2, 4)

    def test_capi_call3(self):
        # [FUNCTION DOC] Same as the first C-call test but the C function is referenced through the
        # full module path; checks 1 frame and 2 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes both differences and calls the untraceable function
            # via its fully qualified module attribute.
            c1 = a - b
            c2 = b - a
            return torch._dynamo.testing.unsupported(c1, c2)

        self._common(fn, 1, 2)

    def test_indirect_unsupported1(self):
        # [FUNCTION DOC] Calls a helper that itself calls the unsupported C function (break occurs
        # in a nested call); checks 1 frame and 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes both differences and returns the module-level helper
            # applied to them.
            c1 = a - b
            c2 = b - a
            return indirectly_unsupported(c1, c2)

        self._common(fn, 1, 3)

    def test_indirect_unsupported2(self):
        # [FUNCTION DOC] Indirect unsupported call used inside arithmetic with local constants, so
        # code runs after the break; checks 2 frames and 5 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Defines two local integer constants and returns the first
            # divided by the second minus the helper result.
            local_const1 = 7
            local_const2 = 22
            c1 = a - b
            c2 = b - a
            return local_const1 / (local_const2 - indirectly_unsupported(c1, c2))

        self._common(fn, 2, 5)

    def test_indirect_unsupported3(self):
        # [FUNCTION DOC] Indirect unsupported call where the helper is invoked with star-unpacked
        # list arguments; checks 1 frame and 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a list of the two differences and calls the helper with
            # that list unpacked as positional arguments.
            args = [a - b, b - a]
            return indirectly_unsupported(*args)

        self._common(fn, 1, 3)

    def test_stack_state1(self):
        # [FUNCTION DOC] Checks that values already on the stack (two scaled tensors) survive a
        # graph break in the middle of an expression; checks 2 frames and 6 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes two scaled copies of the first input and two
            # differences, then divides one scaled value by the other minus the unsupported
            # call result.
            t1 = 1.23 * a
            t2 = 4.56 * a
            c1 = a - b
            c2 = b - a
            return t1 / (t2 - unsupported(c1, c2))

        self._common(fn, 2, 6)

    def test_stack_state2(self):
        # [FUNCTION DOC] Same stack-preservation scenario as the previous test but the break comes
        # from the indirect helper; checks 2 frames and 7 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: As the previous variant but the break comes from the indirect
            # helper rather than the C function directly.
            t1 = 1.23 * a
            t2 = 4.56 * a
            c1 = a - b
            c2 = b - a
            return t1 / (t2 - indirectly_unsupported(c1, c2))

        self._common(fn, 2, 7)

    def test_multigraph(self):
        # [FUNCTION DOC] Checks a tensor-dependent branch after some arithmetic yields two frames
        # and 5 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds and halves the inputs, then multiplies by minus one when
            # the sum is negative, else returns unchanged.
            x = a + b
            x = x / 2.0
            if x.sum() < 0:
                return x * -1.0
            return x

        self._common(fn, 2, 5)

    def test_extended_args(self):
        # [FUNCTION DOC] Builds source text for a lambda with 512 operands summed on each side of a
        # tensor-dependent conditional (enough bytecode to require extended arguments), evaluates
        # it, and checks 3 frames and 1026 ops.
        too_many_adds = "+".join(["a", "b"] * 256)
        source = (
            f"lambda a, b: ({too_many_adds}+a if a.sum() > 0 else {too_many_adds} - b)"
        )
        self._common(eval(source), 3, 1026)

    def test_resume1(self):
        # [FUNCTION DOC] Checks resuming after a graph break from the direct unsupported call, with
        # arithmetic after it; expects 2 frames and 6 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Does several additions and a division, calls the unsupported
            # function with the running value and first input, then adds 2.0 three more times.
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = unsupported(x, a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 6)

    def test_resume2(self):
        # [FUNCTION DOC] Resume after a break caused by the indirect helper called with positional
        # arguments; expects 2 frames and 7 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Same arithmetic chain as the first resume test but the break
            # call goes through the indirect helper.
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = indirectly_unsupported(x, a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 7)

    def test_resume3(self):
        # [FUNCTION DOC] Resume after a break where the indirect helper is called with the second
        # parameter as a keyword; expects 2 frames and 7 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Arithmetic chain with a break call that passes the first
            # argument positionally and the second by keyword.
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = indirectly_unsupported(x, b=a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 7)

    def test_resume4(self):
        # [FUNCTION DOC] Resume after a break where the helper is called with both arguments as
        # keywords; expects 2 frames and 7 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Arithmetic chain with a break call that passes both arguments
            # by keyword.
            x = a + b
            x = x / 2.0
            x = x + 2.0
            x = indirectly_unsupported(a=x, b=a)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 7)

    def test_resume5(self):
        # [FUNCTION DOC] Resume after a break caused by a print call; expects 2 frames and 6 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Arithmetic chain with a print of the running value in the
            # middle, followed by more additions.
            x = a + b
            x = x / 2.0
            x = x + 2.0
            print(x)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 6)

    def test_start1(self):
        # [FUNCTION DOC] Break occurs at the very start of the function (print of an argument);
        # expects 1 frame and 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Prints its first argument, then adds the inputs and adds
            # 2.0 twice.
            print(a)
            x = a + b
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 1, 3)

    def test_start2(self):
        # [FUNCTION DOC] Break at function start through the indirect helper; expects 2 frames and
        # 4 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls the indirect helper first, then adds 2.0 three times.
            x = indirectly_unsupported(a, b)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 2, 4)

    def test_start3(self):
        # [FUNCTION DOC] Break at function start through the direct unsupported call; expects 1
        # frame and 3 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Calls the unsupported function first, then adds 2.0
            # three times.
            x = unsupported(a, b)
            x = x + 2.0
            x = x + 2.0
            x = x + 2.0
            return x

        self._common(fn, 1, 3)

    def test_start4(self):
        # [FUNCTION DOC] Compiles a function branching on a one-element integer tensor argument
        # with both a true and false tensor, compares to eager, and asserts 3 frames and 4 ops
        # overall (does not use the shared checker).
        def fn(a, b, check):
            # [FUNCTION DOC] HELPER: Returns the sum of the first two arguments plus 10 when the
            # third argument is truthy, otherwise minus 10.
            if check:
                return a + b + 10
            else:
                return a + b - 10

        v1 = torch.randn(10)
        v2 = torch.randn(10)
        f = torch.zeros(1, dtype=torch.int32)
        t = torch.ones(1, dtype=torch.int32)
        correct1 = fn(v1, v2, t)
        correct2 = fn(v1, v2, f)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        r1 = opt_fn(v1, v2, t)
        r2 = opt_fn(v1, v2, f)
        self.assertTrue(torch._dynamo.testing.same(r1, correct1))
        self.assertTrue(torch._dynamo.testing.same(r2, correct2))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 4)

    def test_resume_freevars(self):
        # [FUNCTION DOC] Resume after a break in a function that closes over two tensors defined in
        # the test; expects 2 frames and 5 ops.
        c1 = torch.randn(10)
        c2 = torch.randn(10)

        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the inputs plus the difference of the two captured
            # tensors, calls the unsupported function, then adds the same difference again.
            x = a + b + (c1 - c2)
            x = unsupported(x, x)
            return x + (c1 - c2)

        self._common(fn, 2, 5)

    def test_restore_state(self):
        # [FUNCTION DOC] Checks that a local variable bound to the len builtin survives a graph
        # break; expects 2 frames and 4 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Stores the len builtin in a local, breaks the graph via the
            # unsupported call whose result gets one added using torch add, then returns a product
            # plus the length of the second input via the local.
            len_ = len
            x = a + b
            x = torch.add(unsupported(x, x), 1)
            return a * x + len_(b)

        self._common(fn, 2, 4)

    def test_restore_range(self):
        # [FUNCTION DOC] Checks that a live range object survives a break and the loop after it
        # runs. Frame and op counts depend on whether static shapes are the default (2 and 4 versus
        # 1 and 1) because range loops are not unrolled with dynamic shapes.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Creates a stepped range, breaks the graph, then loops over the
            # range adding each value to the running tensor.
            x = a + b
            rng = range(3, 8, 2)
            x = unsupported(x, x)
            for i in rng:
                x = x + i
            return x

        # We don't specialize on range with dynamic shapes, which
        # means we fail to unroll the loop.
        # TODO: Consider forcing specialization when we iterate over
        # the loop
        self._common(fn, ifdynstaticdefault(2, 1), ifdynstaticdefault(4, 1))

    def test_restore_range_iter(self):
        # [FUNCTION DOC] Checks that a partially consumed range iterator is restored after a graph
        # break; expects 2 frames and 2 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Creates an iterator over a range, breaks the graph, adds the
            # next iterator value, and returns the tensor with the list of remaining values.
            x = a + b
            rng = iter(range(3, 8, 2))
            x = unsupported(x, x)
            x += next(rng)
            return x, list(rng)

        self._common(fn, 2, 2)

    def test_pop_after_resume(self):
        # [FUNCTION DOC] Checks that list pop operations work on a list created before a break and
        # used after resuming; expects 2 frames and 6 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Builds a list of three tensors, breaks the graph, then pops
            # its last element three times in a loop adding each to the running value.
            tmp = [a + 1, b + 2, a + b]
            x = a
            x = unsupported(x, x)
            for _ in range(3):
                x += tmp.pop(-1)
            return x

        self._common(fn, 2, 6)

    @patch("torch._dynamo.config.assume_static_by_default", False)
    def test_dynamic_getitem(self):
        # [FUNCTION DOC] With the assume-static-by-default config patched off, compiles an indexing
        # function over many input sizes and asserts only one frame is compiled.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Indexes the first argument at the last position computed from
            # the size of the second argument.
            return a[b.size(0) - 1]

        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        for i in range(3, 12):
            opt_fn(torch.randn(i), torch.randn(i))
        # just one graph
        self.assertEqual(cnt.frame_count, 1)

    def test_dynamic_kwarg(self):
        # [FUNCTION DOC] Compiles with dynamic shapes forced on and runs ten different input sizes;
        # asserts a single compiled frame.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the first input minus ten times the second.
            return a - b * 10

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt_dynamic, dynamic=True)
        start = 2
        end = 12
        for i in range(start, end):
            opt_fn(torch.randn(i), torch.randn(i))

        self.assertEqual(cnt_dynamic.frame_count, 1)

    def test_dynamic_duck_size(self):
        # [FUNCTION DOC] Compiles with dynamic shapes forced on; the function branches on equal
        # sizes. Asserts outputs match eager for equal and unequal sizes and that two frames
        # are compiled.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds the inputs when their first dimensions match, else adds
            # their sums.
            if a.size(0) == b.size(0):
                return a + b
            else:
                return a.sum() + b.sum()

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt_dynamic, dynamic=True)
        x = torch.randn(2)
        y = torch.randn(3)
        self.assertEqual(opt_fn(x, x), fn(x, x))
        self.assertEqual(opt_fn(x, y), fn(x, y))
        self.assertEqual(cnt_dynamic.frame_count, 2)

    def test_dynamic_order_dependence(self):
        # [FUNCTION DOC] Checks order dependence of size duck-typing: runs with different sizes
        # then equal sizes, expecting 2 frames, then resets and runs equal first (overspecializes)
        # then different, again expecting 2 frames.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of the two inputs' sums.
            return a.sum() + b.sum()

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt_dynamic)
        x = torch.randn(2)
        y = torch.randn(3)
        self.assertEqual(opt_fn(x, y), fn(x, y))
        self.assertEqual(opt_fn(x, x), fn(x, x))
        # NB: This COULD validly be 2, but we don't test disjointedness in the
        # guards for when x and y didn't duck size together, so we end up
        # with a generic graph that also works when x and y happen to duck
        # size together.
        self.assertEqual(cnt_dynamic.frame_count, 2)

        torch._dynamo.reset()
        cnt_dynamic.frame_count = 0
        self.assertEqual(opt_fn(x, x), fn(x, x))  # this overspecializes!
        self.assertEqual(opt_fn(x, y), fn(x, y))
        self.assertEqual(cnt_dynamic.frame_count, 2)

    def test_dynamic_zero_inference(self):
        # [FUNCTION DOC] With dynamic compilation, checks a branch on whether the size is zero is
        # correct for both a nonempty and an empty input, using 2 frames.
        def fn(a):
            # [FUNCTION DOC] HELPER: Multiplies by 2 when the first dimension is nonzero, else
            # adds 1.
            if a.size(0) != 0:
                return a * 2
            else:
                return a + 1

        torch._dynamo.reset()
        cnt_dynamic = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt_dynamic, dynamic=True)
        x = torch.randn(0)
        y = torch.randn(2)
        self.assertEqual(opt_fn(y), fn(y))
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnt_dynamic.frame_count, 2)

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", True)
    def test_no_graph_break_on_item(self):
        # [FUNCTION DOC] With scalar-output capture enabled, calling the item method does not break
        # the graph; expects 1 frame and 5 ops (the item call is eliminated as dead code).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Computes a shifted sum, calls the item method discarding the
            # result, and divides by the sum of inputs.
            x = a + b - 1.5
            x = x.sum()
            x.item()
            x = x / (a + b)
            return x

        self._common(fn, 1, 5)  # item gets DCE'd

    @patch.object(torch._dynamo.config, "capture_scalar_outputs", False)
    def test_graph_break_on_item(self):
        # [FUNCTION DOC] With scalar-output capture disabled, the item call breaks the graph;
        # expects 2 frames and 5 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Same function as the previous item test, used with
            # capture disabled.
            x = a + b - 1.5
            x = x.sum()
            x.item()
            x = x / (a + b)
            return x

        self._common(fn, 2, 5)

    def test_resume_paths_join(self):
        # [FUNCTION DOC] Runs all eight combinations of three tensor conditions through a function
        # with sequential branches and asserts that only 7 frames and 10 ops result, so frames do
        # not grow exponentially.
        def fn(x, c1, c2, c3):
            # [FUNCTION DOC] HELPER: Adds successive constants to a tensor, with three additional
            # additions each guarded by a separate condition argument.
            x = x + 1
            if c1:
                x = x + 2
            x = x + 3
            if c2:
                x = x + 4
            x = x + 5
            if c3:
                x = x + 6
            return x + 7

        v1 = torch.randn(10)
        t = torch.Tensor([True])
        f = torch.Tensor([False])
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        for a in (t, f):
            for b in (t, f):
                for c in (t, f):
                    opt_fn(v1, a, b, c)

        # checking here we don't create 2^n graphs
        self.assertEqual(cnt.frame_count, 7)
        self.assertEqual(cnt.op_count, 10)

    def test_resume_with_no_grad1(self):
        # [FUNCTION DOC] Resume across a break inside a no-grad region; checks 2 frames and 9 ops
        # normally, then after a reset under an outer no-grad context 2 frames and 5 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Adds inputs, then inside a no-grad context adds one, converts
            # a sum to a list (graph break), adds two, then adds three after the context.
            x = a + b
            with torch.no_grad():
                x = x + 1
                x.sum().tolist()  # graph break
                x = x + 2
            x = x + 3
            return x

        self._common(fn, 2, 9)
        torch._dynamo.reset()
        with torch.no_grad():
            self._common(fn, 2, 5)

    def test_resume_with_no_grad2(self):
        # [FUNCTION DOC] Two breaks within one no-grad region; expects 3 frames and 13 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Inside a no-grad context, alternates additions with two list
            # conversions that break the graph, then adds four outside.
            x = a + b
            with torch.no_grad():
                x = x + 1
                x.sum().tolist()  # graph break
                x = x + 2
                x.sum().tolist()  # graph break
                x = x + 3
            x = x + 4
            return x

        self._common(fn, 3, 13)

    def test_resume_with_no_grad3(self):
        # [FUNCTION DOC] Nested no-grad contexts with an inner grad-enabled context containing the
        # break; expects 2 frames and 11 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Nests two no-grad contexts and an enable-grad context,
            # converting a sum to a list inside the innermost and indexing the tensor afterwards.
            x = a + b
            with torch.no_grad():
                with torch.no_grad():
                    x = x + 1
                    with torch.enable_grad():
                        x.sum().tolist()  # graph break
                        x = x[0] + 2
                    x = x + 3
            x = x + 4
            return x

        self._common(fn, 2, 11)

    def test_resume_tuple_iterator(self):
        # [FUNCTION DOC] Checks a tuple iterator stays consistent when a break occurs between next
        # calls; expects 2 frames and 8 ops.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Advances an iterator over a tuple of ten integers three times,
            # breaks the graph, then advances it four more times, adding each value.
            x = a + b
            it = iter(tuple(range(10)))
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            return x

        self._common(fn, 2, 8)

    def test_tuple_iterator_return(self):
        # [FUNCTION DOC] Checks a tuple iterator that is returned from the compiled function after
        # two breaks: values and remaining iterator contents must equal eager, with 3 frames and 6
        # ops after a second call.
        def fn(x):
            # [FUNCTION DOC] HELPER: Consumes two, two and two items of an iterator over a tuple of
            # ten integers, separated by two breaks, returning the tensor and the iterator.
            it = iter(tuple(range(10)))
            x = x + next(it)
            x = x + next(it)
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            x = unsupported(x, x)
            x = x + next(it)
            x = x + next(it)
            return x, it

        v1 = torch.randn(10)
        v2, it2 = fn(v1)
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        v3, it3 = opt_fn(v1)
        v4, _ = opt_fn(v1)
        self.assertEqual(v2.tolist(), v3.tolist())
        self.assertEqual(v2.tolist(), v4.tolist())
        self.assertEqual(list(it2), list(it3))
        self.assertEqual(cnt.frame_count, 3)
        self.assertEqual(cnt.op_count, 6)

    def test_tuple_iterator_mutate(self):
        # [FUNCTION DOC] Checks that an iterator passed in as an argument is advanced by the
        # compiled function (mutation visible to caller): result equals input plus 1 plus 2 plus 3,
        # and the iterator then yields the remaining six values.
        def fn(x, it):
            # [FUNCTION DOC] HELPER: Adds the next four values of the given iterator to the tensor.
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            x = x + next(it)
            return x

        v1 = torch.randn(10)
        it1 = iter(tuple(range(10)))
        cnt = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnt)
        self.assertEqual(opt_fn(v1, it1).tolist(), (v1 + 1 + 2 + 3).tolist())
        self.assertEqual(list(it1), [4, 5, 6, 7, 8, 9])

    def test_enumerate_not_break_graph(self):
        # [FUNCTION DOC] Checks enumerate over shapes does not break the graph, with one frame and
        # an op count that depends on the static or dynamic default (2 or 3).
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Loops over enumerate of the first input's shape and of the
            # second's shape with a start offset, adding values or products to the second tensor.
            for _, x in enumerate(a.shape):
                b = b + x
            for i, x in enumerate(b.shape, 8):
                b = b + x * i
            return b

        self._common(fn, 1, ifdynstaticdefault(2, 3))


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
