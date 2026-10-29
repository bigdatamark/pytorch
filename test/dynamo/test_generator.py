# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for tracing Python generators under torch.compile.
# Covers plain and infinite generators, graph breaks inside and around generators,
# reconstructing partially advanced generators for the caller, generators passed as
# arguments, yield from delegation, send, close and throw handling, PEP 479
# conversion of StopIteration, side effects, and generator expressions.
# Contains the shared base class plus several test classes; import-time side effects
# are limited to defining them and registering parametrized variants.
import itertools
import sys
import types
import unittest
from collections import OrderedDict

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import EagerAndRecordGraphs, normalize_gm
from torch._dynamo.utils import counters
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    make_dynamo_test,
    parametrize,
)


class GeneratorTestsBase(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Base test case for the generator tests in this module. Setup turns on
    # two Dynamo config flags (tracing of the build-class opcode and of unittest) and
    # teardown restores the previous values. Provides a helper that compiles a function.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Runs the parent setup, saves the current values of the
        # enable_trace_load_build_class and enable_trace_unittest Dynamo config flags on the
        # instance, then sets both flags to true.
        super().setUp()
        self._prev = torch._dynamo.config.enable_trace_load_build_class
        torch._dynamo.config.enable_trace_load_build_class = True
        self._unittest_old = torch._dynamo.config.enable_trace_unittest
        torch._dynamo.config.enable_trace_unittest = True

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Runs the parent teardown, then restores the two Dynamo
        # config flags (unittest tracing and build-class tracing) saved by setUp.
        super().tearDown()
        torch._dynamo.config.enable_trace_unittest = self._unittest_old
        torch._dynamo.config.enable_trace_load_build_class = self._prev

    def _compile_check(self, fn, args=None, fullgraph=True):
        # [FUNCTION DOC] HELPER: Compiles the given function with a recording eager backend
        # (full graph by default, optional flag to allow graph breaks), calls it with args
        # (default: one random tensor of two elements), asserts at least one graph was
        # captured and returns the call result.
        eager = EagerAndRecordGraphs()
        if args is None:
            args = (torch.randn(2),)
        r = torch.compile(fn, backend=eager, fullgraph=fullgraph)(*args)
        self.assertGreater(len(eager.graphs), 0)
        return r


class GeneratorTests(GeneratorTestsBase):
    # [CLASS DOC] Core generator tests: simple and infinite generators, graph breaks,
    # reconstruction of advanced generators on return, generators as arguments, side
    # effects, CLEANUP_THROW handling, PEP 479 and iteration helpers.
    def test_generator_simple(self):
        # [FUNCTION DOC] Three-yield generator advanced by next inside a full-graph compiled
        # function; the result equals the input plus 1+2+3.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding the integers one, two and three.
            yield 1
            yield 2
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body that adds three successive values from the
            # generator to the input and returns the sum.
            gen = whoo()
            t = t + next(gen)
            t = t + next(gen)
            t = t + next(gen)
            return t

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 6)

    def test_infinite_generator(self):
        # [FUNCTION DOC] An endless counting generator advanced three times in a compiled
        # full-graph function; checks the result equals input plus 0+1+2.
        def whoo():
            # [FUNCTION DOC] HELPER: Infinite generator yielding 0, 1, 2, ... forever.
            i = 0
            while True:
                yield i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body adding three generator values to the input.
            gen = whoo()
            t = t + next(gen)
            t = t + next(gen)
            t = t + next(gen)
            return t

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 3)

    def test_infinite_generator_2(self):
        # [FUNCTION DOC] Infinite generator that depends on a tensor argument, consumed with
        # a bounded zip against a range of three; the compiled list result must equal the
        # eager one.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Infinite generator yielding the tensor plus
            # a growing integer.
            i = 0
            while True:
                yield t + i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the list of zip of range three with
            # the generator.
            return list(zip(range(3), whoo(t)))

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, list(zip(range(3), whoo(t))))

    def test_infinite_generator_3(self):
        # [FUNCTION DOC] Infinite generator of a constant, zipped with range three in a
        # compiled function that also computes a sine; the zip list must match eager.
        def whoo(i):
            # [FUNCTION DOC] HELPER: Infinite generator repeatedly yielding its argument.
            while True:
                yield i

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the zip list and the sine
            # of the input.
            return list(zip(range(3), whoo(1))), t.sin()

        t = torch.randn(2)
        y, _ = fn(t)
        self.assertEqual(y, list(zip(range(3), whoo(1))))

    def test_graph_break_in_generator(self):
        # [FUNCTION DOC] A graph break inside a generator body, with graph breaks allowed.
        # Result is correct (input plus 3) and the recording backend sees zero graphs.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding one, forcing a graph break,
            # then yielding two.
            yield 1
            torch._dynamo.graph_break()
            yield 2

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body that sums two values from the generator and
            # adds them to the input.
            gen = whoo()
            s = next(gen)
            s += next(gen)
            return t + s

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 3)
        self.assertEqual(len(eager.graphs), 0)

    def test_graph_break_in_generator_2(self):
        # [FUNCTION DOC] Generator with a graph break between two yields, driven from a
        # helper function. Checks values and that exactly one graph is captured, whose
        # printed form is compared with an expected inline string (an add of two inputs).
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator yielding sine then (after a graph break) cosine.
            yield x.sin()
            torch._dynamo.graph_break()
            yield x.cos()

        def call_whoo(x):
            # [FUNCTION DOC] HELPER: Pulls two values from the generator and returns them
            # as a pair.
            gen = whoo(x)
            sin = next(gen)
            cos = next(gen)
            return sin, cos

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the sum of the sine
            # and cosine pair.
            sin, cos = call_whoo(t)
            return sin + cos

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin() + t.cos())
        self.assertEqual(len(eager.graphs), 1)
        self.assertExpectedInline(
            normalize_gm(eager.graphs[0].print_readable(False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_stack0_0_: "f32[2]", L_stack0_1_: "f32[2]"):
        l_stack0_0_ = L_stack0_0_
        l_stack0_1_ = L_stack0_1_

        add: "f32[2]" = l_stack0_0_ + l_stack0_1_;  l_stack0_0_ = l_stack0_1_ = None
        return (add,)
""",
        )

    def test_reconstruct_generator_with_local_var_mutation(self):
        # [FUNCTION DOC] Generator with a local variable mutated between yields is advanced
        # once inside compiled code and returned; the caller must be able to finish it,
        # producing the remaining two values with the local counter preserved.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator with a local counter incremented
            # after each yield.
            x = 0
            yield t.sin() + x
            x += 1
            yield t.cos() + x
            x += 1
            yield t.tan() + x

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator once
            # and returning the
            # sine and the live generator.
            gen = whoo(t)
            next(gen)
            return t.sin(), gen

        t = torch.randn(2)
        y, g = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(g), [t.cos() + 1, t.tan() + 2])

    def test_reconstruct_generator_with_dict_mutation(self):
        # [FUNCTION DOC] Returning an advanced generator that mutates a dict argument after
        # the first yield is not reconstructible: the unimplemented counter must hold one
        # entry mentioning variable mutations.
        counters.clear()

        def whoo(t, d):
            # [FUNCTION DOC] HELPER: Generator that writes into the dict before
            # and after yields.
            d[2] = t
            yield t.sin()
            yield t.cos()
            d[3] = t + 1
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, d):
            # [FUNCTION DOC] HELPER: Compiled body advancing one generator, returning
            # the sine and
            # a second fresh generator.
            gen = whoo(t, d)
            next(gen)
            return t.sin(), whoo(t, d)

        t = torch.randn(2)
        d = {1: t}
        fn(t, d)
        self.assertEqual(len(counters["unimplemented"]), 1)
        self.assertIn(
            "Cannot reconstruct a generator with variable mutations",
            next(iter(counters["unimplemented"].keys())),
        )

    def test_reconstruct_generator_with_dict_mutation_before(self):
        # [FUNCTION DOC] Dict mutation occurs only before the first yield, so the advanced
        # generator can be reconstructed: remaining values are correct and the dict argument
        # holds the new key afterwards.
        def whoo(t, d):
            # [FUNCTION DOC] HELPER: Generator that mutates the dict first, then
            # yields three values.
            d[2] = t
            yield t.sin()
            yield t.cos()
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, d):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator once
            # and returning it.
            gen = whoo(t, d)
            next(gen)
            return t.sin(), gen

        t = torch.randn(2)
        d = {1: t}
        y, g = fn(t, d)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(g), [t.cos(), t.tan()])
        self.assertEqual(d, {1: t, 2: t})

    def test_reconstruct_generator_with_object_mutation(self):
        # [FUNCTION DOC] Same as the dict case but with an object attribute incremented after
        # the first yield; reconstruction is rejected and the unimplemented counter records
        # one variable mutations message.
        class Counter:
            # [CLASS DOC] HELPER: Simple counter object with an integer attribute x
            # starting at zero.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the counter attribute to zero.
                self.x = 0

            def incr(self):
                # [FUNCTION DOC] HELPER: Increments the counter attribute by one.
                self.x += 1

        def whoo(t, c):
            # [FUNCTION DOC] HELPER: Generator that increments the counter before
            # and after yields.
            c.incr()
            yield t.sin()
            yield t.cos()
            c.incr()
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, c):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator once
            # and returning it.
            gen = whoo(t, c)
            next(gen)
            return t.sin(), gen

        t = torch.randn(2)
        c = Counter()
        fn(t, c)
        self.assertEqual(len(counters["unimplemented"]), 1)
        self.assertIn(
            "Cannot reconstruct a generator with variable mutations",
            next(iter(counters["unimplemented"].keys())),
        )

    def test_reconstruct_generator_with_object_mutation_before(self):
        # [FUNCTION DOC] Object mutation happens only before the first yield, so the advanced
        # generator is reconstructed: counter is one, remaining values are correct.
        class Counter:
            # [CLASS DOC] HELPER: Simple counter object with an integer attribute x
            # starting at zero.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets the counter attribute to zero.
                self.x = 0

            def incr(self):
                # [FUNCTION DOC] HELPER: Increments the counter attribute by one.
                self.x += 1

        def whoo(t, c):
            # [FUNCTION DOC] HELPER: Generator incrementing the counter once, then
            # yielding three values.
            c.incr()
            yield t.sin()
            yield t.cos()
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t, c):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator once
            # and returning it.
            gen = whoo(t, c)
            next(gen)
            # We should be able to reconstruct the generator as there's no object
            # mutation after the first yield
            return t.sin(), gen

        t = torch.randn(2)
        c = Counter()
        y, g = fn(t, c)
        self.assertEqual(c.x, 1)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(g), [t.cos(), t.tan()])

    def test_graph_break_and_reconstruct_generator(self):
        # [FUNCTION DOC] Advances a generator, hits a graph break in a helper, then drains
        # the generator into a list; the continuing values must be correct.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding sine, cosine and tangent.
            yield t.sin()
            yield t.cos()
            yield t.tan()

        def g(t):
            # [FUNCTION DOC] HELPER: Triggers a graph break and returns nothing.
            torch._dynamo.graph_break()

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body that advances the generator, calls the
            # graph-breaking helper, and returns the sine plus the rest of the generator
            # as a list.
            gen = whoo(t)
            next(gen)
            g(t)
            return t.sin(), list(gen)

        t = torch.randn(2)
        y, gen = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(list(gen), [t.cos(), t.tan()])

    def test_graph_break_in_generator_while_reconstructing(self):
        # [FUNCTION DOC] Graph break inside the generator and another graph break in the
        # caller while the generator is live; result is input plus 3 and no graph recorded.
        counters.clear()

        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding one, graph-breaking, yielding two.
            yield 1
            torch._dynamo.graph_break()
            yield 2

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body with an explicit graph break between two
            # generator advances.
            gen = whoo()
            s = next(gen)
            torch._dynamo.graph_break()
            s += next(gen)
            return t + s

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 3)
        self.assertEqual(len(eager.graphs), 0)

    def test_generator_as_argument(self):
        # [FUNCTION DOC] An already advanced generator passed as argument to a full-graph
        # compiled function must raise Unsupported.
        # The inline tracer needs to be kept in sync if an already advanced generator
        # is given to a compiled function.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding one, two, three.
            yield 1
            yield 2
            yield 3

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            # [FUNCTION DOC] HELPER: Compiled body adding the next generator value to the input.
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo()
        next(ctx)
        with self.assertRaises(Unsupported):
            fn(t, ctx)

    def test_generator_as_argument_2(self):
        # [FUNCTION DOC] Advanced generator that depends on a tensor, passed as argument to
        # a full-graph function; raises Unsupported.
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator yielding the sine and cosine of its argument.
            yield x.sin()
            yield x.cos()

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            # [FUNCTION DOC] HELPER: Compiled body adding the next generator value to the input.
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo(t)
        next(ctx)
        with self.assertRaises(Unsupported):
            fn(t, ctx)

    def test_generator_as_argument_3(self):
        # [FUNCTION DOC] An unstarted argument-less generator passed as an argument is
        # re-inlined, so next works and the result is input plus one.
        # An unstarted generator passed as an argument is re-inlined as if its
        # function were called here, so next() works.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding one, two, three.
            yield 1
            yield 2
            yield 3

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            # [FUNCTION DOC] HELPER: Compiled body adding the next generator value to the input.
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo()
        self.assertEqual(fn(t, ctx), t + 1)

    def test_generator_as_argument_4(self):
        # [FUNCTION DOC] An unstarted generator that takes a tensor argument is passed to a
        # full-graph function; expected to raise Unsupported (unlike the argument-less case).
        def whoo(x):
            # [FUNCTION DOC] HELPER: Generator yielding the sine and cosine of its argument.
            yield x.sin()
            yield x.cos()

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t, ctx):
            # [FUNCTION DOC] HELPER: Compiled body adding the next generator value to the input.
            return t + next(ctx)

        t = torch.randn(2)
        ctx = whoo(t)
        with self.assertRaises(Unsupported):
            fn(t, ctx)

    def test_islice_chain(self):
        # [FUNCTION DOC] Chaining two lists with itertools chain and listing the result in a
        # full-graph function; checks the four tensors are returned in order.
        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body building two lists of tensors
            # and chaining them.
            tmp1 = [t + 1, t + 2]
            tmp2 = [t + 3, t + 4]
            return list(itertools.chain(tmp1, tmp2))

        t = torch.tensor([1.0])
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3, t + 4])

    def test_zip_generator(self):
        # [FUNCTION DOC] Returns a zip over a range and a generator (plus a sine) from a
        # function compiled via the base helper; the lazily returned zip lists to match eager.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the input plus one, two, three.
            yield t + 1
            yield t + 2
            yield t + 3

        def fn(t):
            # [FUNCTION DOC] HELPER: Returns a zip of range three and the generator and a sine.
            return zip(range(3), whoo(t)), t.sin()

        t = torch.randn(2)
        z, _ = self._compile_check(fn, args=(t,))
        self.assertEqual(list(z), list(zip(range(3), whoo(t))))

    @unittest.expectedFailure
    def test_zip_generator_2(self):
        # [FUNCTION DOC] Marked expected failure: zip of a range and a generator that calls a
        # helper function per item, compiled allowing graph breaks, is compared to eager.
        def bar(t, i):
            # [FUNCTION DOC] HELPER: Adds the second argument to the tensor.
            return t + i

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the helper result for offsets
            # one to three.
            yield bar(t, 1)
            yield bar(t, 2)
            yield bar(t, 3)

        def fn(t):
            # [FUNCTION DOC] HELPER: Returns a zip of range three and the generator.
            return zip(range(3), whoo(t))

        t = torch.randn(3)
        y = self._compile_check(fn, args=(t,), fullgraph=False)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, list(y))

    def test_zip_subgenerator(self):
        # [FUNCTION DOC] Zip of a range and a generator that delegates to a sub-generator via
        # yield from; list of the lazily returned zip must match eager.
        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator yielding input plus one and plus two.
            yield t + 1
            yield t + 2

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator and then
            # yielding plus three.
            yield from subgen(t)
            yield t + 3

        def fn(t):
            # [FUNCTION DOC] HELPER: Returns a zip of range three and the generator and a sine.
            return zip(range(3), whoo(t)), t.sin()

        t = torch.randn(2)
        z, _ = self._compile_check(fn, args=(t,))
        self.assertEqual(list(z), list(zip(range(3), whoo(t))))

    def test_list_zip_generator(self):
        # [FUNCTION DOC] Listing a zip of range and a generator inside a full-graph function
        # gives the same list as eager.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the input plus one, two, three.
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the zip as a list.
            return list(zip(range(3), whoo(t)))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, y)

    def test_zip_infinite_generator(self):
        # [FUNCTION DOC] Same as the zip list test but with an infinite generator, bounded by
        # the range of three.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Infinite generator yielding the tensor plus
            # a growing integer.
            i = 0
            while True:
                yield t + i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the zip as a list.
            return list(zip(range(3), whoo(t)))

        t = torch.randn(3)
        y = fn(t)
        expected = list(zip(range(3), whoo(t)))
        self.assertEqual(expected, y)

    @parametrize("container", [list, tuple, dict, OrderedDict])
    def test_dict_tuple_list_generator(self, container):
        # [FUNCTION DOC] Parametrized over list, tuple, dict and OrderedDict containers:
        # builds the container from a generator of key-value pairs under a full-graph compile
        # and compares to the eager result.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator of three (integer, tensor) pairs.
            yield 1, t + 1
            yield 2, t + 2
            yield 3, t + 3

        def fn(t):
            # [FUNCTION DOC] HELPER: Builds the container under test from a generator.
            gen = whoo(t)
            return container(gen)

        t = torch.randn(2)
        expected = fn(t)
        got = torch.compile(backend="eager", fullgraph=True)(fn)(t)
        self.assertEqual(expected, got)

    def test_return_generator(self):
        # [FUNCTION DOC] Returns a fresh generator from a full-graph compiled function; the
        # caller drains it and gets the three expected tensors.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the input plus one, two, three.
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body creating and returning the generator.
            gen = whoo(t)
            return gen

        t = torch.tensor([1.0])
        gen = fn(t)
        self.assertEqual(list(gen), [t + 1, t + 2, t + 3])

    def test_return_tuple_generator(self):
        # [FUNCTION DOC] Returns a tuple of two generators (one built from the shifted
        # tensor) with a sine; each generator drains to the expected values.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the sine and cosine of its argument.
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body creating two generators and returning them
            # nested in a tuple alongside a sine.
            g1, g2 = whoo(t), whoo(t + 1)
            return (g1, g2), t.sin()

        t = torch.randn(2)
        (g1, g2), _ = fn(t)
        self.assertEqual(list(g1), [t.sin(), t.cos()])
        self.assertEqual(list(g2), [(t + 1).sin(), (t + 1).cos()])

    def test_return_advanced_generator(self):
        # [FUNCTION DOC] Returns a generator advanced once inside the compiled function; the
        # caller drains the remaining two values.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the input plus one, two, three.
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator once
            # and returning it.
            gen = whoo(t)
            next(gen)
            return gen

        t = torch.tensor([1.0])
        gen = fn(t)
        self.assertEqual(list(gen), [t + 2, t + 3])

    def test_return_exhaust_generator(self):
        # [FUNCTION DOC] A generator exhausted inside the compiled function and returned:
        # calling next on it afterwards raises StopIteration.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding a single value.
            yield t + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body exhausting the generator and returning it.
            gen = whoo(t)
            next(gen)
            return gen

        t = torch.tensor([1.0])
        gen = fn(t)
        with self.assertRaises(StopIteration):
            next(gen)

    @unittest.expectedFailure
    def test_reconstruct_generator_tensor_mutation(self):
        # [FUNCTION DOC] Marked expected failure: a generator that mutates the tensor in place
        # and is returned should raise Unsupported mentioning variable mutations.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding results of in-place sine and cosine.
            yield t.sin_()
            yield t.cos_()

        def fn(t):
            # [FUNCTION DOC] HELPER: Creates and returns the generator.
            gen = whoo(t)
            return gen

        with self.assertRaisesRegex(
            Unsupported,
            "Cannot reconstruct a generator with variable mutations",
        ):
            self._compile_check(fn)

    def test_subgenerator(self):
        # [FUNCTION DOC] Generator delegating to a sub-generator with yield from, drained into
        # a list inside a full-graph function; yields three values.
        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator yielding input plus one and two.
            yield t + 1
            yield t + 2

        def main_gen(t):
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator then
            # yielding plus three.
            yield from subgen(t)
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the main generator.
            gen = main_gen(t)
            return list(gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3])

    def test_return_subgenerator(self):
        # [FUNCTION DOC] Returns a partially advanced generator that delegates to a
        # sub-generator; the caller sees the remaining two values.
        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator yielding input plus one and two.
            yield t + 1
            yield t + 2

        def main_gen(t):
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator then
            # yielding plus three.
            yield from subgen(t)
            yield t + 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator once
            # and returning it.
            gen = main_gen(t)
            next(gen)
            return gen

        t = torch.randn(2)
        gen = fn(t)
        self.assertEqual(list(gen), [t + 2, t + 3])

    def test_dynamo_disable_generator(self):
        # [FUNCTION DOC] A generator function wrapped with the Dynamo disable decorator is
        # called from compiled code (graph breaks allowed) and drained to a list.
        @torch._dynamo.disable
        def main_gen(t):
            # [FUNCTION DOC] HELPER: Dynamo-disabled generator yielding input plus
            # one, two, three.
            yield t + 1
            yield t + 2
            yield t + 3

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the disabled generator.
            gen = main_gen(t)
            return list(gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3])

    def test_dynamo_disable_sub_generator(self):
        # [FUNCTION DOC] A Dynamo-disabled sub-generator is delegated to from a normal
        # generator via yield from; the whole list matches expected values.
        @torch._dynamo.disable
        def subgen(t):
            # [FUNCTION DOC] HELPER: Dynamo-disabled sub-generator yielding input plus
            # two and three.
            yield t + 2
            yield t + 3

        def main_gen(t):
            # [FUNCTION DOC] HELPER: Generator yielding input plus one, then delegating
            # to the sub-generator.
            yield t + 1
            yield from subgen(t)

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the generator.
            gen = main_gen(t)
            return list(gen)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, [t + 1, t + 2, t + 3])

    def test_graph_break_outside_generator(self):
        # [FUNCTION DOC] Explicit graph break in the caller between two generator advances;
        # the sum of the two yielded values is correct.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding input plus one and plus two.
            yield t + 1
            yield t + 2

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing the generator, graph
            # breaking, advancing again.
            gen = whoo(t)
            x = next(gen)
            torch._dynamo.graph_break()
            y = next(gen)
            return x + y

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, (t + 1) + (t + 2))

    def test_graph_break_before_calling_generator(self):
        # [FUNCTION DOC] Generator built from itertools product and permutations used as an
        # input to another product inside a function compiled with graph breaks allowed;
        # the accumulated sum must match the eager sum.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding the sum of the
            # first permutation component.
            for perm in itertools.product(itertools.permutations((0, 1, 2)), repeat=1):
                yield sum(perm[0])

        def fn(t):
            # [FUNCTION DOC] HELPER: Loops over a product of the generator and permutations of
            # four and five, accumulating the generator values.
            s = 0
            for b, p in itertools.product(whoo(t), itertools.permutations((4, 5))):
                s += b
            return s

        t = torch.randn(2)
        expected = fn(t)
        got = torch.compile(backend="eager", fullgraph=False)(fn)(t)
        self.assertEqual(expected, got)

    def test_generator_with_side_effects(self):
        # [FUNCTION DOC] Generator that increments an outer nonlocal counter is returned
        # from compiled code (no full graph). Reconstruction is rejected, so the unimplemented
        # counter has one entry with a variable mutations message, counted once.
        counters.clear()
        i = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator incrementing an outer counter before each
            # of five yields.
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j

        @torch.compile(backend="eager")
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the generator and a sine.
            return whoo(t), t.sin()

        t = torch.randn(2)
        fn(t)
        self.assertEqual(len(counters["unimplemented"]), 1)
        entry = next(iter(counters["unimplemented"].items()))
        self.assertIn(
            "Cannot reconstruct a generator with variable mutations.", entry[0]
        )
        self.assertEqual(entry[1], 1)

    def test_subgenerator_with_side_effects(self):
        # [FUNCTION DOC] A sub-generator and its delegating generator both mutate an outer
        # counter; the generator returned from compiled code still yields five values, and
        # the unimplemented counters contain a message about mutated generator reconstruction.
        i = 0

        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator that increments an outer counter and
            # yields two values.
            nonlocal i
            i += 1
            yield t
            i += 1
            yield t + 1

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator, then
            # yielding three more
            # values with the counter incremented around them.
            nonlocal i
            yield from subgen(t)
            i += 1
            yield t + 2
            i += 1
            yield t + 3
            i += 1
            yield t + 4

        @torch.compile(backend="eager")
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the generator and a sine.
            return whoo(t), t.sin()

        t = torch.randn(2)
        gen, y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(len(list(gen)), 5)
        for gb in counters["unimplemented"]:
            if "Cannot reconstruct a generator with variable mutations." in gb:
                break
        else:
            self.assertTrue(False, "expected side effect error; not found")

    def test_generator_with_side_effects_graph_break(self):
        # [FUNCTION DOC] Mutating generator created, graph break, then advanced once and
        # returned; remaining four values are correct and the unimplemented counters contain
        # a combined reconstruction-with-mutations message.
        i = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator incrementing an outer counter before each
            # of five yields.
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j

        @torch.compile(backend="eager", fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body creating the generator,
            # graph breaking, advancing
            # once and returning the generator and a sine.
            gen = whoo(t)
            torch._dynamo.graph_break()
            next(gen)
            return gen, t.sin()

        t = torch.randn(2)
        gen, y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(len(list(gen)), 4)
        found = any(
            "Generator reconstruction with mutations" in msg
            and "Cannot reconstruct a generator with variable mutations" in msg
            for msg in counters["unimplemented"]
        )
        self.assertTrue(found)

    def test_generator_with_side_effects_graph_break_2(self):
        # [FUNCTION DOC] Mutating generator with a graph break after each yield, consumed by
        # a zip bounded by a range of three; no graphs are recorded.
        i = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator incrementing an outer counter, yielding, then
            # graph breaking, in a loop.
            nonlocal i
            for j in range(5):
                i += 1
                yield t + j
                torch._dynamo.graph_break()

        eager = EagerAndRecordGraphs()

        @torch.compile(backend=eager, fullgraph=False)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body returning the list of the zip of
            # range and generator.
            gen = whoo(t)
            return list(zip(range(3), gen))

        t = torch.randn(2)
        fn(t)
        self.assertEqual(len(eager.graphs), 0)

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw(self):
        # [FUNCTION DOC] Python 3.12+ only. Throws a StopIteration into an outer generator
        # while it is delegating to a nested generator that catches it and returns 123; the
        # compiled result must be 3 (the next outer yield), matching eager.
        def nested_generator():
            # [FUNCTION DOC] HELPER: Nested generator yielding two values and returning 123 if
            # StopIteration is thrown in.
            try:
                yield 1
                yield 2
            except StopIteration:
                return 123  # noqa: B901

        def outer_generator():
            # [FUNCTION DOC] HELPER: Generator delegating to the nested generator,
            # then yielding three.
            yield from nested_generator()
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the outer generator, throwing
            # StopIteration into it,
            # ignoring a RuntimeError and returning the thrown-call result with a sine.
            gen = outer_generator()
            next(gen)  # Start the outer generator and enter the nested generato

            i = 0
            try:
                # Force an exception while the generator is running
                i = gen.throw(StopIteration("stop"))
            except RuntimeError:
                pass
            return (i, t.sin())

        t = torch.randn(2)
        i, y = self._compile_check(fn, args=(t,))
        self.assertEqual(i, 3)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw_custom_StopIteration(self):
        # [FUNCTION DOC] Python 3.12+ only. Same as the CLEANUP_THROW test but throwing an
        # instance of a StopIteration subclass; result is still 3.
        class MyStopIteration(StopIteration):
            # [CLASS DOC] HELPER: Custom subclass of StopIteration with no extra behavior.
            pass

        def nested_generator():
            # [FUNCTION DOC] HELPER: Nested generator yielding two values and returning 123 if
            # StopIteration is thrown in.
            try:
                yield 1
                yield 2
            except StopIteration:
                return 123  # noqa: B901

        def outer_generator():
            # [FUNCTION DOC] HELPER: Generator delegating to the nested generator,
            # then yielding three.
            yield from nested_generator()
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body throwing the custom StopIteration into the
            # started outer generator.
            gen = outer_generator()
            next(gen)  # Start the outer generator and enter the nested generato

            i = 0
            try:
                # Force an exception while the generator is running
                i = gen.throw(MyStopIteration("stop"))
            except RuntimeError:
                pass
            return (i, t.sin())

        t = torch.randn(2)
        i, y = self._compile_check(fn, args=(t,))
        self.assertEqual(i, 3)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw_subgen_return_value(self):
        # [FUNCTION DOC] Python 3.12+ only. Throwing StopIteration into a delegating generator
        # whose nested generator catches it and returns 123: the yield from expression should
        # become 123 and be yielded next. The compiled result is asserted to be 123.
        # CLEANUP_THROW must resume the delegating generator with the
        # subgenerator's return value (StopIteration.value). When the
        # subgenerator catches the thrown StopIteration and returns, the
        # yield-from expression must evaluate to that returned value. Dynamo
        # currently extracts the value of the *thrown* exception instead.
        def nested_generator():
            # [FUNCTION DOC] HELPER: Nested generator yielding two values and returning
            # 123 on StopIteration.
            try:
                yield 1
                yield 2
            except StopIteration:
                return 123  # noqa: B901

        def outer_generator():
            # [FUNCTION DOC] HELPER: Generator capturing the yield from result and yielding it.
            r = yield from nested_generator()
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the outer generator and throwing
            # StopIteration into it.
            gen = outer_generator()
            next(gen)
            i = gen.throw(StopIteration("stop"))
            return (i, t.sin())

        t = torch.randn(2)
        i, y = fn(t)
        self.assertEqual(i, 123)
        self.assertEqual(y, t.sin())

    @unittest.skipIf(sys.version_info < (3, 12), "Test CLEANUP_THROW")
    def test_cleanup_throw_empty_stopiteration(self):
        # [FUNCTION DOC] Python 3.12+ only. Throws a bare StopIteration into a delegating
        # generator whose nested generator does not catch it; a RuntimeError (PEP 479
        # conversion) is expected from the compiled call.
        def nested_generator():
            # [FUNCTION DOC] HELPER: Nested generator yielding one and two with
            # no exception handling.
            yield 1
            yield 2

        def outer_generator():
            # [FUNCTION DOC] HELPER: Generator delegating to the nested generator
            # then yielding three.
            yield from nested_generator()
            yield 3

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the generator then throwing
            # StopIteration into it.
            gen = outer_generator()
            next(gen)
            gen.throw(StopIteration())
            return t.sin()

        t = torch.randn(2)
        with self.assertRaises(RuntimeError):
            fn(t)

    def test_iter(self):
        # [FUNCTION DOC] Iterating over an infinite generator with a for loop and a break
        # inside a full-graph function; sums values 0 through 5 and adds to the input.
        def whoo():
            # [FUNCTION DOC] HELPER: Infinite generator counting from zero.
            i = 0
            while True:
                yield i
                i += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body summing generator values until
            # one exceeds five.
            s = 0
            for i in whoo():
                if i > 5:
                    break
                s += i
            return t + s

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + sum(range(6)))

    def test_list_extend(self):
        # [FUNCTION DOC] Extending a list with a generator expression inside a full-graph
        # compiled function matches the eager result.
        def f(x):
            # [FUNCTION DOC] HELPER: Builds a list, extends it from a generator expression, and
            # returns the input plus one and the list.
            y = [1]
            y.extend(y[-1] + z for z in range(3))
            return x + 1, y

        self.assertEqual(
            f(torch.ones(3)),
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(3)),
        )

    def test_deque_extendleft(self):
        # [FUNCTION DOC] Extending a deque on the left with a generator expression. The
        # result is compared with the eager output.
        import collections

        def f(x):
            # [FUNCTION DOC] HELPER: Builds a deque, calls extendleft with a generator
            # expression and returns the input plus one with the deque.
            y = collections.deque([1])
            y.extendleft(y[0] + z for z in range(3))
            return x + 1, y

        self.assertEqual(
            f(torch.ones(3)),
            torch.compile(f, backend="eager", fullgraph=True)(torch.ones(3)),
        )

    @make_dynamo_test
    def test_generator___contains__(self):
        # [FUNCTION DOC] Runs under the make_dynamo_test decorator: membership tests on a
        # generator consume items, then next raises StopIteration; a fresh
        # generator lacks three.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding one and two.
            yield 1
            yield 2

        g = whoo()
        self.assertTrue(1 in g)
        self.assertTrue(2 in g)
        self.assertRaises(StopIteration, next, g)
        self.assertFalse(3 in whoo())

    @make_dynamo_test
    def test_generator___contains___side_effects(self):
        # [FUNCTION DOC] Membership tests on a generator run its body, so the outer counter
        # changes to one and then two as items are scanned; afterwards the
        # generator is exhausted.
        n = 0

        def whoo():
            # [FUNCTION DOC] HELPER: Generator that records progress in an outer variable before
            # each of two yields.
            nonlocal n
            n = 1
            yield 1
            n = 2
            yield 2

        g = whoo()
        self.assertTrue(1 in g)
        self.assertEqual(n, 1)
        self.assertTrue(2 in g)
        self.assertEqual(n, 2)
        self.assertRaises(StopIteration, next, g)
        self.assertFalse(3 in whoo())

    def test_raise_immediately(self):
        # [FUNCTION DOC] Generator expression over a non-iterable int in a full-graph function
        # raises Unsupported mentioning the object is not iterable.
        # see https://github.com/python/cpython/issues/143493
        @torch.compile(fullgraph=True, backend="eager")
        def f(s):
            # [FUNCTION DOC] HELPER: Compiled body returning a generator expression
            # over its argument.
            return (x for x in s)

        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "'int' object is not iterable"
        ):
            f(1)

    def test_pep479_raise_stopiteration_in_body(self):
        # [FUNCTION DOC] PEP 479: a StopIteration raised in a generator body becomes
        # RuntimeError on the second advance; the result encodes which exception arrived and
        # must equal the RuntimeError branch (input plus one).
        # PEP 479: a StopIteration that escapes a generator body is converted
        # to RuntimeError. On 3.12+ the compiler emits CALL_INTRINSIC_1 3 for
        # this; on earlier versions Dynamo converts at the generator frame
        # boundary. The branch taken (RuntimeError vs StopIteration) is encoded
        # in the returned tensor so the eager/compiled results can be compared.
        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a generator raising StopIteration after one yield,
            # advances it twice
            # and returns a tensor depending on which exception type was caught.
            def whoo():
                # [FUNCTION DOC] HELPER: Generator yielding once then raising StopIteration.
                yield t + 1
                raise StopIteration("boom")

            g = whoo()
            next(g)
            try:
                next(g)
            except RuntimeError:
                return t + 1.0
            except StopIteration:
                return t + 2.0
            return t

        t = torch.randn(2)
        self.assertEqual(self._compile_check(fn, args=(t,)), t + 1.0)

    @make_dynamo_test
    def test_pep479_stopiteration_error(self):
        # [FUNCTION DOC] Ported from CPython: advancing a generator whose body raises
        # StopIteration raises RuntimeError whose message mentions raising StopIteration.
        # Ported from CPython test_generators ExceptionTest.test_stopiteration_error.
        # Asserts the RuntimeError message, not just the type.
        def gen():
            # [FUNCTION DOC] HELPER: Generator that raises StopIteration
            # immediately (never yields).
            raise StopIteration
            yield

        with self.assertRaisesRegex(RuntimeError, "raised StopIteration"):
            next(gen())

    @make_dynamo_test
    def test_pep479_tutorial_stopiteration(self):
        # [FUNCTION DOC] Ported from CPython: first next gives one, second raises RuntimeError
        # with a StopIteration message because the body raises StopIteration.
        # Ported from CPython test_generators ExceptionTest.test_tutorial_stopiteration.
        def f():
            # [FUNCTION DOC] HELPER: Generator yielding one then raising StopIteration.
            yield 1
            raise StopIteration
            yield 2  # never reached

        g = f()
        self.assertEqual(next(g), 1)
        with self.assertRaisesRegex(RuntimeError, "raised StopIteration"):
            next(g)

    def test_pep479_stopiteration_from_inner_next(self):
        # [FUNCTION DOC] A StopIteration leaking from an inner exhausted iterator inside a
        # generator body is also converted to RuntimeError.
        # Tutorial PEP 479 case: StopIteration leaking from an inner next()
        # inside the generator body is also converted to RuntimeError.
        def fn(t):
            # [FUNCTION DOC] HELPER: Builds the generators, advances twice, and returns a tensor
            # encoding the caught exception type.
            def inner():
                # [FUNCTION DOC] HELPER: Inner generator yielding a single value.
                yield t + 1

            def whoo():
                # [FUNCTION DOC] HELPER: Generator looping forever and yielding items from an
                # inner generator, so inner exhaustion leaks StopIteration.
                it = inner()
                while True:
                    yield next(it)

            g = whoo()
            next(g)
            try:
                next(g)
            except RuntimeError:
                return t + 1.0
            except StopIteration:
                return t + 2.0
            return t

        t = torch.randn(2)
        self.assertEqual(self._compile_check(fn, args=(t,)), t + 1.0)

    @unittest.skipIf(
        sys.version_info >= (3, 12),
        "pre-3.12 converts StopIteration at the generator frame boundary",
    )
    @make_dynamo_test
    def test_pep479_exception_stack_balanced(self):
        # [FUNCTION DOC] Only Python before 3.12. After the StopIteration to RuntimeError
        # conversion, a bare re-raise must report no active exception, proving the
        # exception stack is balanced.
        # The pre-3.12 StopIteration -> RuntimeError conversion must not leave
        # the StopIteration on the exception stack. Otherwise a later bare
        # `raise` re-raises the leftover StopIteration instead of reporting that
        # there is no active exception.
        def gen():
            # [FUNCTION DOC] HELPER: Generator yielding one then raising StopIteration.
            yield 1
            raise StopIteration

        g = gen()
        next(g)
        try:
            next(g)  # StopIteration escapes the body -> RuntimeError
        except RuntimeError:
            pass
        with self.assertRaisesRegex(RuntimeError, "No active exception to reraise"):
            raise  # noqa: PLE0704


class TestGeneratorSend(GeneratorTestsBase):
    # [CLASS DOC] Tests for sending values into generators and yield from return values.
    def test_send(self):
        # [FUNCTION DOC] Sends the tensor into a generator that expects it after a first
        # advance; the compiled result is double the input.
        def double():
            # [FUNCTION DOC] HELPER: Generator that receives a value and yields
            # twice that value.
            x = yield
            yield x * 2

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body priming the generator and sending the input.
            gen = double()
            next(gen)
            return gen.send(t)

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t * 2)

    @parametrize("fullgraph", [True, False])
    def test_send_stop_iteration(self, fullgraph):
        # [FUNCTION DOC] Parametrized by full graph flag. Sending twice into a generator that
        # finished: full graph raises Unsupported about an observed exception, otherwise
        # StopIteration propagates.
        def double():
            # [FUNCTION DOC] HELPER: Generator that receives a value and yields
            # twice that value.
            x = yield
            yield x * 2

        @torch.compile(backend="eager", fullgraph=fullgraph)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body priming, sending twice and adding results.
            gen = double()
            next(gen)
            a = gen.send(t)
            b = gen.send(t)  # should result in StopIteration
            return a + b

        t = torch.randn(2)
        if fullgraph:
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                fn(t)
        else:
            with self.assertRaises(StopIteration):
                fn(t)

    def test_yield_from_return_value(self):
        # [FUNCTION DOC] The value of a yield from expression equals the sub-generator return
        # value; result list is sine then cosine.
        # `yield from` evaluates to the subgenerator's return value, which the
        # SEND opcode extracts from the StopIteration raised on completion.
        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator yielding sine and returning cosine.
            yield t.sin()
            return t.cos()  # noqa: B901

        def outer(t):
            # [FUNCTION DOC] HELPER: Generator capturing the yield from value and yielding it.
            r = yield from subgen(t)
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the outer generator.
            return list(outer(t))

        t = torch.randn(2)
        self.assertEqual(fn(t), [t.sin(), t.cos()])

    def test_yield_from_return_none(self):
        # [FUNCTION DOC] A sub-generator without a return value makes the yield
        # from result None.
        # A subgenerator that falls off the end returns None (StopIteration
        # with no value); `yield from` must evaluate to None.
        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator yielding only the sine.
            yield t.sin()

        def outer(t):
            # [FUNCTION DOC] HELPER: Generator capturing the yield from value and yielding it.
            r = yield from subgen(t)
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the outer generator.
            return list(outer(t))

        t = torch.randn(2)
        out = fn(t)
        self.assertEqual(out[0], t.sin())
        self.assertIsNone(out[1])

    def test_send_through_yield_from(self):
        # [FUNCTION DOC] Values sent to the delegating generator are forwarded to the
        # sub-generator; sequence of sent values yields 10, 8 and 300.
        # Values sent into the delegating generator are forwarded to the
        # subgenerator: SEND with a non-None value routes through `send`.
        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator receiving two values, yielding ten and
            # received plus one, and returning 100 times the second.
            x = yield 10
            y = yield x + 1
            return y * 100  # noqa: B901

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator and yielding
            # its return value.
            r = yield from subgen()
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body sending None, seven and three, returning the
            # three results and a sine.
            gen = outer()
            a = gen.send(None)
            b = gen.send(7)
            c = gen.send(3)
            return [a, b, c, t.sin()]

        t = torch.randn(2)
        out = fn(t)
        self.assertEqual(out[:3], [10, 8, 300])
        self.assertEqual(out[3], t.sin())

    def test_nested_yield_from_return_value(self):
        # [FUNCTION DOC] Return values propagate through two levels of yield from; final
        # yielded value is cosine plus one.
        # Return values propagate through a chain of `yield from` delegations.
        def leaf(t):
            # [FUNCTION DOC] HELPER: Leaf generator yielding sine and returning cosine.
            yield t.sin()
            return t.cos()  # noqa: B901

        def mid(t):
            # [FUNCTION DOC] HELPER: Middle generator returning the leaf result plus one.
            r = yield from leaf(t)
            return r + 1  # noqa: B901

        def top(t):
            # [FUNCTION DOC] HELPER: Top generator yielding the middle result.
            r = yield from mid(t)
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the top generator.
            return list(top(t))

        t = torch.randn(2)
        self.assertEqual(fn(t), [t.sin(), t.cos() + 1])

    def test_yield_from_iterable_return_none(self):
        # [FUNCTION DOC] Yield from over a plain list yields its elements and
        # evaluates to None.
        # `yield from` over a non-generator iterable yields its items via the
        # SEND tp_iternext path and evaluates to None.
        def outer(t):
            # [FUNCTION DOC] HELPER: Generator delegating to a list and yielding the result.
            r = yield from [t.sin(), t.cos()]
            yield r

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body listing the generator.
            return list(outer(t))

        t = torch.randn(2)
        out = fn(t)
        self.assertEqual(out[0], t.sin())
        self.assertEqual(out[1], t.cos())
        self.assertIsNone(out[2])


class TestGeneratorClose(GeneratorTestsBase):
    # [CLASS DOC] Tests for calling close on generators under compile: finally blocks,
    # GeneratorExit handling, sub-generators, repeated close and HOP interplay.
    def test_close(self):
        # [FUNCTION DOC] Closing a started generator in full graph mode; the value from the
        # first advance is returned.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding sine and cosine.
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once then closing the generator.
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())

    def test_close_subgen(self):
        # [FUNCTION DOC] Closing a generator delegating to a sub-generator; the sub-generator
        # had set an outer variable to one before the close and does not progress further.
        z = 0

        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator setting an outer variable
            # then yielding twice.
            nonlocal z
            z = 1
            yield t.sin()
            z = 3
            yield t.cos()

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator
            # then yielding tangent.
            yield from subgen(t)
            yield t.tan()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once and closing the generator.
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(z, 1)

    def test_close_with_side_effects(self):
        # [FUNCTION DOC] Closing runs the generator finally block, which reads an outer
        # variable updated just before the close; list records [1, -123, 2].
        L = []
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator appending to an outer list around yields, with a
            # finally appending an outer variable.
            nonlocal z
            try:
                L.append(1)
                yield t.sin()
                L.append(2)
                yield t.cos()
            finally:
                L.append(z)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing, changing the outer variable,
            # closing, and appending the list length.
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            z = -123
            gen.close()
            L.append(len(L))
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(L, [1, -123, 2])

    def test_close_capture_GeneratorExit_return(self):
        # [FUNCTION DOC] A generator catching GeneratorExit and returning a value: close
        # returns that value; counters show except and finally both ran (111).
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator updating an outer counter in try, except
            # GeneratorExit (returning tangent) and finally.
            nonlocal z
            try:
                z += 1
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                z += 10
                return t.tan()  # noqa: B901
            finally:
                z += 100

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once and returning the first value
            # with the close result.
            gen = whoo(t)
            i = next(gen)
            y = gen.close()
            return (i, y)

        t = torch.randn(2)
        (i, y) = fn(t)
        self.assertEqual(i, t.sin())
        self.assertEqual(y, t.tan())
        self.assertEqual(z, 111)

    @parametrize("fullgraph", [True, False])
    def test_close_capture_GeneratorExit(self, fullgraph):
        # [FUNCTION DOC] Parametrized by full graph flag. A generator that yields again after
        # GeneratorExit: full graph raises Unsupported about an observed exception; otherwise
        # RuntimeError about ignoring GeneratorExit.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator that yields tangent when closed and sets an outer
            # variable in finally.
            nonlocal z
            try:
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                yield t.tan()
            finally:
                z = 1

        @torch.compile(backend="eager", fullgraph=fullgraph)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once then closing the generator.
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)
        if fullgraph:
            # This should actually be RuntimeError("generator ignored GeneratorExit")
            # but Dynamo swallow the exception and raises Unsupported instead
            with self.assertRaisesRegex(Unsupported, "Observed exception"):
                fn(t)
        else:
            with self.assertRaisesRegex(
                RuntimeError, "generator ignored GeneratorExit"
            ):
                fn(t)

    def test_close_capture_and_reraise_GeneratorExit(self):
        # [FUNCTION DOC] Generator catches GeneratorExit and re-raises it; the log shows the
        # ordering of except and finally writes: [1, -123, -1, 456].
        L = []
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator recording to an outer list and variable, with
            # except re-raising and a finally.
            nonlocal z
            try:
                L.append(1)
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                L.append(z)
                z = -1
                raise
            finally:
                L.append(z)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing, changing an outer variable,
            # closing, appending a marker.
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            z = -123
            gen.close()
            L.append(456)
            return i

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(L, [1, -123, -1, 456])

    @parametrize("exc", [RuntimeError, AttributeError])
    @make_dynamo_test
    def test_close_capture_and_reraise_exc(self, exc):
        # [FUNCTION DOC] Parametrized over RuntimeError and AttributeError. Runs under the
        # dynamo test decorator: GeneratorExit is converted into another exception chained
        # from it; close propagates that exception, which the test catches and checks.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator converting GeneratorExit into the
            # parametrized exception type.
            try:
                yield t.sin()
                yield t.cos()
            except GeneratorExit as e:
                raise exc from e
            finally:
                pass

        def fn(t):
            # [FUNCTION DOC] HELPER: Advances the generator once, closes it and returns
            # the first value.
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return i

        t = torch.randn(2)

        z = 0
        try:
            fn(t)
        except exc:
            z = 1
        finally:
            assert z == 1  # noqa: S101

    def test_close_with_subgen(self):
        # [FUNCTION DOC] Closing while the generator is still inside the delegated
        # sub-generator: the generator body past the delegation never runs, finally is not
        # reached; the log holds [10, 456] and the outer variable is -123.
        L = []
        z = 0

        def subgen(t):
            # [FUNCTION DOC] HELPER: Sub-generator yielding sine and cosine.
            yield t.sin()
            yield t.cos()

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator that logs, delegates, and later has a try with
            # finally logging an outer variable.
            nonlocal z
            L.append(10)
            yield from subgen(t)
            L.append(20)
            try:
                L.append(1)
                z = 4
                yield t.tan()
            finally:
                L.append(z)

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once, changing the outer variable,
            # closing, logging and returning a pair.
            nonlocal z
            gen = whoo(t)
            i = next(gen)
            z = -123
            gen.close()
            L.append(456)
            return i, t.sin()

        t = torch.randn(2)
        y, _ = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(L, [10, 456])
        self.assertEqual(z, -123)

    def test_close_after_close(self):
        # [FUNCTION DOC] Calling close twice: finally runs only once, second close gives None
        # and the counter ends at 2.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator counting in the try body and finally
            # around two yields.
            nonlocal z
            try:
                z += 1
                yield t.sin()
                yield t.cos()
            finally:
                # finally should only be executed once
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once, closing, and returning the
            # first value with a second close result.
            gen = whoo(t)
            i = next(gen)
            gen.close()
            return (i, gen.close())

        t = torch.randn(2)
        (i, y) = fn(t)
        self.assertEqual(i, t.sin())
        self.assertEqual(y, None)
        self.assertEqual(z, 2)

    @parametrize("fullgraph", [True, False])
    def test_next_after_close(self, fullgraph):
        # [FUNCTION DOC] Parametrized by full graph flag. Advancing a generator after it was
        # closed: full graph raises Unsupported, otherwise StopIteration.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding sine and cosine.
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=fullgraph)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body closing an unstarted generator
            # then calling next.
            gen = whoo(t)
            gen.close()
            a = next(gen)
            return [t.sin(), a]

        t = torch.randn(3)
        if fullgraph:
            with self.assertRaises(Unsupported):
                fn(t)
        else:
            with self.assertRaises(StopIteration):
                fn(t)

    def test_close_after_exception(self):
        # [FUNCTION DOC] Closing a generator that already raised ValueError on its first
        # advance returns None.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator raising ValueError before its first yield.
            raise ValueError("foo")
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body swallowing the ValueError then closing and
            # returning a sine with the close result.
            gen = whoo(t)
            try:
                next(gen)
            except ValueError:
                pass
            b = gen.close()
            return [t.sin(), b]

        t = torch.randn(2)
        y, b = fn(t)
        self.assertEqual(y, t.sin())
        self.assertIsNone(b)

    def test_close_handling_finally(self):
        # [FUNCTION DOC] Generator catching GeneratorExit and returning tangent inside both
        # except and finally: close returns the tangent, counter reaches two.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator with try yielding twice, except returning
            # tangent, and finally; both increment an outer counter.
            nonlocal z
            try:
                yield t.sin()
                yield t.cos()
            except GeneratorExit:
                z += 1
                return t.tan()  # noqa: B901
            finally:
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing once and returning a sine with
            # the close result.
            gen = whoo(t)
            next(gen)
            b = gen.close()
            return t.sin(), b

        t = torch.randn(2)
        y, b = fn(t)
        self.assertEqual(y, t.sin())
        self.assertEqual(b, t.tan())
        self.assertEqual(z, 2)

    def test_untrack_generator_after_hop(self):
        # [FUNCTION DOC] Regression test: an open generator created before a conditional
        # higher-order op is exhausted after it. Untracking must use the generator list
        # owned by the output graph, so no ValueError occurs; the sum equals
        # the expected value.
        # Regression: speculate_subgraph clones SideEffects and leaves the clone
        # as the live instance, so the set of open generators must be owned by
        # OutputGraph, not SideEffects. torch.cond's branch speculation swaps in
        # a clone (which never carried local_generators); exhausting a generator
        # created before the cond then untracked it on the clone, raising
        # "ValueError: list.remove(x): x not in list".
        # See https://github.com/pytorch/pytorch/pull/157149
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding sine and cosine.
            yield t.sin()
            yield t.cos()

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body that advances a generator, runs torch cond,
            # then exhausts the generator and sums.
            gen = whoo(t)
            a = next(gen)
            # cond speculates both branches, swapping SideEffects to a clone
            b = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
            acc = a + b
            # Exhaust the generator after the swap: untrack must not crash
            for x in gen:
                acc = acc + x
            return acc

        t = torch.randn(2)
        y = fn(t)
        ref = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
        self.assertEqual(y, t.sin() + ref + t.cos())

    def test_close_open_generator_after_hop(self):
        # [FUNCTION DOC] An open generator created before a higher-order op must still be
        # closed at graph compile time so its finally block runs.
        # An open (non-exhausted) generator created before a HOP must still be
        # closed at compile_subgraph time after SideEffects is swapped, so its
        # finally block runs. Reads the tracking list from OutputGraph.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding sine and cosine whose finally block
            # increments an outer counter.
            nonlocal z
            try:
                yield t.sin()
                yield t.cos()
            finally:
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body advancing a generator once, running a torch
            # cond, and returning the sum, leaving the generator open.
            gen = whoo(t)
            a = next(gen)
            b = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
            return a + b  # gen left open; finally must run via close

        t = torch.randn(2)
        y = fn(t)
        ref = torch.cond(t.sum() > 0, lambda x: x + 1, lambda x: x - 1, (t,))
        self.assertEqual(y, t.sin() + ref)
        self.assertEqual(z, 1)

    def test_close_open_generator_fast_path(self):
        # [FUNCTION DOC] An open generator with a pending finally block must be closed when
        # the graph is compiled even if the frame qualifies for the fast path; the function
        # returns its input tensor so side effects stay empty. After the call the outer
        # counter is one.
        # An open generator whose only pending work is a finally block must be
        # closed at compile_subgraph time even when the frame is eligible for
        # the fast path (single frame, all-tensor stack, empty side effects).
        # Returning an input tensor keeps side effects empty -- a freshly
        # produced tensor (e.g. t.sin()) is tracked as a new mutation and would
        # divert to the slow path, which closes generators unconditionally.
        # The fast-path guard `not self.local_generators` at output_graph.py is
        # what forces close_local_generators to run here so the finally fires.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding its argument twice with a finally block
            # incrementing an outer counter.
            nonlocal z
            try:
                yield t
                yield t
            finally:
                z += 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the generator and returning
            # the input tensor.
            gen = whoo(t)
            next(gen)  # start generator; finally now pending
            return t  # return input tensor -> fast-path eligible

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t)
        self.assertEqual(z, 1)

    def test_close_replays_new_object_attr_mutation(self):
        # [FUNCTION DOC] Checks that an attribute set on a new object created in a
        # generator frame survives dead-object pruning so closing the generator at frame
        # exit replays the finally block; compiled output (sine and the marker list) must
        # equal the eager result.
        # A generator closed at frame exit may, in its finally block, act on a
        # new object reachable only through the generator's frame. Its attribute
        # mutations (AttributeMutationNew) must survive prune_dead_object_new so
        # close() can replay the finally; otherwise the mutation is lost.
        class C:
            # [CLASS DOC] HELPER: Empty class used as a plain attribute container.
            pass

        def fn(t):
            # [FUNCTION DOC] HELPER: Defines a generator that stores a list on a new object and
            # appends to it in a finally block; starts it once and returns a sine with the list.
            marker = []

            def gen():
                # [FUNCTION DOC] HELPER: Generator creating an object, attaching the shared list
                # to it, yielding the sine, and in finally appending one to the list.
                obj = C()
                obj.data = marker  # AttributeMutationNew on a new object
                try:
                    yield t.sin()
                finally:
                    obj.data.append(1)  # replayed when the generator is closed

            g = gen()
            next(g)
            return t.sin(), marker

        t = torch.randn(3)
        self.assertEqual(fn(t), self._compile_check(fn, args=(t,)))

    def test_deep_tee_buffer_no_recursion_error(self):
        # [FUNCTION DOC] Regression test: tracing a function that fills a long itertools tee
        # buffer (2000 items, above the recursion limit) must not hit a recursion error during
        # variable tracker traversal.
        # VariableTracker.visit walks generator/iterator state during compile.
        # A long itertools.tee buffer is a deeply chained structure; visit uses
        # an explicit worklist so it must not overflow the Python stack.
        n = 2000  # exceeds the default recursion limit

        def fn(t):
            # [FUNCTION DOC] HELPER: Splits a range iterator with tee, drains one side, returns
            # the input plus one.
            a, b = itertools.tee(iter(range(n)))
            list(a)  # advance one side so the shared tee buffer grows to n
            return t + 1

        t = torch.randn(3)
        self._compile_check(fn, args=(t,))


class TestGeneratorThrow(GeneratorTestsBase):
    # [CLASS DOC] Tests for throwing exceptions into generators (try, except and finally
    # interactions, return values from StopIteration, error propagation).
    def test_throw(self):
        # [FUNCTION DOC] Throws RuntimeError into a generator that catches it and yields
        # cosine; result is sine plus cosine.
        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator yielding sine, with a handler for
            # RuntimeError yielding cosine.
            try:
                yield t.sin()
            except RuntimeError:
                yield t.cos()

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws RuntimeError into it and
            # sums both results.
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(y, t.sin() + t.cos())

    def test_throw_with_finally(self):
        # [FUNCTION DOC] Throws ValueError into a generator with nested try blocks; inner
        # finally and outer finally run in turn, so the outer counter ends at 3.
        z = 0

        def whoo():
            # [FUNCTION DOC] HELPER: Generator with nested try, except and finally blocks that
            # update an outer counter by different amounts.
            nonlocal z
            z = 0
            try:
                try:
                    yield 1
                except ValueError:
                    yield 2
                finally:
                    z += 2
            except ValueError:
                z += 33
                yield 4
            finally:
                z += 1
            z += 10

        def f(x):
            # [FUNCTION DOC] HELPER: Starts the generator, throws ValueError into it,
            # returns a sine.
            gen = whoo()
            next(gen)
            gen.throw(ValueError)
            return x.sin()

        self._compile_check(f)
        self.assertEqual(z, 3)

    def test_throw_without_finally(self):
        # [FUNCTION DOC] Throw into a generator without finally: the handler runs and yields
        # cosine; the outer counter ends at 101.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator with a RuntimeError handler yielding cosine,
            # updating an outer counter at each stage.
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
                z += 10
            except RuntimeError:
                z += 100
                yield t.cos()
                z += 1_000
            z += 10_000

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws RuntimeError, sums results.
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(y, t.sin() + t.cos())
        self.assertEqual(z, 101)

    def test_throw_no_yield_after_throw(self):
        # [FUNCTION DOC] The thrown exception is handled without a following yield, so the
        # generator finishes and throw raises StopIteration with no arguments; counter is 111
        # and the result is the sine.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator with a ValueError handler and a finally block
            # updating an outer counter.
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError:
                z += 10
            finally:
                z += 100

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws ValueError and
            # expects StopIteration with no arguments, returning the first value;
            # raises AssertionError otherwise.
            gen = whoo(t)
            a = next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                if len(e.args) > 0:
                    raise AssertionError(
                        "Expected StopIteration with no arguments"
                    ) from e
                return a
            raise AssertionError("Expected StopIteration")

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(z, 111)
        self.assertEqual(y, t.sin())

    def test_throw_not_catch(self):
        # [FUNCTION DOC] Throwing an exception type the generator does not handle propagates
        # the error out of a full-graph compiled function (RuntimeError).
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator handling only ValueError, with a finally block.
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError:
                z += 10
                yield t.cos()
            finally:
                z += 100

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the generator and throwing
            # RuntimeError into it.
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        with self.assertRaises(RuntimeError):
            fn(t)

    def test_throw_raise_difference_exc(self):
        # [FUNCTION DOC] A generator converts a thrown ValueError into a RuntimeError chained
        # from it; the compiled call raises RuntimeError.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator whose ValueError handler raises RuntimeError and
            # whose finally block updates an outer counter.
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError as e:
                z += 10
                raise RuntimeError from e
            finally:
                z += 100

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the generator
            # and throwing ValueError.
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(ValueError)
            return a + b

        t = torch.randn(2)
        with self.assertRaises(RuntimeError):
            fn(t)

    def test_throw_yield_finally(self):
        # [FUNCTION DOC] Generator yields inside a finally block after an exception is thrown;
        # the full-graph compiled call raises Unsupported.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator whose finally block yields tangent after
            # a RuntimeError handler.
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except RuntimeError:
                z += 10
                yield t.cos()
            finally:
                z += 100
                yield t.tan()  # RuntimeError: generator ignored GeneratorExit

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Compiled body starting the generator
            # and throwing RuntimeError.
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        with self.assertRaises(Unsupported):
            fn(t)

    def test_throw_try_except_finally(self):
        # [FUNCTION DOC] A generator with multiple except clauses and a finally block;
        # throwing RuntimeError selects the matching handler. Result is sine plus tangent and
        # the counter is 1101.
        z = 0

        def whoo(t):
            # [FUNCTION DOC] HELPER: Generator with ValueError and RuntimeError handlers and a
            # finally block updating an outer counter.
            nonlocal z
            z = 0
            try:
                z += 1
                yield t.sin()
            except ValueError:
                z += 10
                yield t.cos()
            except RuntimeError:
                z += 100
                yield t.tan()
            finally:
                z += 1000
            z += 10_000

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws RuntimeError and
            # sums the results.
            gen = whoo(t)
            a = next(gen)
            b = gen.throw(RuntimeError)
            return a + b

        t = torch.randn(2)
        y = self._compile_check(fn, (t,))
        self.assertEqual(y, t.sin() + t.tan())
        self.assertEqual(z, 1 + 100 + 1000)

    def test_exception_context_with_yield(self):
        # [FUNCTION DOC] Throws ValueError into a generator yielding nothing meaningful and
        # catches it inside the compiled function; checks no unrelated exception appears.
        def f():
            # [FUNCTION DOC] HELPER: Generator with a single bare yield.
            yield

        def fn(t):
            # [FUNCTION DOC] HELPER: Primes the generator, throws ValueError, records that it
            # was caught and returns a sine.
            gen = f()
            gen.send(None)
            try:
                gen.throw(ValueError)
            except ValueError:
                z = 1
            except Exception as e:
                raise AssertionError from e
            assert z == 1  # noqa: S101
            return t.sin()

        self._compile_check(fn)

    def test_return_const_value_in_except_and_finally(self):
        # [FUNCTION DOC] A return in a finally block overrides the except block return; the
        # StopIteration raised by throw carries the value three.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding once; except returns two and
            # finally returns three.
            try:
                yield 1
            except ValueError:
                return 2  # noqa: B901
            finally:
                return 3  # noqa: B012, SIM107, B901

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws ValueError and checks the
            # StopIteration value is three; returns a sine.
            gen = whoo()
            next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                assert e.args[0] == 3  # noqa: S101
            except Exception as e:
                raise AssertionError from e
            return t.sin()

        self._compile_check(fn)

    def test_return_value_in_except_and_finally(self):
        # [FUNCTION DOC] Same as the previous test but returning objects of a local class; the
        # StopIteration value must be the object built in finally.
        class Foo:
            # [CLASS DOC] HELPER: Simple object holding the attribute x passed
            # to its initializer.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Stores the argument as the attribute x.
                self.x = x

        def whoo():
            # [FUNCTION DOC] HELPER: Generator whose except returns an object holding two and
            # finally returns an object holding three.
            try:
                yield 1
            except ValueError:
                return Foo(2)  # noqa: B901
            finally:
                return Foo(3)  # noqa: B012, SIM107, B901

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws ValueError and checks the
            # returned object holds three; returns a sine.
            gen = whoo()
            next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                assert e.args[0].x == 3  # noqa: S101
            except Exception as e:
                raise AssertionError from e
            return t.sin()

        self._compile_check(fn)

    def test_return_None_in_except_and_finally(self):
        # [FUNCTION DOC] A bare return in finally overrides the except return, so the
        # StopIteration raised by throw has no arguments.
        def whoo():
            # [FUNCTION DOC] HELPER: Generator yielding once; except returns two and
            # finally returns nothing.
            try:
                yield 1
            except ValueError:
                return 2  # noqa: B901
            finally:
                return  # noqa: B012, SIM107

        def fn(t):
            # [FUNCTION DOC] HELPER: Starts the generator, throws ValueError and checks the
            # StopIteration has no arguments; returns a sine.
            gen = whoo()
            next(gen)
            try:
                gen.throw(ValueError)
            except StopIteration as e:
                assert len(e.args) == 0  # noqa: S101
            except Exception as e:
                raise AssertionError from e
            return t.sin()

        self._compile_check(fn)


class TestGeneratorPEP(GeneratorTestsBase):
    # [CLASS DOC] Tests ported from the CPython PEP doctest block for generators, run
    # under the dynamo test decorator.
    # Ported from CPython Lib/test/test_generators.py `pep_tests` doctest block.

    @make_dynamo_test
    def test_resume_running_generator(self):
        # [FUNCTION DOC] A generator that advances itself while running raises ValueError
        # about already executing.
        # A generator cannot be resumed while it is actively running.
        def g():
            # [FUNCTION DOC] HELPER: Generator that calls next on the outer generator
            # variable while running.
            i = next(me)
            yield i

        me = g()
        self.assertRaisesRegex(ValueError, "generator already executing", next, me)

    @make_dynamo_test
    def test_return_is_not_stopiteration(self):
        # [FUNCTION DOC] A return statement just ends the generator (not caught by a bare
        # except), while a raised StopIteration is caught by a bare except like any exception.
        # return simply exits; it is not caught by a bare except.
        def f1():
            # [FUNCTION DOC] HELPER: Generator returning inside a try with a bare except that
            # would yield one.
            try:
                return
            except:  # noqa: E722
                yield 1

        self.assertEqual(list(f1()), [])

        # a raised StopIteration is caught by a bare except, like any exception.
        def f2():
            # [FUNCTION DOC] HELPER: Generator raising StopIteration inside a try whose bare
            # except yields 42.
            try:
                raise StopIteration
            except:  # noqa: E722
                yield 42

        self.assertEqual(list(f2()), [42])

    @make_dynamo_test
    def test_exception_propagation(self):
        # [FUNCTION DOC] A ZeroDivisionError raised inside a generator propagates and
        # afterwards the generator is finished (StopIteration on the next advance).
        def f():
            # [FUNCTION DOC] HELPER: Function dividing one by zero.
            return 1 // 0

        def g():
            # [FUNCTION DOC] HELPER: Generator yielding the result of the failing function, then
            # 42 (never reached).
            yield f()  # the zero division exception propagates
            yield 42  # and we'll never get here

        k = g()
        self.assertRaises(ZeroDivisionError, next, k)
        self.assertRaises(StopIteration, next, k)  # cannot be resumed

    @make_dynamo_test
    def test_try_except_finally(self):
        # [FUNCTION DOC] A generator mixing nested try, except, raise and finally with yields
        # yields 1, 2, 4, 5, 8, 9, 10, 11 in order.
        def f():
            # [FUNCTION DOC] HELPER: Generator exercising nested handlers with a re-raise and a
            # finally block that yields.
            try:
                yield 1
                try:
                    yield 2
                    1 // 0
                    yield 3  # never get here
                except ZeroDivisionError:
                    yield 4
                    yield 5
                    raise
                except:  # noqa: E722
                    yield 6
                yield 7  # the "raise" above stops this
            except:  # noqa: E722
                yield 8
            yield 9
            try:
                x = 12  # noqa: F841
            finally:
                yield 10
            yield 11

        self.assertEqual(list(f()), [1, 2, 4, 5, 8, 9, 10, 11])

    @unittest.expectedFailure
    @make_dynamo_test
    def test_recursive_inorder_tree(self):
        # [FUNCTION DOC] Expected failure: a recursive in-order tree traversal generator joins
        # labels into the alphabet string.
        class Tree:
            # [CLASS DOC] HELPER: Binary tree node with a label and optional left and right
            # children; iterating over it delegates to the in-order traversal generator.
            def __init__(self, label, left=None, right=None):
                # [FUNCTION DOC] HELPER: Stores the label and the left and right
                # children as attributes.
                self.label = label
                self.left = left
                self.right = right

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Returns the in-order traversal generator for this node.
                return inorder(self)

        def tree(lst):
            # [FUNCTION DOC] HELPER: Builds a balanced tree from a sequence by taking the middle
            # item as root and recursing on both halves; returns an empty list for empty input.
            n = len(lst)
            if n == 0:
                return []
            i = n // 2
            return Tree(lst[i], tree(lst[:i]), tree(lst[i + 1 :]))

        def inorder(t):
            # [FUNCTION DOC] HELPER: Recursive generator yielding labels of the left subtree,
            # the node and the right subtree.
            if t:
                for x in inorder(t.left):
                    yield x
                yield t.label
                for x in inorder(t.right):
                    yield x

        t = tree("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
        self.assertEqual("".join(t), "ABCDEFGHIJKLMNOPQRSTUVWXYZ")


class TestGeneratorCoroutine(GeneratorTestsBase):
    # [CLASS DOC] Tests ported from the CPython coroutine doctest block; stdout-based
    # cases were rewritten to record into lists.
    # Ported from CPython Lib/test/test_generators.py `coroutine_tests` doctest
    # block. Cases relying on stdout capture were rewritten to record into a
    # list and assert; cases relying on gc finalization, traceback-level or
    # gi_frame introspection, and the deprecated 3-arg throw() were omitted.

    @make_dynamo_test
    def test_send_into_started_generator(self):
        # [FUNCTION DOC] Sends a value into a started generator; the value is recorded and the
        # next yield is returned.
        sent = []

        def f():
            # [FUNCTION DOC] HELPER: Generator that appends the sent value to a list
            # and yields two.
            sent.append((yield 1))
            yield 2

        g = f()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.send(42), 2)
        self.assertEqual(sent, [42])

    @make_dynamo_test
    def test_send_non_none_to_just_started(self):
        # [FUNCTION DOC] Sending a non-None value to a just-started generator raises TypeError
        # with the expected message.
        def f():
            # [FUNCTION DOC] HELPER: Generator yielding one and two.
            yield 1
            yield 2

        self.assertRaisesRegex(
            TypeError,
            "can't send non-None value to a just-started generator",
            f().send,
            "foo",
        )

    @make_dynamo_test
    def test_bare_yield_yields_none(self):
        # [FUNCTION DOC] A bare yield produces None as its value.
        def f():
            # [FUNCTION DOC] HELPER: Generator with one bare yield.
            yield

        self.assertEqual(list(f()), [None])

    @make_dynamo_test
    def test_yield_in_generator_expression(self):
        # [FUNCTION DOC] A function containing a yield inside a generator expression is still
        # a generator function; calling it gives a generator object.
        def f():
            # [FUNCTION DOC] HELPER: Function whose body has a yield inside a generator
            # expression passed to list.
            list(i for i in [(yield 26)])  # noqa: C400

        self.assertIsInstance(f(), types.GeneratorType)

    @make_dynamo_test
    def test_augmented_assignment_coroutine(self):
        # [FUNCTION DOC] A coroutine-style generator accumulates sent values with augmented
        # assignment; the recorded list is 10, 20, 30 after three sends.
        def coroutine(seq):
            # [FUNCTION DOC] HELPER: Generator that adds each sent value to a running count and
            # appends the count to the given list until it reaches 200.
            count = 0
            while count < 200:
                count += yield
                seq.append(count)

        seq = []
        c = coroutine(seq)
        next(c)
        self.assertEqual(seq, [])
        c.send(10)
        self.assertEqual(seq, [10])
        c.send(10)
        self.assertEqual(seq, [10, 20])
        c.send(10)
        self.assertEqual(seq, [10, 20, 30])

    @make_dynamo_test
    def test_throw_caught_in_loop(self):
        # [FUNCTION DOC] Throws ValueError twice into a generator that catches it in a loop;
        # recorded messages are an empty string then xyz.
        caught = []

        def f():
            # [FUNCTION DOC] HELPER: Endless generator that catches ValueError and
            # records its message.
            while True:
                try:
                    yield
                except ValueError as v:
                    caught.append(str(v))

        g = f()
        next(g)
        g.throw(ValueError)
        self.assertEqual(caught, [""])
        g.throw(ValueError("xyz"))
        self.assertEqual(caught, ["", "xyz"])

    # gen.throw() of a non-exception should raise a catchable TypeError, but
    # Dynamo surfaces it as an uncatchable InternalTorchDynamoError.
    @unittest.expectedFailure
    @make_dynamo_test
    def test_throw_non_exception_is_type_error(self):
        # [FUNCTION DOC] Expected failure: throwing a non-exception (string, integer, a non-
        # exception class) should raise TypeError, but Dynamo surfaces a different error.
        def f():
            # [FUNCTION DOC] HELPER: Endless generator catching ValueError.
            while True:
                try:
                    yield
                except ValueError:
                    pass

        g = f()
        next(g)
        self.assertRaises(TypeError, g.throw, "abc")
        self.assertRaises(TypeError, g.throw, 0)
        self.assertRaises(TypeError, g.throw, list)

    @make_dynamo_test
    def test_throw_terminates_generator(self):
        # [FUNCTION DOC] Throwing an exception the generator does not handle terminates it, so
        # a later send raises StopIteration.
        caught = []

        def f():
            # [FUNCTION DOC] HELPER: Endless generator catching ValueError and
            # recording the message.
            while True:
                try:
                    yield
                except ValueError as v:
                    caught.append(str(v))

        g = f()
        next(g)
        self.assertRaises(TypeError, g.throw, TypeError)
        self.assertRaises(StopIteration, g.send, 2)

    @make_dynamo_test
    def test_throw_on_just_opened_generator(self):
        # [FUNCTION DOC] Throwing into a just-opened generator raises the thrown error
        # with its message.
        def f():
            # [FUNCTION DOC] HELPER: Generator yielding one.
            yield 1

        self.assertRaisesRegex(ValueError, "7", f().throw, ValueError(7))

    @make_dynamo_test
    def test_close_catches_generator_exit(self):
        # [FUNCTION DOC] A generator catching GeneratorExit logs on close; a second close
        # does nothing more.
        log = []

        def f():
            # [FUNCTION DOC] HELPER: Generator that logs exiting when GeneratorExit is caught.
            try:
                yield
            except GeneratorExit:
                log.append("exiting")

        g = f()
        next(g)
        g.close()
        self.assertEqual(log, ["exiting"])
        g.close()  # should be a no-op now
        self.assertEqual(log, ["exiting"])

    @make_dynamo_test
    def test_close_on_various_states(self):
        # [FUNCTION DOC] Closing a not yet started generator and a normally started
        # one both succeed.
        def f():
            # [FUNCTION DOC] HELPER: Generator with one bare yield.
            yield

        f().close()  # close before opening
        g = f()
        next(g)
        g.close()  # close normally

    @make_dynamo_test
    def test_generator_exit_not_caught_by_except_exception(self):
        # [FUNCTION DOC] GeneratorExit is not caught by an Exception handler, but finally
        # runs; only the finally message is logged.
        log = []

        def f():
            # [FUNCTION DOC] HELPER: Generator with an Exception handler and a finally block
            # that log their entry.
            try:
                yield
            except Exception:
                log.append("except")
            finally:
                log.append("finally")

        g = f()
        next(g)
        g.close()
        self.assertEqual(log, ["finally"])

    @make_dynamo_test
    def test_generator_ignored_generator_exit(self):
        # [FUNCTION DOC] Yielding after GeneratorExit makes close raise RuntimeError about
        # ignoring GeneratorExit; a second close does not raise.
        def f():
            # [FUNCTION DOC] HELPER: Generator that yields a string when
            # GeneratorExit is caught.
            try:
                yield
            except GeneratorExit:
                yield "foo!"

        g = f()
        next(g)
        self.assertRaisesRegex(RuntimeError, "generator ignored GeneratorExit", g.close)
        g.close()

    @make_dynamo_test
    def test_error_during_close_propagates(self):
        # [FUNCTION DOC] An error raised from the GeneratorExit handler propagates from close.
        def f():
            # [FUNCTION DOC] HELPER: Generator whose GeneratorExit handler raises TypeError.
            try:
                yield
            except GeneratorExit:
                raise TypeError("fie!") from None

        g = f()
        next(g)
        self.assertRaisesRegex(TypeError, "fie!", g.close)

    @make_dynamo_test
    def test_yield_expression_makes_generator(self):
        # [FUNCTION DOC] A function containing a yield expression is a generator function.
        def f():
            # [FUNCTION DOC] HELPER: Function assigning the result of a bare yield
            # to a variable.
            x = yield  # noqa: F841

        self.assertIsInstance(f(), types.GeneratorType)

    @make_dynamo_test
    def test_send_to_subscript_targets(self):
        # [FUNCTION DOC] Sends values to a generator whose assignment targets are
        # subscripts evaluated around two yields; list contents change as sends arrive and
        # finally the generator stops.
        def f(d):
            # [FUNCTION DOC] HELPER: Generator that yields a, then b, then stores 27 into the
            # list at the two sent indexes.
            d[(yield "a")] = d[(yield "b")] = 27

        data = [1, 2]
        g = f(data)
        self.assertEqual(g.send(None), "a")
        self.assertEqual(data, [1, 2])
        self.assertEqual(g.send(0), "b")
        self.assertEqual(data, [27, 2])
        self.assertRaises(StopIteration, g.send, 1)
        self.assertEqual(data, [27, 27])


class _DelegatingIterator:
    # [CLASS DOC] HELPER: Iterator object (not a generator) implementing next, send, throw
    # and close; throw and close record log entries. Defined at module level to avoid the
    # local class build limitation.
    # An iterator (not a generator) implementing the generator protocol, used to
    # exercise `yield from <iterator>` where the subiterator has its own
    # throw()/close(). Defined at module scope so tracing doesn't hit the local
    # class definition (__build_class__) limitation.
    def __init__(self, log):
        # [FUNCTION DOC] HELPER: Stores the log list to record throw and close calls.
        self.log = log

    def __iter__(self):
        # [FUNCTION DOC] HELPER: Returns the iterator itself.
        return self

    def __next__(self):
        # [FUNCTION DOC] HELPER: Always returns one.
        return 1

    def send(self, value):
        # [FUNCTION DOC] HELPER: Ignores the sent value and returns one.
        return 1

    def throw(self, typ, val=None, tb=None):
        # [FUNCTION DOC] HELPER: Records that throw was called and returns two
        # (ignoring the exception).
        self.log.append("iter throw")
        return 2

    def close(self):
        # [FUNCTION DOC] HELPER: Records that close was called.
        self.log.append("iter close")


class TestSubgeneratorDelegation(GeneratorTestsBase):
    # [CLASS DOC] Tests for send, throw and close semantics being forwarded through yield
    # from to sub-generators and sub-iterators.
    # Delegation semantics for send/throw/close through `yield from`. CPython
    # forwards throw() and close() into the subiterator the delegating
    # generator is suspended on; these tests assert the subgenerator's own
    # handlers/finally run, which distinguishes real forwarding from raising
    # the exception at the outer `yield from` point.

    @make_dynamo_test
    def test_send_into_subgen(self):
        # [FUNCTION DOC] A value sent to the delegating generator reaches the sub-generator.
        got = []

        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator recording its received value
            # and yielding two.
            x = yield 1
            got.append(x)
            yield 2

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.send(42), 2)
        self.assertEqual(got, [42])

    @make_dynamo_test
    def test_throw_into_subgen_caught(self):
        # [FUNCTION DOC] Throwing into a delegating generator is caught by the sub-generator,
        # which logs and yields two.
        log = []

        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator with a ValueError handler that logs
            # and yields two.
            try:
                yield 1
            except ValueError:
                log.append("subgen caught")
                yield 2

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.throw(ValueError), 2)
        self.assertEqual(log, ["subgen caught"])

    @make_dynamo_test
    def test_throw_into_subgen_uncaught_runs_finally(self):
        # [FUNCTION DOC] An uncaught exception thrown through yield from runs the sub-
        # generator finally block and propagates.
        log = []

        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator with a finally block that logs.
            try:
                yield 1
            finally:
                log.append("subgen finally")

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        next(g)
        self.assertRaises(ValueError, g.throw, ValueError)
        self.assertEqual(log, ["subgen finally"])

    @make_dynamo_test
    def test_throw_generator_exit_into_subgen(self):
        # [FUNCTION DOC] Throwing GeneratorExit through yield from closes the sub-generator
        # (its finally logs) and raises GeneratorExit in the outer generator.
        # throw(GeneratorExit) through `yield from` closes the subiterator, then
        # raises the GeneratorExit in the outer frame (CPython's `goto
        # throw_here`), rather than throwing into the already-closed subiterator.
        log = []

        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator with a finally block that logs.
            try:
                yield 1
            finally:
                log.append("subgen finally")

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertRaises(GeneratorExit, g.throw, GeneratorExit)
        self.assertEqual(log, ["subgen finally"])

    @make_dynamo_test
    def test_throw_through_nested_yield_from(self):
        # [FUNCTION DOC] Throwing through two levels of yield from reaches the innermost
        # generator handler, which logs and yields two.
        log = []

        def leaf():
            # [FUNCTION DOC] HELPER: Innermost generator with a ValueError handler that logs
            # and yields two.
            try:
                yield 1
            except ValueError:
                log.append("leaf caught")
                yield 2

        def mid():
            # [FUNCTION DOC] HELPER: Middle generator delegating to the leaf.
            yield from leaf()

        def outer():
            # [FUNCTION DOC] HELPER: Outer generator delegating to the middle generator.
            yield from mid()

        g = outer()
        next(g)
        self.assertEqual(g.throw(ValueError), 2)
        self.assertEqual(log, ["leaf caught"])

    @make_dynamo_test
    def test_close_runs_subgen_finally(self):
        # [FUNCTION DOC] Closing a delegating generator runs the sub-generator finally block.
        log = []

        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator with a try and finally that logs.
            try:
                yield 1
                yield 2
            finally:
                log.append("subgen finally")

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        next(g)
        g.close()
        self.assertEqual(log, ["subgen finally"])

    @make_dynamo_test
    def test_close_subgen_catches_generator_exit(self):
        # [FUNCTION DOC] Closing a delegating generator whose sub-generator catches
        # GeneratorExit; the handler logs.
        log = []

        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator catching GeneratorExit and logging.
            try:
                yield 1
            except GeneratorExit:
                log.append("subgen exit")

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        next(g)
        g.close()
        self.assertEqual(log, ["subgen exit"])

    @make_dynamo_test
    def test_close_subgen_ignores_generator_exit(self):
        # [FUNCTION DOC] Closing when the sub-generator yields after GeneratorExit raises
        # RuntimeError about ignoring GeneratorExit.
        def subgen():
            # [FUNCTION DOC] HELPER: Sub-generator yielding two when GeneratorExit is caught.
            try:
                yield 1
            except GeneratorExit:
                yield 2

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the sub-generator.
            yield from subgen()

        g = outer()
        next(g)
        self.assertRaisesRegex(RuntimeError, "generator ignored GeneratorExit", g.close)

    @make_dynamo_test
    def test_throw_into_iterator_subiter(self):
        # [FUNCTION DOC] Throwing into a generator delegating to an iterator object with its
        # own throw method calls that method (which logs and returns two).
        log = []

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the custom iterator object.
            yield from _DelegatingIterator(log)

        g = outer()
        self.assertEqual(next(g), 1)
        self.assertEqual(g.throw(ValueError), 2)
        self.assertEqual(log, ["iter throw"])

    @make_dynamo_test
    def test_close_into_iterator_subiter(self):
        # [FUNCTION DOC] Closing a generator delegating to a custom iterator calls the
        # iterator close method.
        log = []

        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to the custom iterator object.
            yield from _DelegatingIterator(log)

        g = outer()
        next(g)
        g.close()
        self.assertEqual(log, ["iter close"])

    @make_dynamo_test
    @unittest.expectedFailure
    def test_throw_into_iterator_without_throw(self):
        # [FUNCTION DOC] Expected failure: for a plain list iterator without a throw method, a
        # thrown ValueError should be raised from the outer yield from point.
        # A plain iterator (no throw method): the exception is raised in the
        # outer frame at the yield-from point (CPython's `goto throw_here`).
        def outer():
            # [FUNCTION DOC] HELPER: Generator delegating to a list iterator.
            yield from iter([1, 2, 3])

        g = outer()
        next(g)
        self.assertRaises(ValueError, g.throw, ValueError)


instantiate_parametrized_tests(GeneratorTests)
instantiate_parametrized_tests(TestGeneratorSend)
instantiate_parametrized_tests(TestGeneratorClose)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
