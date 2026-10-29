# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo tests for unspecialized Python scalars (numpy scalars, floats,
# random values, dynamic ints) with assume_static_by_default turned off, i.e.
# everything as dynamic as possible. Contains UnspecTests (device independent) and
# UnspecTestsDevice (instantiated per device type by instantiate_device_type_tests at
# import, with xpu allowed). Tests mostly compare eager results against torch.compile
# results and check compile counts. Imports numpy and comptime.
import contextlib
import math
import random
import unittest

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
import torch.nn.functional as F
from torch._dynamo.comptime import comptime
from torch._dynamo.testing import CompileCounter, CompileCounterWithBackend, same
from torch.testing._internal.common_cuda import PLATFORM_SUPPORTS_MEM_EFF_ATTENTION
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import skipIfWindows
from torch.testing._internal.logging_utils import logs_to_string


# The intention of this test file is you should put test cases specifically
# for assume_static_by_default=False, aka you want to YOLO make everything as
# dynamic as possible.  If you want to test the more normal situation where
# you assume static by default, put it in a regular test file and
# test_dynamic_shapes will cover both the YOLO and non-YOLO cases.


@torch._dynamo.config.patch(assume_static_by_default=False)
class UnspecTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case decorated with a config patch setting
    # assume_static_by_default to False. Stateless; each method compiles small
    # functions and compares to eager, checks recompile counts via CompileCounter, or
    # checks guard/log output. Covers numpy scalars, the random module and Random
    # objects, unspecialized floats and ints, mark_dynamic/mark_unbacked, shifts.
    def test_numpy_correctness(self):
        # [FUNCTION DOC] Compiles a function taking a float64 tensor, an int64 tensor and a
        # numpy int64 scalar and returning a dict of numpy-derived values plus a tensor
        # expression. The compiled result (counting backend) must equal the eager result.
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Converts the tensors to numpy, returns a dict of inputs,
            # sums, a division, and elementwise sum, plus a tensor sum with the scalar.
            xy = [x + y, y, False]
            np_x = x.numpy()
            np_y = y.numpy()
            return {
                "x": x,
                "z": z,
                "a": np_y.sum(),
                "b": xy,
                "c": np_y[0][0] / 68,
                "d": np_x.sum(),
                "e": np_x + np_y,
            }, x + np_y.sum() + z

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        y = torch.ones([2, 2], dtype=torch.int64)
        z = np.int64(12)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x, y, z)
        self.assertEqual(res1, res2)

    def test_no_recompilations(self):
        # [FUNCTION DOC] Calls a compiled function ten times with different
        # numpy int64 values; asserts only one frame compiled with two ops,
        # so numpy scalar values are not specialized on.
        # no recompilations if passing on different numpy int values
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a dict with the
            # tensor plus one and the scalar halved.
            return {"a": x + 1, "b": y / 2}

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        for i in range(10):
            opt_fn(x, np.int64(i))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

    @unittest.expectedFailure  # array scalars decay to 0D arrays
    def test_builtin_max_min(self):
        # [FUNCTION DOC] Marked as an expected failure because numpy array scalars decay to 0D
        # arrays. Compares eager and compiled max/min of a numpy int64 scalar and a Python int
        # mixed with tensor arithmetic, using the same helper with relaxed numpy equality.
        # test unspecialized primitive max/min
        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: Returns tensor plus one, max of two
            # scalars, min of shifted scalar and the other scalar.
            return z + 1, max(x, y), min(x - 4, y)

        x = np.int64(12)
        y = 10
        z = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float64)
        res1 = fn(x, y, z)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res2 = opt_fn(x, y, z)
        self.assertTrue(same(res1, res2, relax_numpy_equality=True))

    def test_feed_random_values_into_graph_only(self):
        # [FUNCTION DOC] Checks random values inside the compiled graph function
        # reproduce eager values. Seeds python random, runs eager, warms the compiled
        # function once, reseeds, runs compiled, and asserts equality.
        def fn(shape):
            # [FUNCTION DOC] HELPER: Seeds torch RNG, returns a random tensor of the given
            # shape scaled by a random integer drawn from the python random module.
            torch.manual_seed(123)
            x = torch.randn(shape) * random.randint(30, 100)
            return x

        shape = [2, 3]
        random.seed(1)
        res1 = fn(shape)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(shape)
        random.seed(1)
        res2 = opt_fn(shape)

        self.assertTrue(same(res1, res2))

    def test_random_values_with_graph_break(self):
        # [FUNCTION DOC] Checks python random calls around graph breaks are reproducible.
        # Compares eager and compiled results after reseeding, with a warm-up run first.
        def fn(x):
            # [FUNCTION DOC] HELPER: Draws random values, forces graph breaks by
            # reading a tensor sum as a Python number twice, and returns tensor
            # plus the first value together with a random int.
            r1 = random.random()
            y = x + random.uniform(10, 20)
            y.sum().item()
            r2 = random.randint(2, 18)  # no graph output in this frame
            y.sum().item()
            return y + r1, r2

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(x)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    # Really annoying intersection of specialization and RandomValueSource
    # If we get a RandomValueSource with a single element tensor, we should return a ConstantVariable like other
    # unspects... but if we do, we break the bytecode assumptions and guards will not work as we will be referring
    # to a name from a source that is not there. If we call .item() and take the wrapped_value out, where we do
    # wrapped_value = wrapped_value.item() where we send unspec down to wrap_fx_proxy, this test passes and then
    # some models fail on missing codegen.tx.output.random_values_var. If we let the tensor value go into wrap as
    # it is, this test fails.
    # The real solution here is to rewrite RandomValueSource and all the codegen it does from the ground up.
    def test_multiple_consecutive_random_calls_before_graph(self):
        # [FUNCTION DOC] Several python random calls in a row before any graph op
        # must give reproducible results after reseeding (compiled vs eager, with a
        # warm-up). The preceding comment explains why this is delicate for the
        # random value source handling of single-element tensors.
        def fn(x):
            # [FUNCTION DOC] HELPER: Draws three random range values as tensor dimensions,
            # creates a random tensor of that shape, and returns the input plus two with it.
            dim1 = random.randrange(start=0, stop=5)
            dim2 = random.randrange(start=0, stop=5)
            dim3 = random.randrange(start=0, stop=5)
            y = torch.rand(dim1, dim2, dim3)
            return x + 2, y

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        random.seed(1)
        res1 = fn(x)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(x)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_compiled_random_calls_are_random(self):
        # [FUNCTION DOC] A compiled function using python random must give
        # different values on each of five calls (consecutive results are not
        # equal). Refers to a linked issue on stale random values.
        # For compiled functions with random calls,
        # it should return different values for every iteration.
        # https://github.com/pytorch/pytorch/issues/95425
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one times a uniform random number.
            return (x + 1) * random.uniform(0, 1)

        res = []
        for _ in range(5):
            res.append(fn(torch.ones(2)))
        for i in range(1, 5):
            self.assertFalse(same(res[i - 1], res[i]))

    def test_module_random_random_fullgraph(self):
        # [FUNCTION DOC] Same as the previous check but for the random module's random
        # function, a C builtin bound to the global Random instance; with fullgraph it
        # must trace and produce different values over five calls.
        # random.random is a C builtin method bound to the module-global
        # Random instance (unlike randint/randrange/uniform, which are Python
        # methods); it must still route through the RandomValueSource path
        # rather than graph-breaking on a skipped builtin.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns the input plus one times a random float.
            return (x + 1) * random.random()

        res = []
        for _ in range(5):
            res.append(fn(torch.ones(2)))
        for i in range(1, 5):
            self.assertFalse(same(res[i - 1], res[i]))

    def test_random_call_with_while_loop(self):
        # [FUNCTION DOC] A compiled function with a loop that redraws a
        # random value until it differs from the first must match eager
        # results after seeding, for two seeds (1 and 10).
        def fn(x):
            # [FUNCTION DOC] HELPER: Draws a random range value then redraws
            # in a loop while equal, and returns the input doubled.
            dim1 = random.randrange(start=0, stop=3)
            dim2 = dim1
            while dim1 == dim2:
                dim2 = random.randrange(start=0, stop=3)
            return x * 2

        x = torch.randn(4)
        random.seed(1)
        res1 = fn(x)
        opt_fn = torch.compile(fn, backend="eager")
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(x)
        random.seed(1)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

        random.seed(10)
        res1 = fn(x)
        random.seed(10)
        res2 = opt_fn(x)
        self.assertTrue(same(res1, res2))

    def test_random_object(self):
        # [FUNCTION DOC] Tests passing a Random object in, creating one inside, calling it,
        # and returning it from a fullgraph compile: the output tensor, the global random
        # state, and states of both returned Random objects must equal the eager ones.
        # test argument passing, mutation, reconstruction, state correctness
        def fn(x, rand2):
            # [FUNCTION DOC] HELPER: Draws one int from the global generator,
            # one from the passed generator, one from a new seeded generator;
            # returns tensor plus the ints and both generators.
            r1 = random.randint(1, 9)
            r2 = rand2.randint(1, 9)
            rand3 = random.Random(42)
            r3 = rand3.randint(1, 9)

            y = x + r1 + r2 + r3
            return y, rand2, rand3

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        random.seed(0)
        y_1, rand2_1, rand3_1 = fn(inp, random.Random(12))
        state_1 = random.getstate()
        # Especially for internal: before resetting the seed, first shake out any rng
        # calls that occur on compile, e.g., as a result of some module initializations.
        opt_fn(inp, random.Random(12))
        random.seed(0)
        y_2, rand2_2, rand3_2 = opt_fn(inp, random.Random(12))
        state_2 = random.getstate()
        self.assertEqual(y_1, y_2)
        self.assertEqual(state_1, state_2)
        self.assertEqual(rand2_1.getstate(), rand2_2.getstate())
        self.assertEqual(rand3_1.getstate(), rand3_2.getstate())

    def test_random_object_methods(self):
        # [FUNCTION DOC] Calls seed, setstate, getstate, random, randint, randrange
        # and uniform on Random objects inside a fullgraph compile; output and final
        # states of the three passed generators must match eager.
        def fn(x, rand1, rand2, rand3):
            # [FUNCTION DOC] HELPER: Reseeds one generator, copies a new generator state into
            # another, draws from several generators, and returns the input plus the draws.
            rand1.seed(42)
            rand4 = random.Random(9002)
            rand2.setstate(rand4.getstate())
            r1 = rand1.random()
            r2 = rand2.randint(1, 10)
            r3 = rand3.randrange(10)
            r4 = rand4.uniform(0, 1)
            return x + r1 + r2 + r3 + r4

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        rand1_1 = random.Random(1)
        rand2_1 = random.Random(2)
        rand3_1 = random.Random(3)
        rand1_2 = random.Random(1)
        rand2_2 = random.Random(2)
        rand3_2 = random.Random(3)
        y1 = fn(inp, rand1_1, rand2_1, rand3_1)
        y2 = opt_fn(inp, rand1_2, rand2_2, rand3_2)
        self.assertEqual(y1, y2)
        self.assertEqual(rand1_1.getstate(), rand1_2.getstate())
        self.assertEqual(rand2_1.getstate(), rand2_2.getstate())
        self.assertEqual(rand3_1.getstate(), rand3_2.getstate())

    def test_random_object_shuffle(self):
        # [FUNCTION DOC] Shuffling lists on an explicit Random object under
        # fullgraph, including a list of tensors, must match eager exactly,
        # including the next draw after the shuffles.
        # shuffle on an explicit Random object is an exact, reproducible,
        # index-only permutation, so it also works for non-constant (tensor)
        # elements. The trailing draw checks the RNG state advanced correctly.
        def fn(x):
            # [FUNCTION DOC] HELPER: Shuffles a list of ints and a list of tensors with one
            # seeded generator and returns both lists plus the next random float.
            r = random.Random(42)
            items = list(range(10))
            r.shuffle(items)
            tensors = [x + i for i in range(5)]
            r.shuffle(tensors)
            return items, tensors, r.random()

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref_items, ref_tensors, ref_r = fn(x)
        res_items, res_tensors, res_r = opt_fn(x)
        self.assertEqual(ref_items, res_items)
        self.assertEqual(ref_tensors, res_tensors)
        self.assertEqual(ref_r, res_r)

    def test_random_object_sample(self):
        # [FUNCTION DOC] Sampling from range, string and list of tensors with an explicit
        # Random object under fullgraph must match eager exactly, including the following
        # draw.
        # sample selects index positions, so it is exact/reproducible for an
        # explicit Random object and works over sequence populations (ranges,
        # strings) as well as non-constant (tensor) elements.
        def fn(x):
            # [FUNCTION DOC] HELPER: Samples from a range, a string and a list of tensors
            # using one seeded generator; returns the three samples and the next random float.
            r = random.Random(42)
            from_range = r.sample(range(100), 5)
            from_str = r.sample("abcdefghij", 3)
            tensors = [x + i for i in range(8)]
            from_tensors = r.sample(tensors, 3)
            return from_range, from_str, from_tensors, r.random()

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])
        self.assertEqual(ref[2], res[2])
        self.assertEqual(ref[3], res[3])

    def test_random_object_sample_raises(self):
        # [FUNCTION DOC] Sampling more items than the population holds
        # must raise ValueError when the compiled function runs.
        # A sample larger than the population raises ValueError like eager.
        def fn():
            # [FUNCTION DOC] HELPER: Samples five items from a
            # three item list on a seeded generator.
            r = random.Random(0)
            return r.sample([1, 2, 3], 5)

        opt_fn = torch.compile(fn, backend="eager")
        with self.assertRaises(ValueError):
            opt_fn()

    def test_random_module_shuffle_sample(self):
        # [FUNCTION DOC] Module-level shuffle and sample must trace under fullgraph. Since the
        # global state is snapshotted at compile time, only checks structure: shuffled list is
        # a permutation, four picks are distinct characters from the source string.
        # Module-level random.shuffle/random.sample must trace under fullgraph
        # (exercised by the CPython dict/list tests). Like an explicit Random
        # object, the global RNG state is snapshotted at compile time, so assert
        # structural correctness rather than cross-run reproducibility.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Shuffles a list of ten ints, samples four
            # characters from a string, and returns both with the input plus one.
            items = list(range(10))
            random.shuffle(items)
            picks = random.sample("abcdefghij", 4)
            return items, picks, x + 1

        random.seed(0)
        items, picks, _ = fn(torch.zeros(2))
        self.assertEqual(sorted(items), list(range(10)))
        self.assertEqual(len(picks), 4)
        self.assertEqual(len(set(picks)), 4)
        self.assertTrue(all(p in "abcdefghij" for p in picks))

    def test_random_module_seed_shuffle(self):
        # [FUNCTION DOC] A seed call on the global random module inside a
        # fullgraph compile followed by shuffle must produce the same
        # permutation as seeding and shuffling eagerly.
        # Module-level random.seed is bound to the global random.Random instance
        # and must trace under fullgraph rather than graph-breaking on a skipped
        # function. A seed() inside the compiled region makes the subsequent
        # shuffle deterministic, matching the CPython test_sort
        # TestOptimizedCompares pattern (seed(0) then shuffle).
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Seeds the global generator with zero, shuffles a
            # list of ten ints, and returns it with the input plus one.
            items = list(range(10))
            random.seed(0)
            random.shuffle(items)
            return items, x + 1

        compiled_items, _ = fn(torch.zeros(2))

        expected = list(range(10))
        random.seed(0)
        random.shuffle(expected)
        self.assertEqual(compiled_items, expected)

    def test_random_object_overridden_methods(self):
        # [FUNCTION DOC] Random objects with instance-level overrides of random and
        # getstate (wrapping the originals) are used in a compiled function; this causes
        # graph breaks but must not crash, and results must equal eager.
        # these will result in graph breaks, but we shouldn't crash
        def get_rng():
            # [FUNCTION DOC] HELPER: Builds two seeded generators and replaces random on
            # one and getstate on the other with wrapper functions; returns both.
            rand1 = random.Random(1)
            rand2 = random.Random(2)

            orig_random = rand1.random

            def custom_random():
                # [FUNCTION DOC] HELPER: Wrapper calling the original
                # bound random method of the first generator.
                return orig_random()

            orig_getstate = rand2.getstate

            def custom_getstate():
                # [FUNCTION DOC] HELPER: Wrapper calling the original
                # bound getstate method of the second generator.
                return orig_getstate()

            rand1.random = custom_random
            rand2.getstate = custom_getstate
            return rand1, rand2

        def fn(x, rand1, rand2):
            # [FUNCTION DOC] HELPER: Draws from the first generator, copies the second
            # generator state into a fresh generator, draws from it, returns input plus both
            # draws.
            r1 = rand1.random()
            rand3 = random.Random()
            rand3.setstate(rand2.getstate())
            r2 = rand3.random()
            return x + r1 + r2

        inp = torch.randn(3, 3)
        opt_fn = torch.compile(fn, backend="eager")
        y1 = fn(inp, *get_rng())
        y2 = opt_fn(inp, *get_rng())
        self.assertEqual(y1, y2)

    def test_random_in_dynamo(self):
        # [FUNCTION DOC] Checks that random calls made by dynamo internals itself do
        # not disturb user random streams. Patches the compile entry and the
        # translator run method with wrappers that burn a random call, then compares
        # eager and compiled results for a simple function, one with a graph break,
        # and consecutive calls. Patches are held on an exit stack.
        # KNOWN ISSUE: the exit stack is never closed so the patches are never
        # undone, and unittest.mock is used without importing it explicitly.
        # test that system random calls still work even
        # if Dynamo calls random methods.

        exit_stack = contextlib.ExitStack()

        def patch_fn_with_rng_burn(name):
            # [FUNCTION DOC] HELPER: Looks up the named function, defines a
            # wrapper that burns a random call then delegates to the original, and
            # registers a mock patch of that name on the exit stack.
            orig_fn = eval(name)

            def bad(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Draws and discards a random number, then calls
                # the original with the same arguments and returns its result.
                # burn random call within dynamo
                random.random()
                return orig_fn(*args, **kwargs)

            exit_stack.enter_context(unittest.mock.patch(name, bad))

        x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])

        # we don't guard against random calls in eval_frame.py today
        # patch_fn_with_rng_burn("torch._dynamo.eval_frame._maybe_set_eval_frame")
        patch_fn_with_rng_burn("torch._dynamo.convert_frame._compile")
        patch_fn_with_rng_burn(
            "torch._dynamo.symbolic_convert.InstructionTranslator.run"
        )

        def f1(x):
            # [FUNCTION DOC] HELPER: Draws two ints and a uniform
            # float, returns tensor shifted by them.
            # simple test
            r1 = random.randint(1, 9)
            y = x + random.uniform(10, 20)
            r2 = random.randrange(0, 10)
            return y + r1, r2

        random.seed(1)
        ref1 = f1(x)
        opt_f1 = torch.compile(f1, backend="eager", fullgraph=True)
        random.seed(1)
        res1 = opt_f1(x)
        self.assertEqual(ref1, res1)

        def f2(x):
            # [FUNCTION DOC] HELPER: Draws an int, adds to input, forces a graph break,
            # draws a second int, adds again; returns tensor and both ints.
            # test with graph breaks
            r1 = random.randint(1, 9)
            x = x + r1
            torch._dynamo.graph_break()
            r2 = random.randint(10, 19)
            x = x + r2
            return x, r1, r2

        random.seed(2)
        ref2 = f2(x)
        opt_f2 = torch.compile(f2, backend="eager")
        random.seed(2)
        res2 = opt_f2(x)
        self.assertEqual(ref2, res2)

        def f3(x):
            # [FUNCTION DOC] HELPER: Adds a random int in a small range to the input.
            # test consecutive calls
            return x + random.randint(1, 10)

        random.seed(3)
        ref3 = f3(x)
        ref3_ = f3(x)
        opt_f3 = torch.compile(f3, backend="eager", fullgraph=True)
        random.seed(3)
        res3 = opt_f3(x)
        res3_ = opt_f3(x)
        self.assertEqual(ref3, res3)
        self.assertEqual(ref3_, res3_)

    def test_builtin_getitem(self):
        # [FUNCTION DOC] Builtin getitem and slicing on a python list with an
        # unspecialized int index, which also sizes a tensor; compiled equals eager.
        # builtin getitem args[0] is python list and args[1] is unspec
        def fn(x, idx):
            # [FUNCTION DOC] HELPER: Returns a zeros tensor sized by the index,
            # the list element, and a slice of the list from the index.
            return (torch.zeros(idx), x[idx], x[idx:])

        x = list(range(50))
        ref = fn(x, 48)  # 48 is unspecialized
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, 48)
        self.assertTrue(same(ref, res))

    def test_use_and_specialize(self):
        # [FUNCTION DOC] With dynamic compile and fullgraph, a function
        # branching on an int argument specializes on it: four calls with
        # values 2 and 3 yield expected sums and two frames.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds the two inputs, then
            # subtracts one if the int equals two, else adds one.
            x = x + y
            if y == 2:
                return x - 1
            else:
                return x + 1

        self.assertTrue(same(fn(torch.tensor([5]), 2), 6))
        self.assertTrue(same(fn(torch.tensor([6]), 2), 7))
        self.assertTrue(same(fn(torch.tensor([5]), 3), 9))
        self.assertTrue(same(fn(torch.tensor([4]), 3), 8))
        self.assertEqual(cnt.frame_count, 2)

    def test_no_recompiles(self):
        # [FUNCTION DOC] With dynamic compile, calling with different int values and tensors
        # does not recompile: results are right and frame count and op count are one.
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
            return x + y

        self.assertTrue(same(fn(torch.tensor([5]), 100), 105))
        self.assertTrue(same(fn(torch.tensor([4]), 200), 204))
        self.assertTrue(same(fn(torch.tensor([3]), 300), 303))
        self.assertTrue(same(fn(torch.tensor([2]), 400), 402))
        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_no_recompiles_prod_backward(self):
        # [FUNCTION DOC] Regression test for product with keepdim over varied shaped inputs
        # requiring grad, with backward each time; only one frame and one op may be compiled.
        # https://github.com/pytorch/pytorch/issues/120608
        cnt = CompileCounter()

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Product reduction over dim three with keepdim.
            return torch.prod(t, 3, keepdim=True)

        input_shapes = [(8, 10, 3, 2), (8, 3, 5, 2), (8, 4, 8, 2)]
        for s in input_shapes:
            t1 = torch.randn(s, requires_grad=True)
            h_result = fn(t1)
            grad = torch.ones_like(h_result)
            h_result.backward(grad)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(cnt.op_count, 1)

    def test_unspec_float_precision(self):
        # [FUNCTION DOC] Float scale factor with 16-plus digits passed to
        # bilinear interpolate with recompute_scale_factor; the output shape
        # from compiled must equal eager (no precision loss).
        def fn(image, scale_factor):
            # [FUNCTION DOC] HELPER: Interpolates image with the
            # scale factor and returns the output shape.
            image = torch.nn.functional.interpolate(
                image[None],
                size=None,
                scale_factor=scale_factor,
                mode="bilinear",
                recompute_scale_factor=True,
                align_corners=False,
            )[0]

            return image.shape

        x = torch.rand([3, 427, 640])
        scale_factor = 1.873536229133606
        ref = fn(x, scale_factor)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, scale_factor)
        self.assertTrue(same(ref, res))

    @unittest.expectedFailure  # fails as long as numpy scalars are 0D arrays
    def test_specializing_numpy_float_in_control_flow(self):
        # [FUNCTION DOC] Marked expected failure as numpy scalars are 0D arrays.
        # Intends to show numpy float scalars of three dtypes are specialized when
        # used in control flow: compares eager and compiled results.
        # np.float64 is unspecialized by default,
        # but it should be specialized when used in control flow.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Adds one if the scalar exceeds one, else subtracts one.
            if y > 1.0:
                return x + 1
            else:
                return x - 1

        x = torch.rand(4)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        for t in [np.float16, np.float32, np.float64]:
            y = t(1.23)
            ref = fn(x, y)
            res = opt_fn(x, y)
            self.assertTrue(same(ref, res))

    def test_mark_static_inside(self):
        # [FUNCTION DOC] Compiles with dynamic, fullgraph and a test-only comptime static
        # assertion after marking a dimension static inside the function; just runs without
        # error.
        def fn(x):
            # [FUNCTION DOC] HELPER: Marks dim zero static, asserts at compile
            # time that its size is static, returns input plus one.
            torch._dynamo.mark_static(x, 0)
            comptime.assert_static(x.size(0))
            return x + 1

        opt_fn = torch.compile(fn, dynamic=True, fullgraph=True, backend="eager")
        opt_fn(torch.randn(12, 23))

    def test_shape_graph_break(self):
        # [FUNCTION DOC] A tensor shape captured before a compile-time graph break is used
        # afterwards to build a random tensor; the compiled call must not error.
        from torch._dynamo.comptime import comptime

        def fn(x):
            # [FUNCTION DOC] HELPER: Saves the size, triggers a graph break via
            # comptime, and adds a random tensor of the saved size.
            x_shape = x.size()
            comptime.graph_break()
            return x + torch.randn(x_shape)

        x = torch.randn(20)
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)

    def test_isinstance_symint(self):
        # [FUNCTION DOC] Checks isinstance of a size element against int
        # holds for a static size and when the input is marked dynamic; a
        # second call is compiled with a dynamic dim.
        def fn(x):
            # [FUNCTION DOC] HELPER: Asserts the first dim
            # size is an int, returns input doubled.
            assert isinstance(x.size(0), int)  # noqa: S101
            return x * 2

        x = torch.randn(20)
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(x)
        y = torch.randn(30)
        torch._dynamo.mark_dynamic(y, 0)
        opt_fn(y)

    def test_mark_01_dynamic(self):
        # [FUNCTION DOC] An input of size one marked dynamic compiles and runs
        # without error even though zero/one specialization may apply.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns input doubled.
            return x * 2

        x = torch.randn(1)
        torch._dynamo.mark_dynamic(x, 0)
        opt_fn = torch.compile(fn, backend="eager")
        # This will fail to compile a generic kernel, but we should not
        # complain about it (mark dynamic will try its best but 0/1
        # specialization is allowed)
        opt_fn(x)

    def test_conv1d_symint_padding(self):
        # [FUNCTION DOC] Regression test for a convolution whose padding is
        # computed from tensor sizes; runs the compiled function with two
        # input widths (175, then 249), expecting no crash.
        kernel = torch.randn(1, 1, 4)

        def func(x):
            # [FUNCTION DOC] HELPER: Computes padding from kernel and input width,
            # runs a one-dim convolution with stride two and returns it.
            padding = math.ceil((kernel.shape[-1] + x.shape[-1] % 2) / 2) - 1
            out = F.conv1d(x, kernel, padding=padding, stride=2)
            return out

        opt_func = torch.compile(func, backend="eager")

        x = torch.randn(1, 1, 175)
        opt_func(x)  # passes
        x = torch.randn(1, 1, 249)
        opt_func(x)  # crashes

    @torch._dynamo.config.patch("assume_static_by_default", True)
    def test_propagate_dynamic_dim(self):
        # [FUNCTION DOC] With assume_static_by_default True, an input marked dynamic
        # keeps its dynamic dim indices (attribute dynamo propagated dynamic indices
        # equals set containing zero) on the output after a graph break.
        x = torch.randn(20)
        torch._dynamo.mark_dynamic(x, 0)

        @torch.compile()  # noqa: UNSPECIFIED_BACKEND
        def fn(x):
            # [FUNCTION DOC] HELPER: Doubles the input, graph
            # breaks via comptime, doubles again.
            y = x * 2
            comptime.graph_break()
            z = y * 2
            return z

        z = fn(x)
        self.assertEqual(z._dynamo_propagated_dynamic_indices, {0})

    def test_rshift_dynamic(self):
        # [FUNCTION DOC] Right shift of a uint8 tensor under
        # dynamic fullgraph compile; just runs.
        def shift_right(tensor: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Shifts the tensor right by two and converts to long.
            return (tensor >> 2).to(torch.long)

        opt_fn = torch.compile(
            shift_right, fullgraph=True, dynamic=True, backend="eager"
        )
        sample_input = torch.tensor([4, 4, 16, 32], dtype=torch.uint8)
        opt_fn(sample_input)

    def test_lshift_scalar_dynamic(self):
        # [FUNCTION DOC] Left shift of constant one by int input under dynamic
        # fullgraph; checks results 2 for input 1 and 32 for input 5.
        def shift_left(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns one shifted left by x.
            return 1 << x

        opt_fn = torch.compile(
            shift_left, fullgraph=True, dynamic=True, backend="eager"
        )
        self.assertEqual(opt_fn(1), 2)
        self.assertEqual(opt_fn(5), 32)

    def test_rshift_scalar_dynamic(self):
        # [FUNCTION DOC] Right shift of constant 64 by int input under dynamic
        # fullgraph; checks results 16 for input 2 and 4 for input 4.
        def shift_right(x: int) -> int:
            # [FUNCTION DOC] HELPER: Returns 64 shifted right by x.
            return 64 >> x

        opt_fn = torch.compile(
            shift_right, fullgraph=True, dynamic=True, backend="eager"
        )
        self.assertEqual(opt_fn(2), 16)
        self.assertEqual(opt_fn(4), 4)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_symfloat_to_tensor(self):
        # [FUNCTION DOC] With scalar output capture enabled, building tensors from a
        # scalar read out of a tensor (single, nested, zero-dim and tuple forms)
        # compiles under aot_eager with fullgraph and equals eager.
        def f1(v):
            # [FUNCTION DOC] HELPER: Builds a one-element tensor from the scalar value.
            return torch.tensor([v.item()])

        def f2(v):
            # [FUNCTION DOC] HELPER: Builds a two-by-one tensor
            # from the scalar value and a constant.
            return torch.tensor([[v.item()], [2.0]])

        def f3(v):
            # [FUNCTION DOC] HELPER: Builds a zero-dim tensor from the scalar value.
            return torch.tensor(v.item())

        def f4(v):
            # [FUNCTION DOC] HELPER: Builds a one-element tensor
            # from a tuple holding the scalar value.
            return torch.tensor((v.item(),))

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        r = torch.randn(1)

        self.assertEqual(f1(r), optimize(f1)(r))
        self.assertEqual(f2(r), optimize(f2)(r))
        self.assertEqual(f3(r), optimize(f3)(r))
        self.assertEqual(f4(r), optimize(f4)(r))

    @skipIfWindows(
        msg="AssertionError: The values for attribute 'dtype' do not match: torch.int32 != torch.int64."
    )
    def test_to_tensor(self):
        # [FUNCTION DOC] Skipped on Windows (dtype mismatch int32 vs int64). Tensor
        # construction from lists of numpy arrays and tensors under aot_eager fullgraph: f1
        # only compares shapes (random numpy data); f2 to f5 compare values with eager.
        def f1():
            # [FUNCTION DOC] HELPER: Builds a float64 tensor
            # from a list of four random numpy arrays.
            a = np.random.uniform(low=-1, high=1, size=(20, 1))
            return torch.tensor([a, a, a, a], dtype=torch.float64)

        def f2():
            # [FUNCTION DOC] HELPER: Builds a tensor from a list
            # of two nested single-element tensors.
            a = torch.tensor([[[123]]])
            return torch.tensor([a, a])

        def f3():
            # [FUNCTION DOC] HELPER: Builds a tensor from a list of two zero-dim tensors.
            a = torch.tensor(123)
            return torch.tensor([a, a])

        def f4():
            # [FUNCTION DOC] HELPER: Builds a tensor from a
            # zero-dim tensor and a nested one-element tensor.
            a = torch.tensor(123)
            b = torch.tensor([[[456]]])
            return torch.tensor([a, b])

        def f5():
            # [FUNCTION DOC] HELPER: Builds a tensor from a
            # list of two identical numpy arrays.
            a = np.array([1, 2])
            return torch.tensor([a, a])

        optimize = torch.compile(backend="aot_eager", fullgraph=True)

        self.assertEqual(f1().shape, optimize(f1)().shape)
        self.assertEqual(f2(), optimize(f2)())
        self.assertEqual(f3(), optimize(f3)())
        self.assertEqual(f4(), optimize(f4)())
        self.assertEqual(f5(), optimize(f5)())

    def test_sym_int_conversion(self):
        # [FUNCTION DOC] Converting a boolean comparison of a size to int and
        # multiplying inside a fullgraph compile must work for a static input.
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the input by the
            # int value of whether dim zero is zero.
            y = x.size(0)
            return x * int(y == 0)

        opt_fn = torch.compile(f, backend="eager", fullgraph=True)
        x = torch.randn(2, 3)
        opt_fn(x)

    def test_sum_dimlist_spec(self):
        # [FUNCTION DOC] Sum over a dim list (a tuple of dims) with
        # dynamic compile; compiled result equals eager.
        def fn(inputs, dim):
            # [FUNCTION DOC] HELPER: Sums the input over the given dims.
            return torch.sum(inputs, dim)

        inputs = torch.randn(128, 5, 24, 24)
        dim = (-1, 1, 0, 2)
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, dim), fn(inputs, dim))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_item_max(self):
        # [FUNCTION DOC] With scalar output capture, ones sized by the
        # maximum of a tensor scalar and 1024 compiled with fullgraph for
        # two inputs (1000 and 2000); results equal eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns a ones tensor whose
            # length is the max of the scalar and 1024.
            return torch.ones(max(x.item(), 1024))

        x = torch.tensor([1000])
        y = torch.tensor([2000])
        compl_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), compl_fn(x))
        self.assertEqual(fn(y), compl_fn(y))

    # https://github.com/pytorch/pytorch/issues/104812
    def test_argmin_coerces_symint_to_intlist_spec(self):
        # [FUNCTION DOC] Regression for argument parsing coercing a symbolic int into an int
        # list: a reduction with a scalar dim under dynamic compile must match eager.
        # KNOWN ISSUE: name says argmin but the code uses the minimum reduction amin.
        def fn(x, dim):
            # [FUNCTION DOC] HELPER: Takes the keepdim minimum reduction over dim.
            # the python arg parser coerces dim into a vector<int>
            return torch.amin(x, dim=dim, keepdim=True)

        x = torch.randn(4, 4, 4)
        dim = 2
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(x, dim), fn(x, dim))

    def test_exponential(self):
        # [FUNCTION DOC] In-place exponential with kwargs from a dict (rate and
        # generator None) under dynamic compile; compiled result equals eager.
        # KNOWN ISSUE: the in-place op mutates and returns the same input tensor
        # in both calls, so the final comparison is of a tensor with itself and
        # does not check that compiled and eager random values agree.
        def fn(inputs, op_inputs_dict):
            # [FUNCTION DOC] HELPER: Calls in-place exponential
            # on the inputs with the kwargs dict.
            res = inputs.exponential_(**op_inputs_dict)
            return res

        inputs = torch.randn(2, 3, 4)
        op_inputs_dict = {"lambd": 10, "generator": None}
        compl_fn = torch.compile(fn, dynamic=True, backend="eager", fullgraph=True)
        self.assertEqual(compl_fn(inputs, op_inputs_dict), fn(inputs, op_inputs_dict))

    def test_symbol_guard_limit_before_specialize(self):
        # [FUNCTION DOC] Checks the fx config for the symbol guard limit. Compiles a
        # function with four size inequality checks: with default config, three sizes
        # give one frame; after reset and a limit of three, three sizes give three
        # frames (specialization). Expectations are inline strings.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Checks four size inequality
            # facts and returns the input plus two.
            torch._check(x.size(0) != 3)
            torch._check(x.size(0) != 4)
            torch._check(x.size(0) != 5)
            torch._check(x.size(0) != 6)
            return x + 2

        # Control test
        fn(torch.randn(12))
        fn(torch.randn(13))
        fn(torch.randn(14))

        self.assertExpectedInline(cnts.frame_count, """1""")
        cnts.frame_count = 0

        torch._dynamo.reset()

        with torch.fx.experimental._config.patch(
            symbol_guard_limit_before_specialize=3
        ):
            fn(torch.randn(12))
            fn(torch.randn(13))
            fn(torch.randn(14))

            self.assertExpectedInline(cnts.frame_count, """3""")

    def test_defaults(self):
        # [FUNCTION DOC] A function calling another with a default int argument, with dynamic
        # compile; the default must be static at compile time and results equal eager.
        def g(x, i=8):
            # [FUNCTION DOC] HELPER: Asserts the default argument
            # is static and multiplies the input by it.
            comptime.assert_static(i)
            return x * i

        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the default-argument function with only the tensor.
            return g(x)

        inputs = torch.randn(2, 3, 4)
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        self.assertEqual(compl_fn(inputs), fn(inputs))

    @torch._dynamo.config.patch(specialize_float=False)
    def test_symfloat_no_replacement(self):
        # [FUNCTION DOC] With float specialization off, a method call on a float
        # argument must not create a symbol replacement with the symbol on both
        # sides (would recurse infinitely); compiled equals eager.
        # See https://github.com/pytorch/pytorch/pull/139250 for more context
        # The high level idea is if we don't want to set a replacement where a
        # symbol is on both the right and left side, otherwise we'll end up
        # in an infinite self._find recursion.
        def fn(t, m):
            # [FUNCTION DOC] HELPER: Doubles the tensor if the
            # float is integral, else returns it.
            return 2 * t if m.is_integer() else t

        t = torch.tensor([1])
        compl_fn = torch.compile(fn, dynamic=True, backend="eager")
        self.assertEqual(fn(t, 1.0), compl_fn(t, 1.0))

    @torch._dynamo.config.patch(specialize_float=False)
    def test_unspec_roundtrip_float_input(self):
        # [FUNCTION DOC] With float specialization off, two floats that differ
        # only in the last digits must be passed to the compiled function
        # exactly (not rounded), compared with a near-equality check.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds two if y equals five, else adds y.
            # KNOWN ISSUE: the final return statement after the branches is unreachable.
            if y == 5.0:
                return x + 2
            else:
                return x + y
            return (x, y)

        cf = torch.compile(backend="eager", fullgraph=True)(f)
        x = 1.1234567891234568
        y = 1.1234567891234569
        self.assertAlmostEqual(f(x, y), cf(x, y))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_input(self):
        # [FUNCTION DOC] With float specialization off and static-by-default,
        # float inputs: calls with 2, 3, 4 give 2 frames (automatic dynamic after
        # the second), 5 triggers guard-driven third compile, nan a fourth. Frame
        # counts are asserted inline with expected strings.
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            # [FUNCTION DOC] HELPER: Adds two if y equals five, otherwise adds y.
            if y == 5.0:
                return x + 2
            else:
                return x + y

        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        x = torch.randn(3)
        self.assertEqual(f(x, 2.0), cf(x, 2.0))
        self.assertEqual(f(x, 3.0), cf(x, 3.0))  # automatic dynamic kicks in here
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        self.assertExpectedInline(cnts.frame_count, """2""")  # no recompile
        self.assertEqual(f(x, 5.0), cf(x, 5.0))
        self.assertExpectedInline(cnts.frame_count, """3""")  # guard worked
        self.assertEqual(f(x, math.nan), cf(x, math.nan))
        self.assertExpectedInline(cnts.frame_count, """4""")  # nan always recompiles

    @torch._dynamo.config.patch(specialize_float=False, capture_scalar_outputs=True)
    def test_unspecialized_float_multiply_precision(self):
        # [FUNCTION DOC] For each of four float dtypes, a tensor times
        # several nearly equal Python floats under aot_eager must match
        # eager and compile only one frame.
        dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        for dtype in dtypes:

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Multiplies the two inputs.
                return x * y

            cnt = CompileCounterWithBackend("aot_eager")
            fn_opt = torch.compile(fn, backend=cnt)
            x = torch.randn(5, dtype=dtype, requires_grad=True)
            y1 = 1.00048828125
            y2 = 1.00048828126
            y3 = 1.00048828127

            self.assertEqual(fn_opt(x, y1), fn(x, y1))
            self.assertEqual(fn_opt(x, y2), fn(x, y2))
            self.assertEqual(fn_opt(x, y3), fn(x, y3))
            self.assertEqual(cnt.frame_count, 1)

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorfiy_python_scalars_1(self):
        # [FUNCTION DOC] With scalar capture, adding a scalar read from the sum to a
        # tensor of each float dtype under aot_eager must match eager.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Adds the tensor sum read as a Python scalar to the input.
            y = x.sum()
            return x + y.item()

        dtypes = [torch.bfloat16, torch.float16, torch.float32, torch.float64]
        for dtype in dtypes:
            x = torch.ones(3, 3, dtype=dtype)
            self.assertEqual(f(x), x + x.sum().item())

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorfiy_python_scalars_2(self):
        # [FUNCTION DOC] Checks a float32 scalar of 1e20 squared and multiplied by a
        # float64 ones scalar tensor matches eager in float64 precision.
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the scalar read
            # twice with a float64 ones scalar tensor.
            return x.item() * x.item() * torch.ones((), dtype=torch.float64)

        x = torch.tensor(1e20, dtype=torch.float32)
        self.assertEqual(
            f(x), x.item() * x.item() * torch.ones((), dtype=torch.float64)
        )

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_tensorfiy_python_scalars_3(self):
        # [FUNCTION DOC] Checks a float16 max value scalar times 101 times a
        # float32 tensor matches eager (no overflow in half precision).
        @torch.compile(backend="aot_eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Multiplies the scalar read by
            # 101 and then by a one-element float32 tensor.
            y = x.item() * 101
            return y * torch.tensor([1], dtype=torch.float32)

        finfo_float16 = torch.finfo(torch.float16)
        x = torch.tensor([finfo_float16.max], dtype=torch.float16)
        self.assertEqual(f(x), x.item() * 101 * torch.tensor([1], dtype=torch.float32))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=False)
    def test_unspec_float_input_f64(self):
        # [FUNCTION DOC] With float specialization off and dynamic default, a 17-digit float
        # added to a float64 tensor must match eager (precision retained in float64).
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of its inputs.
            return x + y

        cf = torch.compile(backend=cnts, fullgraph=True)(f)

        x = torch.zeros(3, dtype=torch.float64)
        # 17 digits of precision so unrepresentable in float32
        flt = 1.2345678901234567
        self.assertEqual(f(x, flt), cf(x, flt))

    @torch._dynamo.config.patch(specialize_float=False, assume_static_by_default=True)
    def test_unspec_float_output(self):
        # [FUNCTION DOC] With float specialization off, a float input that is
        # multiplied and returned works for three values with fullgraph.
        cnts = torch._dynamo.testing.CompileCounter()

        def f(x, y):
            # [FUNCTION DOC] HELPER: Returns the input plus one and the float doubled.
            return x + 1, y * 2

        cf = torch.compile(backend=cnts, fullgraph=True)(f)
        x = torch.randn(3)

        self.assertEqual(f(x, 3.0), cf(x, 3.0))
        self.assertEqual(f(x, 4.0), cf(x, 4.0))
        self.assertEqual(f(x, 5.0), cf(x, 5.0))

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_data_dependent_evaluate_expr_graph_break(self):
        # [FUNCTION DOC] With scalar capture, a data-dependent condition via a
        # helper graph breaks the function; asserts the continuation is
        # compiled with two frames and four ops (inline expected).
        cnts = torch._dynamo.testing.CompileCounter()

        # To ensure that the continuation frame is compiled,
        # have to write the test function in this funny way.
        # See https://github.com/pytorch/pytorch/issues/111918
        def test(y):
            # [FUNCTION DOC] HELPER: Returns whether the
            # value exceeds two, as a separate frame.
            if y > 2:
                return True
            else:
                return False

        @torch.compile(backend=cnts)
        def fn(x):
            # [FUNCTION DOC] HELPER: Adds one, reads a scalar, then
            # multiplies by two or three based on the helper result.
            x = x + 1
            y = x.item()
            if test(y):
                return x * 2
            else:
                return x * 3

        x = torch.tensor([3.0])
        fn(x)

        self.assertExpectedInline(cnts.frame_count, """2""")
        self.assertExpectedInline(cnts.op_count, """4""")

    def test_prune_torch_check(self):
        # [FUNCTION DOC] Captures the generated graph code log of the output graph; for a
        # dynamic fullgraph function with two checks, expects exactly two scalar assert
        # ops and shows the y-plus-five and x-size expressions in the log.
        log_stream, ctx = logs_to_string("torch._dynamo.output_graph", "graph_code")

        @torch.compile(fullgraph=True, dynamic=True, backend="eager")
        def f(x, y):
            # [FUNCTION DOC] HELPER: Checks the int plus five
            # equals 85 and dim zero equals 80.
            torch._check(y + 5 == 85)
            torch._check(x.size(0) == 80)

        with ctx():
            f(torch.randn(80, 100), 80)

        out = "\n".join(log_stream.getvalue().strip().split("\n")[3:]).strip()
        self.assertEqual(out.count("torch.ops.aten._assert_scalar.default"), 2)
        self.assertRegex(out, r"l_(y|args_1)_ \+ 5")
        self.assertRegex(out, r"l_(x|args_0)_\.size\(0\)")

    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_split_aot_autograd(self):
        # [FUNCTION DOC] With scalar capture, splitting a tensor by sizes
        # taken from a list conversion of an int tensor under aot_eager
        # fullgraph; just prints the result (no assertion).
        @torch.compile(backend="aot_eager", fullgraph=True)
        def f(x, i):
            # [FUNCTION DOC] HELPER: Converts the index tensor to
            # a list of two ints, splits x by them.
            y, z = i.tolist()
            return torch.split(x, [y, z])

        print(f(torch.randn(10, requires_grad=True), torch.tensor([7, 3])))

    def test_bool_tensor_ctor(self):
        # [FUNCTION DOC] A tensor built from a boolean comparison of a computed size against
        # zero, with dynamic fullgraph; true for size eight, false for size thirteen.
        cnts = torch._dynamo.testing.CompileCounter()

        @torch.compile(backend=cnts, dynamic=True, fullgraph=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Allocates an empty tensor with size rounded down to a
            # multiple of 13 and returns a tensor from whether it has zero elements.
            y = torch.empty((x.size(0) // 13) * 13)
            return torch.tensor(y.numel() == 0)

        self.assertTrue(f(torch.empty(8)).item())
        self.assertFalse(f(torch.empty(13)).item())

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked(self):
        # [FUNCTION DOC] With error on recompile on, a model compiled with dynamic
        # and max-autotune whose first dim is marked unbacked must run for batch
        # sizes 3 and 1 with same results as eager and no recompile.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module whose forward
            # doubles the input and ignores the int.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Calls the base module initializer only.
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the input doubled; the int value is unused.
                return x * 2

        main_model = TestModel()
        opt_model = torch.compile(main_model, mode="max-autotune", dynamic=True)  # noqa: UNSPECIFIED_BACKEND

        x1 = torch.rand(3, 5, 4, 8)
        x2 = torch.rand(1, 5, 4, 8)

        torch._dynamo.decorators.mark_unbacked(x1, 0)

        o1_ref = main_model(x1, 2)
        o1 = opt_model(x1, 2)
        self.assertEqual(o1_ref, o1)

        o1_2_ref = main_model(x2, 2)
        o1_2 = opt_model(x2, 2)
        self.assertEqual(o1_2_ref, o1_2)

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked_hint_consistency(self):
        # [FUNCTION DOC] With error on recompile on, an unbacked size-one
        # input: a size-oblivious check of the size not equal to one must
        # take the true hint branch (result input plus three).
        from torch.fx.experimental.symbolic_shapes import guard_size_oblivious

        x = torch.randn(1)
        torch._dynamo.decorators.mark_unbacked(x, 0)

        @torch.compile(backend="eager")
        def f(x):
            # [FUNCTION DOC] HELPER: Branches on size-oblivious
            # inequality of the size with one: adds three or four.
            if guard_size_oblivious(x.size(0) != 1):
                return x + 3
            else:
                return x + 4

        self.assertEqual(f(x), x + 3)

    @torch._dynamo.config.patch(error_on_recompile=True)
    def test_mark_unbacked_channels_last(self):
        # [FUNCTION DOC] Same as the unbacked test but with channels_last inputs; no
        # recompile allowed and outputs equal eager for batch sizes 3 and 1.
        class TestModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Minimal module whose forward
            # doubles the input and ignores the int.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Calls the base module initializer only.
                super().__init__()

            def forward(self, x: torch.Tensor, val: int) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the input doubled; the int value is unused.
                return x * 2

        main_model = TestModel()
        opt_model = torch.compile(main_model, mode="max-autotune", dynamic=True)  # noqa: UNSPECIFIED_BACKEND

        x1 = torch.rand(3, 5, 4, 8).to(memory_format=torch.channels_last)
        x2 = torch.rand(1, 5, 4, 8).to(memory_format=torch.channels_last)

        torch._dynamo.decorators.mark_unbacked(x1, 0)

        o1_ref = main_model(x1, 2)
        o1 = opt_model(x1, 2)
        self.assertEqual(o1_ref, o1)

        o1_2_ref = main_model(x2, 2)
        o1_2 = opt_model(x2, 2)
        self.assertEqual(o1_2_ref, o1_2)

    def test_float_guard_source_on_recompile(self):
        # [FUNCTION DOC] Regression: float attributes that go dynamic after recompiling must
        # have a proper source in guard logs. Compiles modules with keys 1, 2, 3 while
        # capturing the guards log, and asserts the text "unknown source" never appears.
        # Regression test: when a float attribute triggers recompilation and
        # becomes dynamic, the guard produced by produce_guards_verbose should
        # have a proper source annotation, not "(unknown source)".
        cache = {}

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a float key and registering a
            # random vector in the shared cache mapping under that key.
            def __init__(self, key: float):
                # [FUNCTION DOC] HELPER: Stores the key and puts a
                # random vector in the cache for it.
                super().__init__()
                self.key = key
                cache[key] = torch.randn(16)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Adds the cached vector
                # for this module key to the input.
                return x + cache[self.key]

        x = torch.randn(16)
        log_stream, ctx = logs_to_string("torch._dynamo.guards", "guards")
        with ctx():
            for key in [1.0, 2.0, 3.0]:
                model = torch.compile(Module(key))  # noqa: UNSPECIFIED_BACKEND
                model(x)

        guard_log = log_stream.getvalue()
        self.assertNotIn("unknown source", guard_log)


class UnspecTestsDevice(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Device-generic test class; per-device subclasses are generated by
    # instantiate_device_type_tests and receive a device argument. Tests compile with
    # a counter backend and check recompilation and device placement.
    @torch._dynamo.config.patch(assume_static_by_default=False)
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Platform does not support efficient attention",
    )
    def test_no_recompilations_with_efficient_attention(self, device):
        # [FUNCTION DOC] Skipped if the platform lacks memory-efficient attention;
        # raises SkipTest on cpu. Compiles scaled dot product attention with the
        # efficient backend using two kv lengths (15 and 16) under
        # assume_static_by_default False and requires a single frame (no recompile).
        if self.device_type == "cpu":
            raise unittest.SkipTest("EFFICIENT_ATTENTION requires a non-CPU device")

        def fn(q, k, v, attn_mask):
            # [FUNCTION DOC] HELPER: Runs scaled dot product attention restricted
            # to the efficient backend with the mask and scale one.
            from torch.nn.attention import sdpa_kernel, SDPBackend
            from torch.nn.functional import scaled_dot_product_attention

            with sdpa_kernel(backends=[SDPBackend.EFFICIENT_ATTENTION]):
                return scaled_dot_product_attention(
                    q, k, v, attn_mask=attn_mask, scale=1.0
                )

        def make_q_k_v_mask(batch, num_heads, head_dim, seq_len_kv):
            # [FUNCTION DOC] HELPER: Builds float16 random query, key, value tensors
            # requiring grad on the device and a random float mask for the given batch,
            # heads, head dim and kv length. Query length is fixed at 64.
            from collections import namedtuple
            from functools import partial

            dtype = torch.float16
            make_tensor = partial(
                torch.rand, device=device, dtype=dtype, requires_grad=True
            )
            seq_len_q = 64
            SdpaShape = namedtuple(
                "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
            )
            query = make_tensor(SdpaShape(batch, num_heads, seq_len_q, head_dim))
            kv_shape = SdpaShape(batch, num_heads, seq_len_kv, head_dim)
            key, value = make_tensor(kv_shape), make_tensor(kv_shape)
            mask = torch.randn(
                (batch, num_heads, seq_len_q, seq_len_kv), device=device, dtype=dtype
            )

            return query, key, value, mask

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        q, k, v, mask = make_q_k_v_mask(16, 16, 64, 15)
        opt_fn(q, k, v, mask)

        q, k, v, mask = make_q_k_v_mask(16, 16, 64, 16)
        opt_fn(q, k, v, mask)

        self.assertEqual(cnts.frame_count, 1)

    def test_builtin_functions_on_device(self, device):
        # [FUNCTION DOC] A module call on the device with a float scaler
        # (unspecialized) must match eager in value and in device.
        def fn(x, scaler):
            # [FUNCTION DOC] HELPER: Creates a relu module moved to the
            # device, applies to the input and scales by the float.
            m = torch.nn.ReLU()
            m.to(device)
            y = m(x) * scaler
            return y

        x = torch.randn([3, 6], device=device)
        scaler = 0.23  # 0.23 is unspecialized
        ref = fn(x, scaler)
        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)
        res = opt_fn(x, scaler)
        self.assertTrue(same(ref, res))
        self.assertEqual(ref.device, res.device)


instantiate_device_type_tests(UnspecTestsDevice, globals(), allow_xpu=True)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
