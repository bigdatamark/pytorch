# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo's lazy constant handling: Python constants (str, bool, float,
# int) passed into compiled functions should not be guarded on value unless used in control
# flow or math, and constants computed from them by simple operators stay lazy until needed.
# Tests count frames with CompileCounter and compare compiled output with eager. No import-
# time side effects beyond imports; run_tests is called when executed as a script.

import keyword
import sys

import torch
import torch._dynamo
from torch._dynamo.test_case import run_tests, TestCase
from torch._dynamo.testing import CompileCounter, same


class LazyConstantVariableTests(TestCase):
    # [CLASS DOC] Test case verifying when passed-in constants cause recompilation: pass-
    # through values do not, while branching, math, or realization (keyword check, membership,
    # dim argument) do. Type-only queries add a type guard only.
    def _assert_compile_count(self, fn, arg_sets, expected_frames):
        # [FUNCTION DOC] HELPER: Compiles the given function with a CompileCounter backend,
        # then for each argument tuple runs eager and compiled versions and asserts the
        # outputs match via same. Finally asserts the counter frame count equals the expected
        # number of frames.
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        for args in arg_sets:
            eager = fn(*args)
            compiled = opt_fn(*args)
            self.assertTrue(same(eager, compiled))

        self.assertEqual(counter.frame_count, expected_frames)

    def test_passthrough_python_constants_do_not_guard(self):
        # [FUNCTION DOC] For string, bool, float and int cases, a function returning a tensor
        # transform plus the constant is compiled once; two different constant values must
        # still need only one frame, using a subtest per type.
        tensor_input = torch.ones(2)
        cases = [
            ("str", "alpha", "beta"),
            ("bool", True, False),
            ("float", 2.5, -3.0),
            ("int", 11, 42),
        ]

        for name, first, second in cases:
            with self.subTest(name=name):

                def fn(t, constant):
                    # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the
                    # sine of the tensor plus one together with the unchanged constant.
                    return t.sin() + 1, constant

                self._assert_compile_count(
                    fn,
                    [(tensor_input, first), (tensor_input, second)],
                    expected_frames=1,
                )

    def test_branching_on_constant_recompiles(self):
        # [FUNCTION DOC] Function branching on a string equality: calls with neg and then pos
        # both match eager and the frame count ends above one.
        tensor_input = torch.randn(2)

        def fn(t, token):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Subtracts one from
            # the tensor if the token equals neg, else adds one.
            if token == "neg":
                return t - 1
            return t + 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        eager0 = fn(tensor_input, "neg")
        compiled0 = opt_fn(tensor_input, "neg")
        self.assertTrue(same(eager0, compiled0))

        eager1 = fn(tensor_input, "pos")
        compiled1 = opt_fn(tensor_input, "pos")
        self.assertTrue(same(eager1, compiled1))

        self.assertGreater(counter.frame_count, 1)

    def test_numeric_constant_participating_in_math_recompiles(self):
        # [FUNCTION DOC] Multiplying a tensor by a float argument: two different scales match
        # eager and force more than one frame.
        tensor_input = torch.randn(3)

        def fn(t, scale):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the tensor
            # times the scale.
            return t * scale

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        eager0 = fn(tensor_input, 0.1)
        compiled0 = opt_fn(tensor_input, 0.1)
        self.assertTrue(same(eager0, compiled0))

        eager1 = fn(tensor_input, 2.0)
        compiled1 = opt_fn(tensor_input, 2.0)
        self.assertTrue(same(eager1, compiled1))

        self.assertGreater(counter.frame_count, 1)

    @torch._dynamo.config.patch(specialize_int=False, assume_static_by_default=False)
    def test_lazy_constant_int_comparison_becomes_symnode(self):
        # [FUNCTION DOC] Config patched to not specialize ints and not assume static shapes.
        # Comparing an int argument to a literal inside a function must still compile and
        # match eager when the lazy int resolves to a symbolic node.
        """Test that comparing lazy constant ints works when they realize to
        SymNodeVariable.

        The compare_by_value handler assumes ConstantVariable args (.value
        access).  When specialize_int=False makes the lazy int resolve to
        SymNodeVariable, the lazy_constant_handler must re-dispatch to the
        symbolic comparison path.
        """

        def fn(x, n):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sum of
            # the tensor if n equals two, else its mean.
            if n == 2:
                return x.sum()
            return x.mean()

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(3, 4)

        eager = fn(x, 2)
        compiled = opt_fn(x, 2)
        self.assertTrue(same(eager, compiled))

    def test_slice_indices_unspecialized_ints(self):
        # [FUNCTION DOC] With unspecialized ints, a function indexing a list by an int or a
        # slice (appending a constant to slice results) is compiled with full graph; compiled
        # and eager results must match for two slices and one integer index.
        """Test that slice indices work with unspecialized ints.

        This tests that LazyConstantVariable properly interacts with the
        unspecialized int (symint) codepath. When specialize_int=False,
        integers become symbolic, and slice objects containing these
        symbolic ints should still work correctly.
        """
        layers = list(range(10))

        def getitem(a, idx):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns a tuple of a
            # zero tensor and the indexed list, adding an extra element to slice results.
            if isinstance(idx, slice):
                return (
                    torch.zeros(1),
                    a[idx]
                    + [
                        100,
                    ],
                )
            return (torch.zeros(1), a[idx])

        cnts = CompileCounter()

        with torch._dynamo.config.patch(
            assume_static_by_default=False,
            specialize_int=False,
            specialize_float=True,
        ):
            opt_getitem = torch.compile(getitem, backend=cnts, fullgraph=True)
            ref0 = getitem(layers, slice(0, 2, 1))
            ref1 = getitem(layers, 2)
            ref2 = getitem(layers, slice(3, 8, 2))

            res0 = opt_getitem(layers, slice(0, 2, 1))
            res1 = opt_getitem(layers, 2)
            res2 = opt_getitem(layers, slice(3, 8, 2))

        self.assertEqual(ref0, res0)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_range_iteration_with_unspec_ints(self):
        # [FUNCTION DOC] With unspecialized ints, a function with a graph break followed by a
        # comprehension filtering a range through a membership test is compiled with the eager
        # backend; output must match eager.
        def make_fn():
            # [FUNCTION DOC] HELPER: Builds a closure with a range of ten and an allowed list,
            # and returns the inner function.
            keys = range(10)
            allowed = [0, 1, 2, 3]

            def fn(x):
                # [FUNCTION DOC] HELPER: Function under compile in this test. Adds one to x,
                # forces a graph break, then adds the first range element that is in the
                # allowed list.
                x = x + 1
                torch._dynamo.graph_break()
                key = [key for key in keys if key in allowed]
                return x + key[0]

            return fn

        fn = make_fn()
        tensor_input = torch.ones(3)
        with torch._dynamo.config.patch(
            assume_static_by_default=False, specialize_int=False
        ):
            opt_fn = torch.compile(fn, backend="eager")
            self.assertTrue(same(fn(tensor_input), opt_fn(tensor_input)))

    def test_keyword_iskeyword(self):
        # [FUNCTION DOC] Calls keyword.iskeyword with the strings if and foo; both match eager
        # and the frame count exceeds one, since the result is branched on.
        """Test that keyword.iskeyword is properly traced."""

        def fn(x, word):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Adds one if the word
            # is a Python keyword, else subtracts one.
            if keyword.iskeyword(word):
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        # "if" is a Python keyword
        eager_kw = fn(tensor_input, "if")
        compiled_kw = opt_fn(tensor_input, "if")
        self.assertTrue(same(eager_kw, compiled_kw))

        # "foo" is not a keyword
        eager_not_kw = fn(tensor_input, "foo")
        compiled_not_kw = opt_fn(tensor_input, "foo")
        self.assertTrue(same(eager_not_kw, compiled_not_kw))

        # Both should cause recompilation since they branch differently
        self.assertGreater(counter.frame_count, 1)

    def test_keyword_iskeyword_no_branch(self):
        # [FUNCTION DOC] Same check without branching: the boolean result is returned; class
        # and notakeyword match eager and cause recompilation because the value is needed.
        """Test keyword.iskeyword without branching on result.

        Even though we don't branch on the result, iskeyword() needs the
        actual string value to compute its result, so different strings
        will still cause recompilation.
        """

        def fn(x, word):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the tensor
            # plus one with the keyword check result.
            result = keyword.iskeyword(word)
            return x + 1, result

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        eager0 = fn(tensor_input, "class")
        compiled0 = opt_fn(tensor_input, "class")
        self.assertEqual(eager0[1], compiled0[1])
        self.assertTrue(same(eager0[0], compiled0[0]))

        eager1 = fn(tensor_input, "notakeyword")
        compiled1 = opt_fn(tensor_input, "notakeyword")
        self.assertEqual(eager1[1], compiled1[1])
        self.assertTrue(same(eager1[0], compiled1[0]))

        # Recompiles because iskeyword needs the actual value
        self.assertGreater(counter.frame_count, 1)

    def test_frozenset_contains(self):
        # [FUNCTION DOC] Membership in a closed-over frozenset of strings used in a branch:
        # alpha and delta match eager and cause more than one frame.
        """Test that frozenset.__contains__ is properly traced."""
        valid_options = frozenset({"alpha", "beta", "gamma"})

        def fn(x, option):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Adds one if the
            # option is in the closed-over frozenset, else subtracts one.
            if option in valid_options:
                return x + 1
            return x - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        # "alpha" is in the frozenset
        eager_in = fn(tensor_input, "alpha")
        compiled_in = opt_fn(tensor_input, "alpha")
        self.assertTrue(same(eager_in, compiled_in))

        # "delta" is not in the frozenset
        eager_not_in = fn(tensor_input, "delta")
        compiled_not_in = opt_fn(tensor_input, "delta")
        self.assertTrue(same(eager_not_in, compiled_not_in))

        # Both should cause recompilation since they branch differently
        self.assertGreater(counter.frame_count, 1)

    def test_frozenset_contains_no_branch(self):
        # [FUNCTION DOC] Membership result returned rather than branched on, with ints two and
        # five; results match eager and recompilation happens.
        """Test frozenset.__contains__ without branching on result.

        Even though we don't branch on the result, the `in` operator needs
        the actual value to check membership, so different values will
        still cause recompilation.
        """
        valid_options = frozenset({1, 2, 3})

        def fn(x, value):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the tensor
            # plus one and whether the value is in the frozenset.
            result = value in valid_options
            return x + 1, result

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        tensor_input = torch.randn(3)

        eager0 = fn(tensor_input, 2)
        compiled0 = opt_fn(tensor_input, 2)
        self.assertEqual(eager0[1], compiled0[1])
        self.assertTrue(same(eager0[0], compiled0[0]))

        eager1 = fn(tensor_input, 5)
        compiled1 = opt_fn(tensor_input, 5)
        self.assertEqual(eager1[1], compiled1[1])
        self.assertTrue(same(eager1[0], compiled1[0]))

        # Recompiles because `in` needs the actual value
        self.assertGreater(counter.frame_count, 1)

    def test_tensor_method_with_lazy_kwargs(self):
        # [FUNCTION DOC] Sum over a dim given as keyword argument, with dim zero then one;
        # results match eager and more than one frame results.
        """Test that tensor methods work when kwargs contain lazy constants."""

        def fn(x, dim):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sum of x
            # over the given dim.
            # sum() with dim kwarg - the dim value goes through as lazy constant
            return x.sum(dim=dim)

        tensor_input = torch.randn(3, 4, 5)
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # Different dim values should cause recompile since dim affects the output
        eager0 = fn(tensor_input, 0)
        compiled0 = opt_fn(tensor_input, 0)
        self.assertTrue(same(eager0, compiled0))

        eager1 = fn(tensor_input, 1)
        compiled1 = opt_fn(tensor_input, 1)
        self.assertTrue(same(eager1, compiled1))

        # This should recompile because dim is used in the computation
        self.assertGreater(counter.frame_count, 1)

    def test_global_constant_passthrough(self):
        # [FUNCTION DOC] Mutable dict holders stand in for globals. A compiled function
        # returning a holder value alongside a tensor op is called, the holder value is
        # changed, and it is called again: the new value is returned with no recompile (frame
        # count stays one) for a string holder and an int holder.
        """Test that global constants benefit from lazy constant optimization.

        When a global/closure constant is just passed through without being
        used in control flow or math, changing its value should not cause
        recompilation.
        """
        tensor_input = torch.randn(3)

        # Use a mutable container to allow modifying the "global" value
        global_holder = {"value": "hello"}

        def fn(t):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor plus one with the string held in the dict.
            return t.sin() + 1, global_holder["value"]

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call
        result1 = opt_fn(tensor_input)
        self.assertEqual(result1[1], "hello")
        self.assertEqual(counter.frame_count, 1)

        # Modify the global and call again - should NOT recompile
        global_holder["value"] = "world"
        result2 = opt_fn(tensor_input)
        self.assertEqual(result2[1], "world")
        self.assertEqual(counter.frame_count, 1)  # No recompile!

        # Also test with int
        int_holder = {"value": 42}

        def fn_int(t):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the cosine
            # of the tensor plus one with the int held in the dict.
            return t.cos() + 1, int_holder["value"]

        counter2 = CompileCounter()
        opt_fn2 = torch.compile(fn_int, backend=counter2)

        result3 = opt_fn2(tensor_input)
        self.assertEqual(result3[1], 42)
        self.assertEqual(counter2.frame_count, 1)

        int_holder["value"] = 100
        result4 = opt_fn2(tensor_input)
        self.assertEqual(result4[1], 100)
        self.assertEqual(counter2.frame_count, 1)  # No recompile!

    def test_type_change_does_not_recompile(self):
        # [FUNCTION DOC] Pass-through value: int values forty-two then one hundred, then
        # strings hello and world, all keep the frame count at one, and results match eager
        # each time, so even a change of value type does not recompile for pass-through.
        tensor_input = torch.randn(3)

        def fn(t, val):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the tensor
            # plus one with the unchanged value.
            # Just pass through - should not guard on value, only type
            return t + 1, val

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with int
        eager1 = fn(tensor_input, 42)
        compiled1 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])
        self.assertEqual(counter.frame_count, 1)

        # Second call with different int value - should NOT recompile
        eager2 = fn(tensor_input, 100)
        compiled2 = opt_fn(tensor_input, 100)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(eager2[1], compiled2[1])
        self.assertEqual(counter.frame_count, 1)  # Still 1, no recompile

        # Third call with string
        eager3 = fn(tensor_input, "hello")
        compiled3 = opt_fn(tensor_input, "hello")
        self.assertTrue(same(eager3[0], compiled3[0]))
        self.assertEqual(eager3[1], compiled3[1])
        self.assertEqual(counter.frame_count, 1)

        # Fourth call with different string
        eager4 = fn(tensor_input, "world")
        compiled4 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager4[0], compiled4[0]))
        self.assertEqual(eager4[1], compiled4[1])
        self.assertEqual(counter.frame_count, 1)

    def test_type_does_not_recompile_on_value_change(self):
        # [FUNCTION DOC] Using the type builtin in a branch only installs a type guard: two
        # different strings keep one frame, then passing an int yields a second frame. Results
        # match eager throughout.
        """Test that type() checks do NOT trigger recompilation on value change.

        When type() is called on a LazyConstantVariable, it only installs a
        TYPE_MATCH guard (not CONSTANT_MATCH), so different values of the same type
        do not cause recompilation. This is similar to isinstance() behavior but
        tests a different code path.

        Note: We use string values here because with specialize_int=False (the default),
        int values must be realized during handler dispatch to determine if they become
        ConstantVariable or SymNodeVariable. Strings always become ConstantVariable.
        """
        tensor_input = torch.randn(3)

        def fn(t, val):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Adds one if the
            # value type is str, else subtracts one.
            if type(val) is str:
                return t + 1
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with string
        eager1 = fn(tensor_input, "hello")
        compiled1 = opt_fn(tensor_input, "hello")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call with different string - should NOT recompile since
        # type() only installs TYPE_MATCH guard
        eager2 = fn(tensor_input, "world")
        compiled2 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 1)  # No recompilation!

        # Third call with int - should recompile due to type change
        eager3 = fn(tensor_input, 42)
        compiled3 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(counter.frame_count, 2)  # Recompile for type change

    def test_isinstance_does_not_recompile_on_value_change(self):
        # [FUNCTION DOC] Same pattern with the isinstance builtin: two strings, one frame; an
        # int causes a second frame.
        """Test that isinstance checks do NOT trigger recompilation on value change.

        When isinstance() is called on a LazyConstantVariable, it only installs a
        TYPE_MATCH guard (not CONSTANT_MATCH), so different values of the same type
        do not cause recompilation.

        Note: We use string values here because with specialize_int=False (the default),
        int values must be realized during handler dispatch to determine if they become
        ConstantVariable or SymNodeVariable. Strings always become ConstantVariable.
        """
        tensor_input = torch.randn(3)

        def fn(t, val):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Adds one if the
            # value is an instance of str, else subtracts one.
            if isinstance(val, str):
                return t + 1
            return t - 1

        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)

        # First call with string
        eager1 = fn(tensor_input, "hello")
        compiled1 = opt_fn(tensor_input, "hello")
        self.assertTrue(same(eager1, compiled1))
        self.assertEqual(counter.frame_count, 1)

        # Second call with different string - should NOT recompile since
        # isinstance only installs TYPE_MATCH guard
        eager2 = fn(tensor_input, "world")
        compiled2 = opt_fn(tensor_input, "world")
        self.assertTrue(same(eager2, compiled2))
        self.assertEqual(counter.frame_count, 1)  # No recompilation!

        # Third call with int - should recompile due to type change
        eager3 = fn(tensor_input, 42)
        compiled3 = opt_fn(tensor_input, 42)
        self.assertTrue(same(eager3, compiled3))
        self.assertEqual(counter.frame_count, 2)  # Recompile for type change

    def test_container_with_lazy_constant_no_recompile(self):
        # [FUNCTION DOC] Returning containers that hold the constant: a list with an int, a
        # tuple with a string, and a list with an int and a string. For each, a second call
        # with different values must match eager and not recompile (frame count remains one).
        """Test that returning containers with LazyConstantVariables works without realization.

        LazyConstantVariables inside returned containers (list, tuple) should stay
        lazy and not cause recompilation when the constant value changes.
        """
        tensor_input = torch.randn(3)

        # Test returning a list with constants
        def fn_list(t, val):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns a list of
            # the tensor plus one and the value.
            return [t + 1, val]

        counter_list = CompileCounter()
        opt_fn_list = torch.compile(fn_list, backend=counter_list)

        eager1 = fn_list(tensor_input, 42)
        compiled1 = opt_fn_list(tensor_input, 42)
        self.assertTrue(same(eager1[0], compiled1[0]))
        self.assertEqual(eager1[1], compiled1[1])
        self.assertEqual(counter_list.frame_count, 1)

        # Different value should NOT recompile
        eager2 = fn_list(tensor_input, 100)
        compiled2 = opt_fn_list(tensor_input, 100)
        self.assertTrue(same(eager2[0], compiled2[0]))
        self.assertEqual(eager2[1], compiled2[1])
        self.assertEqual(counter_list.frame_count, 1)

        # Test returning a tuple with constants
        def fn_tuple(t, val):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns a tuple of
            # the tensor plus one and the value.
            return (t + 1, val)

        counter_tuple = CompileCounter()
        opt_fn_tuple = torch.compile(fn_tuple, backend=counter_tuple)

        eager3 = fn_tuple(tensor_input, "hello")
        compiled3 = opt_fn_tuple(tensor_input, "hello")
        self.assertTrue(same(eager3[0], compiled3[0]))
        self.assertEqual(eager3[1], compiled3[1])
        self.assertEqual(counter_tuple.frame_count, 1)

        # Different string should NOT recompile
        eager4 = fn_tuple(tensor_input, "world")
        compiled4 = opt_fn_tuple(tensor_input, "world")
        self.assertTrue(same(eager4[0], compiled4[0]))
        self.assertEqual(eager4[1], compiled4[1])
        self.assertEqual(counter_tuple.frame_count, 1)

        # Test returning list with multiple constants
        def fn_multi(t, val1, val2):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns a list of
            # the tensor plus one and the two values.
            return [t + 1, val1, val2]

        counter_multi = CompileCounter()
        opt_fn_multi = torch.compile(fn_multi, backend=counter_multi)

        eager5 = fn_multi(tensor_input, 1, "a")
        compiled5 = opt_fn_multi(tensor_input, 1, "a")
        self.assertTrue(same(eager5[0], compiled5[0]))
        self.assertEqual(eager5[1], compiled5[1])
        self.assertEqual(eager5[2], compiled5[2])
        self.assertEqual(counter_multi.frame_count, 1)

        # Different values should NOT recompile
        eager6 = fn_multi(tensor_input, 2, "b")
        compiled6 = opt_fn_multi(tensor_input, 2, "b")
        self.assertTrue(same(eager6[0], compiled6[0]))
        self.assertEqual(eager6[1], compiled6[1])
        self.assertEqual(eager6[2], compiled6[2])
        self.assertEqual(counter_multi.frame_count, 1)


class ComputedLazyConstantTests(TestCase):
    # [CLASS DOC] Test case for constants derived from inputs by operators (add, subtract,
    # multiply, inplace add, chains): unused derived values should not recompile, but
    # division, operand type changes, branching, tensor math, container keys or indices, and
    # over-budget chains do. Frame counts are asserted exactly.
    def _check(self, fn, arg_sets, expected_frames):
        # [FUNCTION DOC] HELPER: Compiles the function with a CompileCounter backend, runs
        # each argument tuple eager and compiled asserting matching outputs via same, then
        # asserts the frame count equals the expected number.
        counter = CompileCounter()
        opt_fn = torch.compile(fn, backend=counter)
        for args in arg_sets:
            eager = fn(*args)
            compiled = opt_fn(*args)
            self.assertTrue(same(eager, compiled))
        self.assertEqual(counter.frame_count, expected_frames)

    def test_unused_computed_int_does_not_recompile(self):
        # [FUNCTION DOC] A sum of two int arguments is returned but unused otherwise; three
        # argument sets compile in one frame.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor and the sum of a and b.
            return t.sin(), a + b

        self._check(fn, [(t, 1, 2), (t, 3, 4), (t, 100, -5)], expected_frames=1)

    def test_unused_computed_ops_do_not_recompile(self):
        # [FUNCTION DOC] Three lambda cases (int subtraction, int multiplication, string
        # concatenation) each run after resetting dynamo; each must compile with one frame
        # across two argument sets, using a subtest per case.
        t = torch.ones(2)
        cases = [
            (lambda t, a, b: (t.sin(), a - b), [(t, 5, 2), (t, 9, 3)]),
            (lambda t, a, b: (t.sin(), a * b), [(t, 5, 2), (t, 9, 3)]),
            (lambda t, a, b: (t.sin(), a + b), [(t, "x", "y"), (t, "p", "q")]),
        ]
        for i, (fn, arg_sets) in enumerate(cases):
            with self.subTest(case=i):
                torch._dynamo.reset()
                self._check(fn, arg_sets, expected_frames=1)

    def test_unused_computed_inplace_op_does_not_recompile(self):
        # [FUNCTION DOC] Inplace addition on an int argument returned with a tensor result;
        # two argument sets use one frame.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Adds b into a in
            # place and returns the sine of the tensor and a.
            a += b
            return t.sin(), a

        self._check(fn, [(t, 1, 2), (t, 3, 4)], expected_frames=1)

    def test_unused_division_recompiles(self):
        # [FUNCTION DOC] Dividing two int arguments and returning the quotient needs two
        # frames across two argument sets.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor and a divided by b.
            return t.sin(), a / b

        self._check(fn, [(t, 4, 2), (t, 9, 3)], expected_frames=2)

    def test_operand_type_change_recompiles(self):
        # [FUNCTION DOC] Sum of two args with int, int, then float arguments: two frames in
        # total.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor and a plus b.
            return t.sin(), a + b

        self._check(fn, [(t, 1, 2), (t, 3, 4), (t, 1.5, 2.5)], expected_frames=2)

    def test_chained_computed_constants_do_not_recompile(self):
        # [FUNCTION DOC] A chained expression of several operations on two ints stays lazy,
        # one frame across two argument sets.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor and the chained arithmetic result.
            return t.sin(), (a + b) * 2 - a

        self._check(fn, [(t, 1, 2), (t, 7, 5)], expected_frames=1)

    def test_long_left_associated_chain_no_recursion_error(self):
        # [FUNCTION DOC] Accumulates a sum over a list of three hundred ints in a loop; two
        # argument lists must compile without recursion error and yield two frames.
        t = torch.ones(2)

        def fn(t, vals):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Sums the values in a
            # loop and returns the sine of the tensor and the total.
            total = 0
            for v in vals:
                total = total + v
            return t.sin(), total

        n = 300
        arg_sets = [(t, list(range(n))), (t, list(range(1, n + 1)))]
        self._check(fn, arg_sets, expected_frames=2)

    def test_recursion_limit_sized_chain_no_recursion_error(self):
        # [FUNCTION DOC] Same accumulation with a list whose length equals the interpreter
        # recursion limit; no recursion error, two frames.
        t = torch.ones(2)

        def fn(t, vals):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Sums the values in a
            # loop and returns the sine of the tensor and the total.
            total = 0
            for v in vals:
                total = total + v
            return t.sin(), total

        n = sys.getrecursionlimit()
        arg_sets = [(t, list(range(n))), (t, list(range(1, n + 1)))]
        self._check(fn, arg_sets, expected_frames=2)

    @torch._dynamo.config.patch(computed_lazy_constant_max_nodes=2)
    def test_over_budget_chain_falls_back_to_guards(self):
        # [FUNCTION DOC] With the node budget config set to two, a chain of repeated additions
        # exceeds it and falls back to guards, giving two frames for two argument sets.
        t = torch.ones(2)

        def fn(t, a):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor and four copies of a added together.
            return t.sin(), a + a + a + a

        self._check(fn, [(t, 1), (t, 2)], expected_frames=2)

    @torch._dynamo.config.patch(computed_lazy_constant_max_nodes=0)
    def test_disabled_computed_lazy_constant_installs_guards(self):
        # [FUNCTION DOC] With the node budget config set to zero (disabled), summing two ints
        # installs guards, so two argument sets need two frames.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the sine of
            # the tensor and a plus b.
            return t.sin(), a + b

        self._check(fn, [(t, 1, 2), (t, 3, 4)], expected_frames=2)

    def test_computed_constant_in_branch_recompiles(self):
        # [FUNCTION DOC] Branching on whether the sum of two ints exceeds five: two argument
        # sets land on different sides, giving two frames.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Adds one to the
            # tensor if a plus b exceeds five, else subtracts one.
            if a + b > 5:
                return t + 1
            return t - 1

        self._check(fn, [(t, 1, 2), (t, 7, 5)], expected_frames=2)

    def test_computed_constant_in_tensor_math_recompiles(self):
        # [FUNCTION DOC] A tensor multiplied by the sum of two floats forces realization; two
        # argument sets give two frames.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the tensor
            # times the sum of a and b.
            return t * (a + b)

        self._check(fn, [(t, 1.5, 2.0), (t, 3.25, 4.0)], expected_frames=2)

    def test_computed_constant_as_dict_key_realizes(self):
        # [FUNCTION DOC] A derived constant used as a dictionary key and for lookup realizes
        # the value; two argument sets give two frames.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Stores the tensor in
            # a dict keyed by a plus b, then reads it back by the same key and returns its
            # sine.
            d = {a + b: t}
            return d[a + b].sin()

        self._check(fn, [(t, 1, 2), (t, 3, 4)], expected_frames=2)

    def test_computed_constant_as_list_index_realizes(self):
        # [FUNCTION DOC] A derived constant used as a list index realizes the value; two
        # argument sets give two frames.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Returns the element
            # of a three-tensor list at index a plus b.
            lst = [t, t + 1, t + 2]
            return lst[a + b]

        self._check(fn, [(t, 0, 1), (t, 1, 1)], expected_frames=2)

    def test_computed_constant_type_query_does_not_recompile(self):
        # [FUNCTION DOC] Derived string constants queried only by isinstance do not recompile;
        # two argument sets give one frame.
        t = torch.ones(2)

        def fn(t, a, b):
            # [FUNCTION DOC] HELPER: Function under compile in this test. Concatenates a and
            # b, returns the sine of the tensor and the result if it is a string, else the
            # cosine.
            c = a + b
            if isinstance(c, str):
                return t.sin(), c
            return t.cos(), c

        self._check(fn, [(t, "a", "b"), (t, "c", "d")], expected_frames=1)


if __name__ == "__main__":
    run_tests()
