# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for the compiler stance that skips guard evaluation as unsafe. Covers
# reusing existing cache entries without guard checks for changed bool flags, dtypes, plain
# attribute values and keyword arguments, the error raised when a new compilation would be
# needed under the stance, and a filter that drops all guards so one cache entry has no guards
# at all.

import torch
import torch._dynamo.test_case
import torch._dynamo.testing


def my_custom_function(x):
    # [FUNCTION DOC] HELPER: Trivial module-level function returning its input plus one; not
    # used by any test in this file.
    return x + 1


class RunDiffGuardTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for the skip_guard_eval_unsafe stance. Tests first warm the compile
    # cache normally, then repeat calls inside the stance and compare against reference results.
    def test_bool_recompile(self):
        # [FUNCTION DOC] Compiles a function that multiplies or adds two tensors depending on a
        # bool flag, with the inductor backend. Runs both flag values normally for references,
        # then again inside the stance in reversed order, and asserts the results equal the
        # references.
        def fn(x, y, c):
            # [FUNCTION DOC] HELPER: Returns the product of x and y if the flag is truthy,
            # otherwise their sum.
            if c:
                return x * y
            else:
                return x + y

        opt_fn = torch.compile(fn, backend="inductor")
        x = 2 * torch.ones(4)
        y = 3 * torch.ones(4)

        ref1 = opt_fn(x, y, True)
        ref2 = opt_fn(x, y, False)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res2 = opt_fn(x, y, False)
            res1 = opt_fn(x, y, True)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_tensor_recompile(self):
        # [FUNCTION DOC] Compiles an elementwise product with the eager backend for float32
        # inputs, then for float64 inputs, getting reference results. Inside the stance repeats
        # both calls and asserts both match the references, so the cached entries for each dtype
        # are picked.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns the elementwise product of x and y.
            return x * y

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4, dtype=torch.float32)
        y = torch.randn(4, dtype=torch.float32)

        ref1 = opt_fn(x, y)

        x64 = torch.randn(4, dtype=torch.float64)
        y64 = torch.randn(4, dtype=torch.float64)
        ref2 = opt_fn(x64, y64)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res1 = opt_fn(x, y)
            res2 = opt_fn(x64, y64)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_post_recompile(self):
        # [FUNCTION DOC] Tracks recompilation caused by changes to attributes of a closed-over
        # object, using a frame-counting backend. Frame count is 1 after the first run and 2
        # after changing the first attribute. Inside the stance, resetting the attribute to its
        # original value and then to the second value still gives correct results using existing
        # entries. After leaving the stance, changing the other attribute recompiles, giving a
        # count of 3.
        class Foo:
            # [CLASS DOC] HELPER: Plain object with two integer attributes holding 4 and 5.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Sets attribute a to 4 and attribute b to 5.
                self.a = 4
                self.b = 5

        foo = Foo()

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus the closed-over object's two attributes.
            return x + foo.a + foo.b

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts)

        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 1)

        foo.a = 11
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 2)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            # Set it back to original value
            foo.a = 4
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

            foo.a = 11
            ref = fn(x)
            res = opt_fn(x)
            self.assertEqual(ref, res)

        # Check that we are back to original behavior
        foo.b = 8
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(cnts.frame_count, 3)

    def test_fail_on_tensor_shape_change(self):
        # [FUNCTION DOC] Compiles a function reading a key from a dictionary of tensors, warms
        # the cache with a vector, then inside the stance replaces the entry with a matrix and
        # calls again. Asserts a runtime error saying recompilation was triggered with the
        # stance, because the shape change would need a new compile.
        def fn(dt):
            # [FUNCTION DOC] HELPER: Returns the tensor under the key x of the dictionary plus
            # one.
            return dt["x"] + 1

        x = torch.randn(4)
        dt = {}
        dt["x"] = x
        opt_fn = torch.compile(fn, backend="eager")
        opt_fn(dt)

        with self.assertRaisesRegex(
            RuntimeError, "Recompilation triggered with skip_guard_eval_unsafe stance"
        ):
            with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
                x = torch.randn(4, 4)
                dt["x"] = x
                opt_fn(dt)

    def test_cache_line_pickup(self):
        # [FUNCTION DOC] Calls a compiled function with three keyword argument combinations
        # (neither, one, both optional flags set) normally to create entries, then again inside
        # the stance, and asserts each result equals the corresponding reference.
        def fn(x, a=None, b=None):
            # [FUNCTION DOC] HELPER: Multiplies x by 3, then by 5 if a is truthy, then by 7 if b
            # is truthy, and returns it.
            x = x * 3
            if a:
                x = x * 5
            if b:
                x = x * 7
            return x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.ones(4)

        ref1 = opt_fn(x, a=None, b=None)
        ref2 = opt_fn(x, a=1, b=None)
        ref3 = opt_fn(x, a=1, b=1)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            res1 = opt_fn(x, a=None, b=None)
            res2 = opt_fn(x, a=1, b=None)
            res3 = opt_fn(x, a=1, b=1)

        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)
        self.assertEqual(ref3, res3)

    def test_skip_all_guards_single_cache_entry(self):
        # [FUNCTION DOC] Compiles with a guard filter that drops every guard. Asserts one frame
        # compiled and one cache entry whose guard manager root has no leaf guards, accessors or
        # epilogue lambda guards. Then inside the stance a differently shaped input gives the
        # correct result and no new frame is compiled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one.
            return x + 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(
            fn,
            backend=cnts,
            options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe},
        )

        x = torch.randn(4)
        self.assertEqual(opt_fn(x), fn(x))
        self.assertEqual(cnts.frame_count, 1)

        entries = torch._dynamo.eval_frame._debug_get_cache_entry_list(fn)
        self.assertEqual(len(entries), 1)
        root = entries[0].guard_manager.root
        self.assertEqual(len(root.get_leaf_guards()), 0)
        self.assertEqual(len(root.get_accessors()), 0)
        self.assertEqual(len(root.get_epilogue_lambda_guards()), 0)

        with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
            y = torch.randn(4, 4)
            self.assertEqual(opt_fn(y), fn(y))

        self.assertEqual(cnts.frame_count, 1)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
