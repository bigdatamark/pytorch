# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests that Python builtin operators and indexing syntax compile under TorchScript
# and agree with eager Python: arithmetic, power, matrix multiply, slicing, tensor and tuple
# indexing, advanced (tensor or list) indexing, ellipsis, infinity and string-to-float conversion.
# Import-time side effect: appends the parent test directory to the module search path. Running
# the file directly raises an error pointing at test_jit.

import os
import random
import sys
import tempfile
from textwrap import dedent

import torch
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import execWrapper, JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


def get_fn(file_name, script_path):
    # [FUNCTION DOC] HELPER: Loads a Python source file as a module under the given name using
    # importlib, executes it, and returns the attribute named fn from it. Raises AttributeError if
    # the module defines no such attribute; any error from executing the file propagates.
    import importlib.util

    spec = importlib.util.spec_from_file_location(file_name, script_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    fn = module.fn
    return fn


class TestPythonBuiltinOP(JitTestCase):
    # [CLASS DOC] JitTestCase suite comparing TorchScript behavior with eager Python for builtin
    # operators and indexing forms, mostly through checkScript and through source strings compiled
    # in a CompilationUnit.
    def test_add(self):
        # [FUNCTION DOC] Checks that addition plus an in-place add of tensors requiring grad
        # scripts correctly.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Adds two inputs, then adds the first again in place, and
            # returns the result.
            c = a + b
            c += a
            return c

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_mul(self):
        # [FUNCTION DOC] Checks that multiplying two grad-requiring tensors scripts right.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of the two inputs.
            return a * b

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)

    def test_matmul_py3(self):
        # [FUNCTION DOC] Writes source defining a matrix-multiply function to a temporary file,
        # loads it with the get_fn helper, and checks it scripts correctly on random 4x3 and 3x2
        # grad-requiring tensors. The temporary directory is removed afterwards.
        code = dedent(
            """
        def fn(a, b):
            return a @ b
        """
        )

        with tempfile.TemporaryDirectory() as tmp_dir:
            script_path = os.path.join(tmp_dir, "script.py")
            with open(script_path, "w") as f:
                f.write(code)
            fn = get_fn("test_matmul_py3", script_path)

            a = torch.rand(4, 3, requires_grad=True)
            b = torch.rand(3, 2, requires_grad=True)
            self.checkScript(fn, (a, b), optimize=True)

    def test_pow(self):
        # [FUNCTION DOC] Checks the power operator in scripts: tensor operands, chained power, an
        # int base with float exponent, a constant negative exponent, and item-based scalar power
        # over all pairs of sample tensors, skipping pairs with a negative base.
        def func(a, b):
            # [FUNCTION DOC] HELPER: Returns the first input raised to the second.
            return a**b

        def func2(a, b, c, d):
            # [FUNCTION DOC] HELPER: Adds the third input to the power of the first to the power
            # of the second to the fourth (power is right associative).
            return c + a**b**d

        def func3(a, b):
            # [FUNCTION DOC] HELPER: Raises an int to a float power; relies on the legacy
            # comment-style signature on the next line.
            # type: (int, float) -> float
            return a**b

        def func4():
            # [FUNCTION DOC] HELPER: Returns two raised to a negative constant; takes no arguments
            # and is annotated through the legacy comment-style signature on the next line.
            # type: () -> float
            return 2**-2

        def func5(x, y):
            # [FUNCTION DOC] HELPER: Converts both tensor inputs to Python scalars and returns the
            # first to the power of the second.
            return x.item() ** y.item()

        a = torch.rand(1, requires_grad=True)
        b = torch.rand(1, requires_grad=True)
        c = torch.rand(1, requires_grad=True)
        d = torch.rand(1, requires_grad=True)
        self.checkScript(func, (a, b), optimize=True)
        self.checkScript(func2, (a, b, c, d), optimize=True)
        self.checkScript(func3, (4, -0.5), optimize=True)
        self.checkScript(func4, ())

        inputs = [
            torch.tensor(2),
            torch.tensor(-2),
            torch.tensor(0.5),
            torch.tensor(0.2),
        ]
        for x in inputs:
            for y in inputs:
                if x < 0:
                    continue
                else:
                    self.checkScript(func5, (x, y))

    def test_triple(self):
        # [FUNCTION DOC] Checks scripting of multiplication of a float constant by a
        # grad-requiring tensor.
        def func(x):
            # [FUNCTION DOC] HELPER: Returns the input multiplied by the float constant three.
            return 3.0 * x

        x = torch.rand(1, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_slice(self):
        # [FUNCTION DOC] Checks scripting of four slice forms on a 10-element tensor: leading,
        # trailing, stepped and start-with-step slices.
        def func(x):
            # [FUNCTION DOC] HELPER: Returns the first five elements.
            return x[:5]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

        def func2(x):
            # [FUNCTION DOC] HELPER: Returns everything from index five on.
            return x[5:]

        self.checkScript(func2, [x], optimize=True)

        def func3(x):
            # [FUNCTION DOC] HELPER: Returns elements before index eight with step two.
            return x[:8:2]

        self.checkScript(func3, [x], optimize=True)

        def func4(x):
            # [FUNCTION DOC] HELPER: Returns every fourth element starting at index one.
            return x[1::4]

        self.checkScript(func4, [x], optimize=True)

    def test_gather(self):
        # [FUNCTION DOC] Checks scripting of integer indexing into a tensor.
        def func(x):
            # [FUNCTION DOC] HELPER: Returns the first element along the leading dimension.
            return x[0]

        x = torch.rand(10, dtype=torch.float, requires_grad=True)
        self.checkScript(func, [x], optimize=True)

    def test_random(self):
        # [FUNCTION DOC] Checks that a scripted call to the normal sampler gives the same output
        # as the eager call when the random generator state is forked and restored before each
        # call (fork with no devices, so only the CPU generator state is saved).
        @torch.jit.script
        def f(mean, std):
            # [FUNCTION DOC] HELPER: Scripted function returning a tensor sampled from a normal
            # distribution with the given mean and standard deviation tensors.
            return torch.normal(mean, std)

        mean, std = torch.zeros(5, 5), torch.ones(5, 5)
        with torch.random.fork_rng(devices=[]):
            output = torch.normal(mean, std)
        with torch.random.fork_rng(devices=[]):
            script_output = f(mean, std)
        self.assertEqual(output, script_output)

    def _check_code(self, code_str, fn_name, inputs):
        # [FUNCTION DOC] HELPER: Executes the source text in plain Python (using module globals
        # plus a local scope), compiles the same text in a CompilationUnit, and asserts both
        # versions agree when called on the given inputs. The scope entry named by the second
        # argument is called in eager mode, but the compiled side is always looked up under the
        # name func, so the source must define that name. Side effect: runs arbitrary source text.
        scope = {}
        exec(code_str, globals(), scope)
        cu = torch.jit.CompilationUnit(code_str)
        self.assertEqual(cu.func(*inputs), scope[fn_name](*inputs))

    def test_stepped_tuple_slicing(self):
        # [FUNCTION DOC] Compares eager and scripted stepped slicing of tuples, including negative
        # steps, steps larger than the tuple, empty results and out-of-range bounds, over tuples
        # of 3 to 7 ints.
        def check_slicing_tuple(slicing, tuple_type, tuple):
            # [FUNCTION DOC] HELPER: Fills a source template defining a function that applies the
            # given slicing text to a tuple parameter annotated with the given tuple type string,
            # then runs the eager-versus-script comparison on the given tuple value.
            template = dedent(
                """
            def func(x):
                # type: ({}) -> Any
                return x{}
            """
            )
            self._check_code(template.format(tuple_type, slicing), "func", [tuple])

        check_slicing_tuple("[-3:3:2]", "Tuple[int, int, int]", (0, 1, 2))
        check_slicing_tuple("[::55]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple("[:4:4]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple(
            "[::-1]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6)
        )
        check_slicing_tuple(
            "[7:5:2]", "Tuple[int, int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5, 6)
        )
        check_slicing_tuple(
            "[5:7:-2]",
            "Tuple[int, int, int, int, int, int, int]",
            (0, 1, 2, 3, 4, 5, 6),
        )
        check_slicing_tuple("[::-2]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4))
        check_slicing_tuple(
            "[:4:-3]", "Tuple[int, int, int, int, int, int]", (0, 1, 2, 3, 4, 5)
        )
        check_slicing_tuple(
            "[3::-2]", "Tuple[int, int, int, int, int]", (0, 1, 2, 3, 4)
        )

    def test_index(self):
        # [FUNCTION DOC] Compares eager and scripted indexing of tensors using many index forms
        # built from source strings: integer, slice, negative, multi-dimensional, mixed,
        # zero-sized, expression-valued, None for new dimensions, and runtime-computed indices.
        def consec(size, start=0):
            # [FUNCTION DOC] HELPER: Builds an arange tensor reshaped to the given size.
            # KNOWN ISSUE: the start parameter is accepted but never used, so tensors always start
            # at zero.
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def check_indexing(indexing, tensor):
            # [FUNCTION DOC] HELPER: Formats a template function that indexes its tensor argument
            # with the given index text, then compares eager and scripted results on that tensor.
            template = dedent(
                """
            def func(x):
                return x{}
            """
            )

            self._check_code(template.format(indexing), "func", [tensor])

        def check_dynamic_indexing(indexing, tensor, value1, value2):
            # [FUNCTION DOC] HELPER: Like the static variant but the template converts two extra
            # tensor arguments to ints named i and j first, so the index text may use them. Both
            # values are wrapped as tensors before the comparison.
            value1 = torch.tensor(value1)
            value2 = torch.tensor(value2)

            template = dedent(
                """
            def func(x, value1, value2):
                i = int(value1)
                j = int(value2)
                return x{}
            """
            )

            self._check_code(
                template.format(indexing), "func", [tensor, value1, value2]
            )

        # basic slices
        check_indexing("[0]", consec((3, 3)))
        check_indexing("[1]", consec((3, 3), 10))
        check_indexing("[2]", consec((3, 3), 19))
        check_indexing("[2]", consec((3,)))
        check_indexing("[-1]", consec((3, 3), 19))
        check_indexing("[0:2]", consec((3, 3, 3)))
        check_indexing("[1:-1]", consec((3, 3, 3)))
        check_indexing("[-3:-1]", consec((6, 3)))
        check_indexing("[1:]", consec((3, 3)))
        check_indexing("[:1]", consec((3, 3)))
        check_indexing("[:]", consec((3, 2)))

        # multi-dim: indexes
        check_indexing("[0, 1]", consec((3, 3)))
        check_indexing("[0, 1]", consec((3, 3, 2)))
        check_indexing("[1, 0, 2]", consec((3, 3, 3)))
        check_indexing("[2, -1]", consec((3, 3)))

        # multi-dim: mixed slicing and indexing
        check_indexing("[0, 1:2]", consec((3, 3)))
        check_indexing("[0, :1]", consec((3, 3, 2)))
        check_indexing("[1, 2:]", consec((3, 3, 3)))
        check_indexing("[-1, 1:, 0]", consec((3, 3, 3, 3)))
        check_indexing("[1:, -1, 0]", consec((3, 3, 3, 3)))
        check_indexing("[-1, 2:, 1:2]", consec((3, 3, 3, 3)))
        check_indexing("[-1, 1:, 0]", consec((3, 3, 3, 3)))
        check_indexing("[-1, :, 0, 2]", consec((3, 3, 3, 3)))

        # zero-sized slices
        check_indexing("[0:0]", consec((2, 2)))
        check_indexing("[0:0, 1]", consec((3, 3)))

        # trivial expression usage
        check_indexing("[1+1]", consec((3, 3)))
        check_indexing("[1:(0 + 2)]", consec((3, 3, 3)))

        # None for new dimensions
        check_indexing("[None, 0]", consec((3, 3)))
        check_indexing("[1, None]", consec((3, 3), 10))
        check_indexing("[None, None, 2]", consec((3, 3), 19))
        check_indexing("[None, 2, None]", consec((3,)))
        check_indexing("[0:2, None]", consec((3, 3, 3)))
        check_indexing("[None, 1:-1]", consec((3, 3, 3)))
        check_indexing("[None, -3:-1, None]", consec((6, 3)))
        check_indexing("[-1, None, 2:, None, 1:2]", consec((3, 3, 3, 3)))
        check_indexing("[None, -1, None, 2:, None, 1:2, None]", consec((3, 3, 3, 3)))

        # dynamic expression usage
        check_dynamic_indexing("[i + j]", consec((3, 3)), 0, 1)
        check_dynamic_indexing("[i:j, i]", consec((3, 3, 2)), 0, 2)

    def test_advancedindex(self):
        # [FUNCTION DOC] Compares eager and scripted advanced indexing with integer index tensors,
        # first for basic single-tensor cases and then for a table of index expressions mixing
        # tensors, ints and slices, each tensor created from nested lists. Lists are used because
        # sequence indexing is not supported directly in these scripts.
        def consec(size, start=0):
            # [FUNCTION DOC] HELPER: Builds an arange tensor reshaped to the given size.
            # KNOWN ISSUE: the start parameter is accepted but never used, so tensors always start
            # at zero.
            numel = torch.tensor(size).prod().item()
            return torch.arange(numel).view(size)

        def check_indexing(indexing, tensor, **kwargs):
            # [FUNCTION DOC] HELPER: Builds a template function taking the tensor plus one extra
            # parameter per keyword argument, indexes the tensor with the given index text, and
            # compares eager and scripted results using the keyword values as extra
            # arguments in order.
            indices_dict = kwargs

            template = dedent(
                """
            def func(x{formals}):
                return x{expr}
            """
            )

            formals = []
            values = []
            for formal, value in indices_dict.items():
                formals.append(formal)
                values.append(value)

            formals = "".join(map(", {}".format, formals))
            inputs = [tensor] + values
            self._check_code(
                template.format(formals=formals, expr=indexing), "func", inputs
            )

        # Indexing with tensor (basic)
        check_indexing("[i]", consec((3, 3)), i=torch.tensor([0]))
        check_indexing("[i]", consec((3, 3)), i=torch.tensor(1))
        check_indexing("[i]", consec((3, 3)), i=torch.tensor([-2]))
        check_indexing("[i]", consec((3, 3), 2), i=torch.tensor([0, 0]))
        check_indexing("[i]", consec((3, 3, 2, 2)), i=torch.tensor([0, -2, 1]))

        # NB: indexing with tensors and indexing with sequences can be implemented
        # in a very similar way (sequences are converted to tensors), so only one
        # case needs to be tested extensively.
        # XXX: When we can index with sequences, replace these cases with
        # sequence indexing expressions; those are much easier to read.

        # Misc sequence advanced indexing
        inp = consec((4, 8, 5))
        to_check = [
            # [[0, 1, 3]]
            ["[i]", {"i": [0, 1, 3]}],
            # [[0, 2], [1, 3]]
            ["[i, j]", {"i": [0, 2], "j": [1, 3]}],
            # [[[0, 1], [0, 1]], [[0, 1], [0, 1]]]
            ["[i, j]", {"i": [[0, 1], [0, 1]], "j": [[0, 1], [0, 1]]}],
            # [[0, 2], [1, 3], [1, 1]]
            ["[i, j, k]", {"i": [0, 2], "j": [1, 3], "k": [1, 1]}],
            # [[0, 2], 1, [1, 1]]
            ["[i, j, k]", {"i": [0, 2], "j": 1, "k": [1, 1]}],
            # [:, :, [0, 3, 4]]
            ["[:, :, i]", {"i": [0, 3, 4]}],
            # [:, [2, 4, 5, 7], 2:4]
            ["[:, i, 2:4]", {"i": [0, 2, 3]}],
            # [[2, 3], :, :]
            ["[i, :, :]", {"i": [2, 3]}],
            # [:, [0, 2, 3], [1, 3, 4]]
            ["[:, i, j]", {"i": [0, 2, 3], "j": [1, 3, 4]}],
            # [:, [0], [1, 2, 4]]
            ["[:, i, j]", {"i": [0], "j": [1, 2, 4]}],
            # [:, [0, 1, 3], [4]]
            ["[:, i, j]", {"i": [0, 1, 3], "j": [4]}],
            # [:, [[0, 1], [1, 0]], [[2, 3]]]
            ["[:, i, j]", {"i": [[0, 1], [1, 0]], "j": [[2, 3]]}],
            # [:, [[0, 1], [2, 3]], [[0]]]
            ["[:, i, j]", {"i": [[0, 1], [2, 3]], "j": [[0]]}],
            # [:, [[5, 6]], [[0, 3], [4, 4]]]
            ["[:, i, j]", {"i": [[5, 6]], "j": [[0, 3], [4, 4]]}],
            # [[0, 2, 3], [1, 3, 4], :]
            ["[i, j, :]", {"i": [0, 2, 3], "j": [1, 3, 4]}],
            # [0, [1, 2, 4], :]
            ["[i, j, :]", {"i": 0, "j": [1, 2, 4]}],
            # [[0, 1, 3], 4, :]
            ["[i, j, :]", {"i": [0, 1, 3], "j": 4}],
            # [[[0, 1], [1, 0]], [[2, 1], [3, 5]], :]
            ["[i, j, :]", {"i": [[0, 1], [1, 0]], "j": [[2, 1], [3, 5]]}],
            # [[[0, 1], [1, 0]], [[2, 3]], :]
            ["[i, j, :]", {"i": [[0, 1], [1, 0]], "j": [[2, 3]]}],
            # [[[0, 1], [2, 3]], [[0]], :]
            ["[i, j, :]", {"i": [[0, 1], [2, 3]], "j": [[0]]}],
            # [[[2, 1]], [[0, 3], [4, 4]], :]
            ["[i, j, :]", {"i": [[2, 1]], "j": [[0, 3], [4, 4]]}],
            # [[[2]], [[0, 3], [4, 1]], 0:2]
            ["[i, j, 0:2]", {"i": [[2]], "j": [[0, 3], [4, 1]]}],
        ]

        for expr, argdict in to_check:
            tensordict = {k: torch.tensor(v) for (k, v) in argdict.items()}
            check_indexing(expr, inp, **tensordict)

    def test_adv_indexing_list(self):
        # [FUNCTION DOC] Checks that indexing a tensor with Python lists (literal, nested, built
        # by appending, and floats) scripts the same as eager indexing on a 6x2 random tensor.
        # indexing with list is equivalent to indexing with tensor
        def func1(x):
            # [FUNCTION DOC] HELPER: Indexes with a literal list of three ints.
            return x[[0, 1, 5]]

        def func2(x):
            # [FUNCTION DOC] HELPER: Indexes with two literal lists, one per dimension.
            return x[[0, 1], [0, 1]]

        def func3(x):
            # [FUNCTION DOC] HELPER: Indexes with two literal nested lists.
            return x[[[0, 1], [0, 1]], [[0, 1], [0, 1]]]

        def func4(x):
            # [FUNCTION DOC] HELPER: Builds an int list by appending and indexes with it.
            ls = [0]
            ls.append(1)
            ls.append(2)
            return x[ls]

        def func5(x):
            # [FUNCTION DOC] HELPER: Indexes with a list of float values; this is the only case
            # with a non-integer list.
            ls = [0.1, 1.2, 2.3]
            return x[ls]

        input = torch.rand((6, 2))
        self.checkScript(func1, (input,))
        self.checkScript(func2, (input,))
        self.checkScript(func3, (input,))
        self.checkScript(func4, (input,))
        self.checkScript(func5, (input,))

    def test_index_ellipses(self):
        # [FUNCTION DOC] One hundred random trials: builds a four-entry index list from colon, one
        # and None, replaces one random entry with an ellipsis, compiles a function indexing a 5-d
        # ones tensor with it and returns the result shape, and compares with eager execution.
        vals = [":", 1, None]
        for _ in range(100):
            indices = [random.choice(vals) for _ in range(4)]
            indices[random.randint(0, len(indices) - 1)] = "..."
            test_str = dedent(
                """
            def f():
                x = torch.ones(10, 9, 8, 7, 6)
                return x{indices}.shape
            """.format(indices=indices)
            )
            test_str = test_str.replace(r"'", r"")
            scope = {}
            execWrapper(test_str, globals(), scope)
            cu = torch.jit.CompilationUnit(test_str)
            res1 = cu.f()
            res2 = scope["f"]()
            self.assertEqual(res1, res2)

    def test_inf(self):
        # [FUNCTION DOC] Checks that comparing a scalar tensor with positive and negative infinity
        # compiles and holds, and that a function from imported source reassigning a variable
        # inside a branch compiles and returns the right values for true and false inputs.
        # KNOWN ISSUE: the function bar is defined but never called; the check after it calls foo
        # a second time.
        @torch.jit.script
        def foo(a):
            # [FUNCTION DOC] HELPER: Scripted function returning whether the input is below
            # positive infinity.
            return a < float("inf")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def bar(a):
            # [FUNCTION DOC] HELPER: Scripted function returning whether the input is above
            # negative infinity; never called.
            return a > float("-inf")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        # test re-assignment on imported source
        str = """
        def foo(x):
            # type: (bool)
            a = float("-inf")
            if not x:
                a = float(torch.tensor([5]))
            return a < 4
        """
        cu = torch.jit.CompilationUnit(str)
        self.assertTrue(cu.foo(True))
        self.assertFalse(cu.foo(False))

    def test_str_to_float(self):
        # [FUNCTION DOC] Checks that converting a malformed string to float inside a script raises
        # RuntimeError mentioning a failed conversion, and that valid numeric strings convert fine.
        @torch.jit.script
        def foo(a):
            # [FUNCTION DOC] HELPER: Scripted function that compares a float constant with the
            # conversion of a malformed numeric string; the conversion fails when run.
            return 0.5 == float("0.5 hello")

        s = torch.rand(1)
        with self.assertRaisesRegex(RuntimeError, "could not convert string to float"):
            self.assertTrue(foo(s))

        @torch.jit.script
        def foo(a):
            # [FUNCTION DOC] HELPER: Scripted function comparing a float constant with the
            # conversion of a valid decimal string.
            return 0.5 == float("0.5")

        s = torch.rand(1)
        self.assertTrue(foo(s))

        @torch.jit.script
        def foo(a):
            # [FUNCTION DOC] HELPER: Scripted function comparing zero with the conversion of the
            # string for zero.
            return 0.0 == float("0")

        s = torch.rand(1)
        self.assertTrue(foo(s))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
