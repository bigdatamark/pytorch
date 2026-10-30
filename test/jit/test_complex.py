# Owner(s): ["oncall: jit"]

# [MODULE DOC] Tests for complex number support in TorchScript: scalar, list and dict
# arguments, pickling, the cmath functions and constants, the complex constructor, comparison
# and arithmetic operators, and mixed complex/tensor operations. Single test class
# TestComplex; most checks compare scripted against eager results through checkScript. No
# import-time side effects except extending sys.path. Running the file directly is rejected.
import cmath
import os
import sys
from itertools import product
from textwrap import dedent
from typing import Dict, List

import torch
from torch.testing._internal.common_utils import IS_MACOS, raise_on_run_directly
from torch.testing._internal.jit_utils import execWrapper, JitTestCase


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestComplex(JitTestCase):
    # [CLASS DOC] JitTestCase for complex scalars in TorchScript. Each test defines small
    # functions with complex type annotations and verifies eager and scripted behavior agree.
    def test_script(self):
        # [FUNCTION DOC] Check that a scripted function taking and returning one complex
        # scalar matches eager execution.
        def fn(a: complex):
            # [FUNCTION DOC] HELPER: Return the complex argument unchanged.
            return a

        self.checkScript(fn, (3 + 5j,))

    def test_complexlist(self):
        # [FUNCTION DOC] Check scripting a function that indexes into a list of complex
        # numbers (mixed with ints and plain complex values) at a given position.
        def fn(a: List[complex], idx: int):
            # [FUNCTION DOC] HELPER: Return the list element at the given index.
            return a[idx]

        input = [1j, 2, 3 + 4j, -5, -7j]
        self.checkScript(fn, (input, 2))

    def test_complexdict(self):
        # [FUNCTION DOC] Check scripting a function that looks up a value in a dictionary with
        # complex keys and complex values.
        def fn(a: Dict[complex, complex], key: complex) -> complex:
            # [FUNCTION DOC] HELPER: Return the dictionary value stored under the given
            # complex key.
            return a[key]

        input = {2 + 3j: 2 - 3j, -4.3 - 2j: 3j}
        self.checkScript(fn, (input, -4.3 - 2j))

    def test_pickle(self):
        # [FUNCTION DOC] Export and re-import a scripted module holding complex attributes
        # (scalar, list and dictionary) and check each attribute round trips equal, and that
        # the loaded forward adds a complex literal to an integer argument.
        class ComplexModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module with complex scalar, list and dictionary
            # attributes and a forward that adds an imaginary constant to its integer input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module and set the complex attributes.
                super().__init__()
                self.a = 3 + 5j
                self.b = [2 + 3j, 3 + 4j, 0 - 3j, -4 + 0j]
                self.c = {2 + 3j: 2 - 3j, -4.3 - 2j: 3j}

            @torch.jit.script_method
            def forward(self, b: int):
                # [FUNCTION DOC] HELPER: Return the integer argument plus a
                # purely imaginary constant.
                return b + 2j

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded.a, 3 + 5j)
        self.assertEqual(loaded.b, [2 + 3j, 3 + 4j, -3j, -4])
        self.assertEqual(loaded.c, {2 + 3j: 2 - 3j, -4.3 - 2j: 3j})
        self.assertEqual(loaded(2), 2 + 2j)

    def test_complex_parse(self):
        # [FUNCTION DOC] Check scripting a function with complex literals in arithmetic
        # and a write of a complex literal into a real tensor element, comparing
        # scripted with eager.
        def fn(a: int, b: torch.Tensor, dim: int):
            # [FUNCTION DOC] HELPER: Assign a complex literal into a tensor element, then
            # return a complex arithmetic expression of the integer argument and literals.
            # verifies `emitValueToTensor()` 's behavior
            b[dim] = 2.4 + 0.5j
            return (3 * 2j) + a + 5j - 7.4j - 4

        t1 = torch.tensor(1)
        t2 = torch.tensor([0.4, 1.4j, 2.35])

        self.checkScript(fn, (t1, t2, 2))

    def test_complex_constants_and_ops(self):
        # [FUNCTION DOC] Compare scripted and Python behavior over a grid of real/complex
        # values for the cmath unary functions (including inf/nan/finite predicates with extra
        # special inputs), abs, the three pow combinations, rect (skipped on macOS), and the
        # cmath constants. Where Python raises and the scripted version does not, the mismatch
        # is tolerated.
        # KNOWN ISSUE: the complex pow check pairs each value with itself via zip rather than
        # over all combinations, so only the diagonal is covered.
        vals = (
            [0.0, 1.0, 2.2, -1.0, -0.0, -2.2, 1, 0, 2]
            + [10.0**i for i in range(2)]
            + [-(10.0**i) for i in range(2)]
        )
        complex_vals = tuple(complex(x, y) for x, y in product(vals, vals))

        funcs_template = dedent(
            """
            def func(a: complex):
                return cmath.{func_or_const}(a)
            """
        )

        def checkCmath(func_name, funcs_template=funcs_template):
            # [FUNCTION DOC] HELPER: Compile a one-function source made from the template
            # with the named cmath function or constant, run the Python and scripted
            # versions for all candidate inputs, and assert equal results unless Python
            # raised an exception. Uses extended special values for the inf/nan/finite
            # predicates. Raises an assertion failure with a message naming the function
            # and input on mismatch.
            funcs_str = funcs_template.format(func_or_const=func_name)
            scope = {}
            execWrapper(funcs_str, globals(), scope)
            cu = torch.jit.CompilationUnit(funcs_str)
            f_script = cu.func
            f = scope["func"]

            if func_name in ["isinf", "isnan", "isfinite"]:
                new_vals = vals + ([float("inf"), float("nan"), -1 * float("inf")])
                final_vals = tuple(
                    complex(x, y) for x, y in product(new_vals, new_vals)
                )
            else:
                final_vals = complex_vals

            for a in final_vals:
                res_python = None
                res_script = None
                try:
                    res_python = f(a)
                except Exception as e:
                    res_python = e
                try:
                    res_script = f_script(a)
                except Exception as e:
                    res_script = e

                if res_python != res_script:
                    if isinstance(res_python, Exception):
                        continue

                    msg = f"Failed on {func_name} with input {a}. Python: {res_python}, Script: {res_script}"
                    self.assertEqual(res_python, res_script, msg=msg)

        unary_ops = [
            "log",
            "log10",
            "sqrt",
            "exp",
            "sin",
            "cos",
            "asin",
            "acos",
            "atan",
            "sinh",
            "cosh",
            "tanh",
            "asinh",
            "acosh",
            "atanh",
            "phase",
            "isinf",
            "isnan",
            "isfinite",
        ]

        # --- Unary ops ---
        for op in unary_ops:
            checkCmath(op)

        def fn(x: complex):
            # [FUNCTION DOC] HELPER: Return the absolute value of the complex argument.
            return abs(x)

        for val in complex_vals:
            self.checkScript(fn, (val,))

        def pow_complex_float(x: complex, y: float):
            # [FUNCTION DOC] HELPER: Raise the complex base to a float exponent.
            return pow(x, y)

        def pow_float_complex(x: float, y: complex):
            # [FUNCTION DOC] HELPER: Raise the float base to a complex exponent.
            return pow(x, y)

        self.checkScript(pow_float_complex, (2, 3j))
        self.checkScript(pow_complex_float, (3j, 2))

        def pow_complex_complex(x: complex, y: complex):
            # [FUNCTION DOC] HELPER: Raise the complex base to a complex exponent.
            return pow(x, y)

        for x, y in zip(complex_vals, complex_vals):
            # Reference: https://github.com/pytorch/pytorch/issues/54622
            if x == 0:
                continue
            self.checkScript(pow_complex_complex, (x, y))

        if not IS_MACOS:
            # --- Binary op ---
            def rect_fn(x: float, y: float):
                # [FUNCTION DOC] HELPER: Build a complex number from magnitude and phase via
                # the cmath rectangular conversion.
                return cmath.rect(x, y)

            for x, y in product(vals, vals):
                self.checkScript(
                    rect_fn,
                    (
                        x,
                        y,
                    ),
                )

        func_constants_template = dedent(
            """
            def func():
                return cmath.{func_or_const}
            """
        )
        float_consts = ["pi", "e", "tau", "inf", "nan"]
        complex_consts = ["infj", "nanj"]
        for x in float_consts + complex_consts:
            checkCmath(x, funcs_template=func_constants_template)

    def test_infj_nanj_pickle(self):
        # [FUNCTION DOC] Export and re-import a scripted module whose forward returns sums
        # involving the cmath infinite and nan imaginary constants, then check both branches
        # of the loaded module equal the expected values.
        class ComplexModule(torch.jit.ScriptModule):
            # [CLASS DOC] HELPER: Script module whose forward branches on an integer and
            # adds either the infinite imaginary constant or the nan imaginary constant
            # to an integer.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Initialize the module and store
                # one complex attribute.
                super().__init__()
                self.a = 3 + 5j

            @torch.jit.script_method
            def forward(self, infj: int, nanj: int):
                # [FUNCTION DOC] HELPER: Return the first integer plus the infinite imaginary
                # constant when it equals two, otherwise the second integer plus the nan
                # imaginary constant.
                if infj == 2:
                    return infj + cmath.infj
                else:
                    return nanj + cmath.nanj

        loaded = self.getExportImportCopy(ComplexModule())
        self.assertEqual(loaded(2, 3), 2 + cmath.infj)
        self.assertEqual(loaded(3, 4), 4 + cmath.nanj)

    def test_complex_constructor(self):
        # [FUNCTION DOC] Check scripted complex construction from every pair of int, float and
        # bool argument types, with several value pairs each, against eager results.
        # Test all scalar types
        def fn_int(real: int, img: int):
            # [FUNCTION DOC] HELPER: Construct a complex number from two integers.
            return complex(real, img)

        self.checkScript(
            fn_int,
            (
                0,
                0,
            ),
        )
        self.checkScript(
            fn_int,
            (
                -1234,
                0,
            ),
        )
        self.checkScript(
            fn_int,
            (
                0,
                -1256,
            ),
        )
        self.checkScript(
            fn_int,
            (
                -167,
                -1256,
            ),
        )

        def fn_float(real: float, img: float):
            # [FUNCTION DOC] HELPER: Construct a complex number from two floats.
            return complex(real, img)

        self.checkScript(
            fn_float,
            (
                0.0,
                0.0,
            ),
        )
        self.checkScript(
            fn_float,
            (
                -1234.78,
                0,
            ),
        )
        self.checkScript(
            fn_float,
            (
                0,
                56.18,
            ),
        )
        self.checkScript(
            fn_float,
            (
                -1.9,
                -19.8,
            ),
        )

        def fn_bool(real: bool, img: bool):
            # [FUNCTION DOC] HELPER: Construct a complex number from two booleans.
            return complex(real, img)

        self.checkScript(
            fn_bool,
            (
                True,
                True,
            ),
        )
        self.checkScript(
            fn_bool,
            (
                False,
                False,
            ),
        )
        self.checkScript(
            fn_bool,
            (
                False,
                True,
            ),
        )
        self.checkScript(
            fn_bool,
            (
                True,
                False,
            ),
        )

        def fn_bool_int(real: bool, img: int):
            # [FUNCTION DOC] HELPER: Construct a complex number from a boolean and an integer.
            return complex(real, img)

        self.checkScript(
            fn_bool_int,
            (
                True,
                0,
            ),
        )
        self.checkScript(
            fn_bool_int,
            (
                False,
                0,
            ),
        )
        self.checkScript(
            fn_bool_int,
            (
                False,
                -1,
            ),
        )
        self.checkScript(
            fn_bool_int,
            (
                True,
                3,
            ),
        )

        def fn_int_bool(real: int, img: bool):
            # [FUNCTION DOC] HELPER: Construct a complex number from an integer and a boolean.
            return complex(real, img)

        self.checkScript(
            fn_int_bool,
            (
                0,
                True,
            ),
        )
        self.checkScript(
            fn_int_bool,
            (
                0,
                False,
            ),
        )
        self.checkScript(
            fn_int_bool,
            (
                -3,
                True,
            ),
        )
        self.checkScript(
            fn_int_bool,
            (
                6,
                False,
            ),
        )

        def fn_bool_float(real: bool, img: float):
            # [FUNCTION DOC] HELPER: Construct a complex number from a boolean and a float.
            return complex(real, img)

        self.checkScript(
            fn_bool_float,
            (
                True,
                0.0,
            ),
        )
        self.checkScript(
            fn_bool_float,
            (
                False,
                0.0,
            ),
        )
        self.checkScript(
            fn_bool_float,
            (
                False,
                -1.0,
            ),
        )
        self.checkScript(
            fn_bool_float,
            (
                True,
                3.0,
            ),
        )

        def fn_float_bool(real: float, img: bool):
            # [FUNCTION DOC] HELPER: Construct a complex number from a float and a boolean.
            return complex(real, img)

        self.checkScript(
            fn_float_bool,
            (
                0.0,
                True,
            ),
        )
        self.checkScript(
            fn_float_bool,
            (
                0.0,
                False,
            ),
        )
        self.checkScript(
            fn_float_bool,
            (
                -3.0,
                True,
            ),
        )
        self.checkScript(
            fn_float_bool,
            (
                6.0,
                False,
            ),
        )

        def fn_float_int(real: float, img: int):
            # [FUNCTION DOC] HELPER: Construct a complex number from a float and an integer.
            return complex(real, img)

        self.checkScript(
            fn_float_int,
            (
                0.0,
                1,
            ),
        )
        self.checkScript(
            fn_float_int,
            (
                0.0,
                -1,
            ),
        )
        self.checkScript(
            fn_float_int,
            (
                1.8,
                -3,
            ),
        )
        self.checkScript(
            fn_float_int,
            (
                2.7,
                8,
            ),
        )

        def fn_int_float(real: int, img: float):
            # [FUNCTION DOC] HELPER: Construct a complex number from an integer and a float.
            return complex(real, img)

        self.checkScript(
            fn_int_float,
            (
                1,
                0.0,
            ),
        )
        self.checkScript(
            fn_int_float,
            (
                -1,
                1.7,
            ),
        )
        self.checkScript(
            fn_int_float,
            (
                -3,
                0.0,
            ),
        )
        self.checkScript(
            fn_int_float,
            (
                2,
                -8.9,
            ),
        )

    def test_torch_complex_constructor_with_tensor(self):
        # [FUNCTION DOC] Check scripted complex construction when one or both parts
        # are one- element tensors (float, int and bool tensors) mixed with float,
        # int and bool scalars.
        tensors = [torch.rand(1), torch.randint(-5, 5, (1,)), torch.tensor([False])]

        def fn_tensor_float(real, img: float):
            # [FUNCTION DOC] HELPER: Construct a complex number from a tensor and a float.
            return complex(real, img)

        def fn_tensor_int(real, img: int):
            # [FUNCTION DOC] HELPER: Construct a complex number from a tensor and an integer.
            return complex(real, img)

        def fn_tensor_bool(real, img: bool):
            # [FUNCTION DOC] HELPER: Construct a complex number from a tensor and a boolean.
            return complex(real, img)

        def fn_float_tensor(real: float, img):
            # [FUNCTION DOC] HELPER: Construct a complex number from a float and a tensor.
            return complex(real, img)

        def fn_int_tensor(real: int, img):
            # [FUNCTION DOC] HELPER: Construct a complex number from an integer and a tensor.
            return complex(real, img)

        def fn_bool_tensor(real: bool, img):
            # [FUNCTION DOC] HELPER: Construct a complex number from a boolean and a tensor.
            return complex(real, img)

        for tensor in tensors:
            self.checkScript(fn_tensor_float, (tensor, 1.2))
            self.checkScript(fn_tensor_int, (tensor, 3))
            self.checkScript(fn_tensor_bool, (tensor, True))

            self.checkScript(fn_float_tensor, (1.2, tensor))
            self.checkScript(fn_int_tensor, (3, tensor))
            self.checkScript(fn_bool_tensor, (True, tensor))

        def fn_tensor_tensor(real, img):
            # [FUNCTION DOC] HELPER: Construct a complex number from two tensors and add a
            # complex number built from the single real part two.
            return complex(real, img) + complex(2)

        for x, y in product(tensors, tensors):
            self.checkScript(
                fn_tensor_tensor,
                (
                    x,
                    y,
                ),
            )

    def test_comparison_ops(self):
        # [FUNCTION DOC] Check scripted equality and inequality between two complex values and
        # between a complex value and a float.
        def fn1(a: complex, b: complex):
            # [FUNCTION DOC] HELPER: Return whether two complex values are equal.
            return a == b

        def fn2(a: complex, b: complex):
            # [FUNCTION DOC] HELPER: Return whether two complex values differ.
            return a != b

        def fn3(a: complex, b: float):
            # [FUNCTION DOC] HELPER: Return whether a complex value equals a float.
            return a == b

        def fn4(a: complex, b: float):
            # [FUNCTION DOC] HELPER: Return whether a complex value differs from a float.
            return a != b

        x, y = 2 - 3j, 4j
        self.checkScript(fn1, (x, x))
        self.checkScript(fn1, (x, y))
        self.checkScript(fn2, (x, x))
        self.checkScript(fn2, (x, y))

        x1, y1 = 1 + 0j, 1.0
        self.checkScript(fn3, (x1, y1))
        self.checkScript(fn4, (x1, y1))

    def test_div(self):
        # [FUNCTION DOC] Check scripted division of one complex number by another.
        def fn1(a: complex, b: complex):
            # [FUNCTION DOC] HELPER: Return the quotient of two complex values.
            return a / b

        x, y = 2 - 3j, 4j
        self.checkScript(fn1, (x, y))

    def test_complex_list_sum(self):
        # [FUNCTION DOC] Check scripted sum over a list of complex doubles taken from a random
        # complex tensor.
        def fn(x: List[complex]):
            # [FUNCTION DOC] HELPER: Return the sum of the list elements.
            return sum(x)

        self.checkScript(fn, (torch.randn(4, dtype=torch.cdouble).tolist(),))

    def test_tensor_attributes(self):
        # [FUNCTION DOC] Check the real and imaginary part attributes of a complex tensor in
        # scripted functions.
        def tensor_real(x):
            # [FUNCTION DOC] HELPER: Return the real part of the tensor.
            return x.real

        def tensor_imag(x):
            # [FUNCTION DOC] HELPER: Return the imaginary part of the tensor.
            return x.imag

        t = torch.randn(2, 3, dtype=torch.cdouble)
        self.checkScript(tensor_real, (t,))
        self.checkScript(tensor_imag, (t,))

    def test_binary_op_complex_tensor(self):
        # [FUNCTION DOC] Script the binary operations multiply, add, equal, not equal,
        # subtract and divide between a complex scalar and a complex tensor of two shapes, and
        # compare each result with eager execution.
        # KNOWN ISSUE: the helper named for division actually subtracts, so division
        # is not tested.
        def mul(x: complex, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return the product of the scalar and the tensor.
            return x * y

        def add(x: complex, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return the sum of the scalar and the tensor.
            return x + y

        def eq(x: complex, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return the elementwise equality of the scalar
            # and the tensor.
            return x == y

        def ne(x: complex, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return the elementwise inequality of the scalar
            # and the tensor.
            return x != y

        def sub(x: complex, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return the scalar minus the tensor.
            return x - y

        def div(x: complex, y: torch.Tensor):
            # [FUNCTION DOC] HELPER: Return the scalar minus the tensor.
            # KNOWN ISSUE: despite the name, this subtracts instead of dividing.
            return x - y

        ops = [mul, add, eq, ne, sub, div]

        for shape in [(1,), (2, 2)]:
            x = 0.71 + 0.71j
            y = torch.randn(shape, dtype=torch.cfloat)
            for op in ops:
                eager_result = op(x, y)
                scripted = torch.jit.script(op)
                jit_result = scripted(x, y)
                self.assertEqual(eager_result, jit_result)


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
