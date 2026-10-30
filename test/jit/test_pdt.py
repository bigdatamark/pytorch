# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for profile directed typing in TorchScript: scripting modules, classes and
# functions whose argument types are inferred by running them on example inputs (recorded through
# the monkeytype package) instead of annotations. Import-time side effects: appends the parent
# test directory to the module search path; if monkeytype is not installed it prints a notice to
# standard error and replaces the test base class with a no-op placeholder so no tests run.
# Running the file directly raises an error pointing at test_jit.

import os
import sys
from typing import Any, Dict, List, NamedTuple, Optional, Tuple  # noqa: F401

import torch
from torch.jit._monkeytype_config import _IS_MONKEYTYPE_INSTALLED
from torch.testing._internal.common_utils import NoTest, raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, make_global


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)

if not _IS_MONKEYTYPE_INSTALLED:
    print(
        "monkeytype is not installed. Skipping tests for Profile-Directed Typing",
        file=sys.stderr,
    )
    JitTestCase = NoTest  # type: ignore[misc, assignment]


class TestPDT(JitTestCase):
    # [CLASS DOC] Suite of profile directed typing tests. Each test defines local modules, classes
    # or functions, registers them as globally visible with make_global, scripts them with example
    # inputs, and compares scripted results to eager results. Becomes an empty placeholder when
    # monkeytype is missing.
    """
    A suite of tests for profile directed typing in TorchScript.
    """

    def test_nn_module(self):
        # [FUNCTION DOC] Scripts a module whose forward branches on int and float inputs, using
        # int, float and bool examples, and checks scripted and eager results agree for int and
        # float inputs.
        # KNOWN ISSUE: the bool check uses a truth assertion with the eager result as its message,
        # so it only checks the scripted result is truthy.
        class TestPDTModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward has no annotated argument type and returns
            # a value derived from the type of its input.
            def forward(self, x) -> Any:
                # [FUNCTION DOC] HELPER: Returns an int plus one, a float minus one, and any other
                # value unchanged.
                if isinstance(x, int):
                    return x + 1
                elif isinstance(x, float):
                    return x - 1
                else:
                    return x

        make_global(TestPDTModel)
        pdt_model = TestPDTModel()
        inp: List[Tuple[Any, ...]] = [
            (20,),
            (2.7,),
            (False,),
        ]
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model: inp}
        )
        self.assertEqual(scripted_pdt_model(50), pdt_model(50))
        self.assertEqual(scripted_pdt_model(1.8), pdt_model(1.8))
        self.assertTrue(scripted_pdt_model(True), pdt_model(True))

    def test_nested_nn_module_class(self):
        # [FUNCTION DOC] Scripts a wrapper module holding an inner module, giving examples only
        # for the wrapper, and checks results match eager for int and float inputs.
        # KNOWN ISSUE: the bool check only asserts the scripted result is truthy; the eager result
        # is just its failure message.
        class NestedPDTInner(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module whose forward multiplies an int by ten and returns
            # anything else unchanged.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns ten times an int input; else returns the input.
                if isinstance(x, int):
                    return x * 10
                return x

        class NestedModulePDTWrapper(torch.nn.Module):
            # [CLASS DOC] HELPER: Wrapper module that stores an inner module and forwards its
            # input to it.
            def __init__(self, inner):
                # [FUNCTION DOC] HELPER: Stores the given module as the attribute named inner.
                super().__init__()
                self.inner = inner

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the inner module on the input, returns its result.
                return self.inner(x)

        make_global(NestedPDTInner, NestedModulePDTWrapper)
        inner_pdt_model = NestedPDTInner()
        wrapped_pdt_model = NestedModulePDTWrapper(inner_pdt_model)
        inp: List[Tuple[Any, ...]] = [(20,), (False,)]
        scripted_pdt_model = torch.jit.script(
            wrapped_pdt_model, example_inputs={wrapped_pdt_model: inp}
        )
        self.assertEqual(scripted_pdt_model(30), wrapped_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), wrapped_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), wrapped_pdt_model(True))

    def test_nested_nn_module_class_with_args(self):
        # [FUNCTION DOC] Scripts an outer module that calls an inner module with an extra constant
        # argument, giving example inputs for both modules, and checks results match eager for int
        # and float inputs.
        # KNOWN ISSUE: the bool check only asserts the scripted result is truthy; the eager result
        # is just its failure message.
        class NestedModulePDTInner(torch.nn.Module):
            # [CLASS DOC] HELPER: Inner module whose forward takes two arguments.
            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Returns ten times the first argument plus the second when
                # the first is an int; otherwise returns the first argument.
                if isinstance(x, int):
                    return x * 10 + y
                return x

        class NestedModulePDTOuter(torch.nn.Module):
            # [CLASS DOC] HELPER: Outer module that stores an inner module and calls it with the
            # input and the constant twenty.
            def __init__(self, inner):
                # [FUNCTION DOC] HELPER: Stores the given module as the attribute named inner.
                super().__init__()
                self.inner = inner

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Calls the inner module with the input and twenty and
                # returns the result.
                return self.inner(x, 20)

        make_global(NestedModulePDTInner, NestedModulePDTOuter)
        inner_pdt_model = NestedModulePDTInner()
        outer_pdt_model = NestedModulePDTOuter(inner_pdt_model)
        inner_input: List[Tuple[Any, ...]] = [
            (10, 10),
            (1.9, 20),
        ]
        outer_input: List[Tuple[Any, ...]] = [(20,), (False,)]
        scripted_pdt_model = torch.jit.script(
            outer_pdt_model,
            example_inputs={
                inner_pdt_model: inner_input,
                outer_pdt_model: outer_input,
            },
        )
        self.assertEqual(scripted_pdt_model(30), outer_pdt_model(30))
        self.assertEqual(scripted_pdt_model(1.9), outer_pdt_model(1.9))
        self.assertTrue(scripted_pdt_model(True), outer_pdt_model(True))

    def test_nested_function_in_forward(self):
        # [FUNCTION DOC] Scripts a module whose forward calls a helper method that branches on
        # bool and int inputs, with int and bool examples, and compares scripted and eager results
        # for an int and a bool.
        class NestedFunctionInForward(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward adds ten to its helper method result.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the helper method result plus ten.
                return self.fun(x) + 10

            def fun(self, x):
                # [FUNCTION DOC] HELPER: Returns zero for bool inputs, the input plus one for
                # other ints, and zero for anything else.
                if isinstance(x, bool):
                    return 0
                elif isinstance(x, int):
                    return x + 1
                return 0

        make_global(NestedFunctionInForward)
        pdt_model = NestedFunctionInForward()
        inp: List[Tuple[Any, ...]] = [(-1,), (False,)]
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model: inp}
        )
        self.assertEqual(scripted_pdt_model(30), pdt_model(30))
        self.assertEqual(scripted_pdt_model(True), pdt_model(True))

    def test_nn_module_with_export_function(self):
        # [FUNCTION DOC] Scripts a module with an exported method taking two arguments, keyed by
        # that bound method in the example inputs (int pair and float pair), and compares results
        # of the exported method with eager for ints, floats and a tensor.
        # KNOWN ISSUE: the tensor check uses a truth assertion with the eager result as its
        # message, so it only checks the scripted result is truthy.
        class TestModelWithExport(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with only an exported method, no forward.
            @torch.jit.export
            def fn(self, x, y) -> Any:
                # [FUNCTION DOC] HELPER: Asserts the arguments are not both bool; returns the sum
                # for two ints, the difference for two floats, and minus one otherwise.
                assert not (isinstance(x, bool) and isinstance(y, bool))  # noqa: S101
                if isinstance(x, int) and isinstance(y, int):
                    return x + y
                elif isinstance(x, float) and isinstance(y, float):
                    return x - y
                else:
                    return -1

        make_global(TestModelWithExport)
        pdt_model = TestModelWithExport()
        inp: List[Tuple[Any, ...]] = [
            (
                20,
                10,
            ),
            (
                2.7,
                8.9,
            ),
        ]
        scripted_pdt_model = torch.jit.script(
            pdt_model, example_inputs={pdt_model.fn: inp}
        )
        self.assertEqual(scripted_pdt_model.fn(10, 90), pdt_model.fn(10, 90))
        self.assertEqual(scripted_pdt_model.fn(1.8, 2.2), pdt_model.fn(1.8, 2.2))
        self.assertTrue(
            scripted_pdt_model.fn(torch.ones(1), 2), pdt_model.fn(torch.ones(1), 2)
        )

    def test_class_methods(self):
        # [FUNCTION DOC] Scripts a plain class (not a module) using examples keyed by a bound
        # method, instantiates the scripted class, and checks the method returns the same sum as
        # eager on a list.
        class PDTModel:
            # [CLASS DOC] HELPER: Plain Python class with a single method that sums a sequence.
            def test_sum(self, a):
                # [FUNCTION DOC] HELPER: Returns the sum of the elements of the argument.
                return sum(a)

        make_global(PDTModel)
        pdt_model = PDTModel()
        inp: List[Tuple[Any, ...]] = [
            (
                [
                    10,
                    20,
                ],
            ),
        ]
        scripted_pdt_model = torch.jit.script(
            PDTModel, example_inputs={pdt_model.test_sum: inp}
        )
        script_model = scripted_pdt_model()
        self.assertEqual(
            script_model.test_sum(
                [
                    10,
                    20,
                    30,
                ],
            ),
            pdt_model.test_sum(
                [
                    10,
                    20,
                    30,
                ],
            ),
        )

    def test_class_with_multiple_methods(self):
        # [FUNCTION DOC] Scripts a class with two methods using examples for each (a float list
        # and a pair of strings), and checks each method matches eager, including substring found
        # and not found cases.
        class PDTModelWithManyMethods:
            # [CLASS DOC] HELPER: Plain class with a method that builds a dictionary from a list
            # and a method that tests substring membership.
            def test_list_to_dict(self, a):
                # [FUNCTION DOC] HELPER: Returns a dictionary mapping each element of the argument
                # to true.
                new_dictionary: Dict[float, bool] = {}
                for element in a:
                    new_dictionary[element] = True
                return new_dictionary

            def test_substring(self, a, b):
                # [FUNCTION DOC] HELPER: Returns whether the second argument is in the first.
                return b in a

        make_global(PDTModelWithManyMethods)
        pdt_model = PDTModelWithManyMethods()
        list_inp: List[Tuple[Any, ...]] = [
            (
                [
                    1.2,
                    2.3,
                ],
            ),
        ]
        str_inp: List[Tuple[Any, ...]] = [
            (
                "abc",
                "b",
            ),
        ]
        scripted_pdt_model = torch.jit.script(
            PDTModelWithManyMethods,
            example_inputs={
                pdt_model.test_list_to_dict: list_inp,
                pdt_model.test_substring: str_inp,
            },
        )
        script_model = scripted_pdt_model()
        self.assertEqual(
            script_model.test_list_to_dict(
                [
                    1.1,
                    2.2,
                    3.3,
                ],
            ),
            pdt_model.test_list_to_dict(
                [
                    1.1,
                    2.2,
                    3.3,
                ],
            ),
        )
        self.assertEqual(
            script_model.test_substring(
                "helloworld",
                "world",
            ),
            pdt_model.test_substring(
                "helloworld",
                "world",
            ),
        )
        self.assertEqual(
            script_model.test_substring(
                "helloworld",
                "def",
            ),
            pdt_model.test_substring(
                "helloworld",
                "def",
            ),
        )

    def test_multiple_class_with_same_method(self):
        # [FUNCTION DOC] Scripts two classes that define a same-named method but are profiled with
        # different input types (dictionary and float key versus list of strings and a string),
        # and checks each scripted class matches eager.
        class PDTModelOne:
            # [CLASS DOC] HELPER: Plain class with one membership-test method.
            def test_find(self, a, b):
                # [FUNCTION DOC] HELPER: Returns whether the second argument is in the first.
                return b in a

        class PDTModelTwo:
            # [CLASS DOC] HELPER: A second plain class with an identical membership-test method.
            def test_find(self, a, b):
                # [FUNCTION DOC] HELPER: Returns whether the second argument is in the first.
                return b in a

        make_global(PDTModelOne, PDTModelTwo)
        pdt_model_one = PDTModelOne()
        pdt_model_two = PDTModelTwo()
        dict_inp: List[Tuple[Any, ...]] = [
            (
                {
                    1.2: True,
                    2.3: False,
                },
                1.2,
            ),
        ]
        list_inp: List[Tuple[Any, ...]] = [
            (
                [
                    "abc",
                    "b",
                ],
                "c",
            ),
        ]
        scripted_pdt_model_one = torch.jit.script(
            PDTModelOne, example_inputs={pdt_model_one.test_find: dict_inp}
        )
        scripted_pdt_model_two = torch.jit.script(
            PDTModelTwo, example_inputs={pdt_model_two.test_find: list_inp}
        )

        script_model_one, script_model_two = (
            scripted_pdt_model_one(),
            scripted_pdt_model_two(),
        )
        self.assertEqual(
            script_model_one.test_find(
                {
                    1.1: True,
                    2.2: True,
                    3.3: False,
                },
                4.4,
            ),
            pdt_model_one.test_find(
                {
                    1.1: True,
                    2.2: True,
                    3.3: False,
                },
                4.4,
            ),
        )
        self.assertEqual(
            script_model_two.test_find(
                [
                    "hello",
                    "world",
                ],
                "world",
            ),
            pdt_model_two.test_find(
                [
                    "hello",
                    "world",
                ],
                "world",
            ),
        )

    def test_pdt(self):
        # [FUNCTION DOC] Scripts several free functions using example inputs for ints, floats,
        # complex construction from two random tensors, bool and string arguments, and compares
        # scripted with eager results for new values.
        def test_sum(a, b):
            # [FUNCTION DOC] HELPER: Returns the sum of the two arguments.
            return a + b

        make_global(test_sum)
        scripted_fn_add = torch.jit.script(test_sum, example_inputs=[(3, 4)])
        self.assertEqual(scripted_fn_add(10, 2), test_sum(10, 2))

        def test_sub(a, b):
            # [FUNCTION DOC] HELPER: Returns the difference of the two arguments.
            return a - b

        make_global(test_sub)
        scripted_fn_sub = torch.jit.script(test_sub, example_inputs=[(3.9, 4.10)])
        self.assertEqual(scripted_fn_sub(6.5, 2.9), test_sub(6.5, 2.9))

        def test_mul(a, b):
            # [FUNCTION DOC] HELPER: Returns the product of the two arguments.
            return a * b

        make_global(test_mul)
        scripted_fn_mul = torch.jit.script(test_mul, example_inputs=[(-10, 9)])
        self.assertEqual(scripted_fn_mul(-1, 3), test_mul(-1, 3))

        def test_args_complex(real, img):
            # [FUNCTION DOC] HELPER: Builds a complex tensor from real and imaginary tensors.
            return torch.complex(real, img)

        make_global(test_args_complex)
        scripted_fn_complex = torch.jit.script(
            test_args_complex, example_inputs=[(torch.rand(3, 4), torch.rand(3, 4))]
        )
        arg1, arg2 = torch.rand(3, 4), torch.rand(3, 4)
        self.assertEqual(scripted_fn_complex(arg1, arg2), test_args_complex(arg1, arg2))

        def test_bool(a):
            # [FUNCTION DOC] HELPER: Returns minus one if the argument is truthy, else zero.
            if a:
                return -1
            else:
                return 0

        make_global(test_bool)
        scripted_fn_bool = torch.jit.script(test_bool, example_inputs=[(True,)])
        self.assertEqual(scripted_fn_bool(True), test_bool(True))

        def test_str(a):
            # [FUNCTION DOC] HELPER: Returns false for the empty string and true otherwise.
            if a == "":
                return False
            else:
                return True

        make_global(test_str)
        scripted_fn_str = torch.jit.script(test_str, example_inputs=[("",)])
        self.assertEqual(scripted_fn_str("abc"), test_str("abc"))

    def test_pdt_list_and_tuple(self):
        # [FUNCTION DOC] Scripts one sum function separately with float, bool and int lists and
        # then with float, bool and int tuples as examples, and checks each scripted version
        # matches eager.
        def test_list_and_tuple(a):
            # [FUNCTION DOC] HELPER: Returns the sum of the elements of the argument.
            return sum(a)

        make_global(test_list_and_tuple)

        scripted_fn_float_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([4.9, 8.9],)]
        )
        self.assertEqual(
            scripted_fn_float_list_input([11.9, 7.6]), test_list_and_tuple([11.9, 7.6])
        )

        scripted_fn_bool_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([True, False, True],)]
        )
        self.assertEqual(
            scripted_fn_bool_list_input([True, True, True]),
            test_list_and_tuple([True, True, True]),
        )

        scripted_fn_int_list_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[([3, 4, 5],)]
        )
        self.assertEqual(
            scripted_fn_int_list_input([1, 2, 3]), test_list_and_tuple([1, 2, 3])
        )

        scripted_fn_float_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((4.9, 8.9),)]
        )
        self.assertEqual(
            scripted_fn_float_tuple_input((11.9, 7.6)), test_list_and_tuple((11.9, 7.6))
        )

        scripted_fn_bool_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((True, False, True),)]
        )
        self.assertEqual(
            scripted_fn_bool_tuple_input((True, True, True)),
            test_list_and_tuple((True, True, True)),
        )

        scripted_fn_int_tuple_input = torch.jit.script(
            test_list_and_tuple, example_inputs=[((3, 4, 5),)]
        )
        self.assertEqual(
            scripted_fn_int_tuple_input((1, 2, 3)), test_list_and_tuple((1, 2, 3))
        )

    def test_nested_list_and_tuple(self):
        # [FUNCTION DOC] Scripts a nested-list function with list and tuple outer containers and a
        # nested-tuple function with float and bool tuples, and compares scripted with eager.
        def test_nested_list(inp):
            # [FUNCTION DOC] HELPER: Returns a list holding the sum of each inner sequence.
            return [sum(v) for v in inp]

        def test_nested_tuple(inp):
            # [FUNCTION DOC] HELPER: Walks nested sequences, multiplying a float accumulator that
            # starts at zero by each positive value.
            # KNOWN ISSUE: the accumulator starts at zero, so the result is always zero and the
            # comparison is weak.
            ans = 0.0
            for tup in inp:
                for val in tup:
                    if val > 0:
                        ans *= val
            return ans

        make_global(test_nested_list, test_nested_tuple)

        list_inp = [
            [
                1,
                2,
                3,
            ],
            [
                5,
                6,
                7,
            ],
        ]
        scripted_fn = torch.jit.script(
            test_nested_list,
            example_inputs=[
                (list_inp,),
            ],
        )
        inp = [
            [
                0,
                4,
                7,
            ],
            [
                8,
                11,
            ],
            [
                6,
                -1,
                -20,
            ],
        ]
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_list(
                inp,
            ),
        )

        list_inp = (
            [
                1,
                2,
                3,
            ],
            [
                5,
                6,
                7,
            ],
        )
        scripted_fn = torch.jit.script(
            test_nested_list,
            example_inputs=[
                (list_inp,),
            ],
        )
        inp = (
            [
                0,
                4,
                7,
            ],
            [
                8,
                11,
            ],
            [
                6,
                -1,
                -20,
            ],
        )
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_list(
                inp,
            ),
        )

        tup_inp = [
            (
                1.0,
                2.6,
                3.7,
            ),
            (
                5.7,
                6.1,
                1.7,
            ),
        ]
        scripted_fn = torch.jit.script(
            test_nested_tuple,
            example_inputs=[
                (tup_inp,),
            ],
        )
        inp = [
            (
                1.0,
                4.1,
                7.4,
            ),
            (
                4.8,
                1.1,
                -1.2,
            ),
            (
                6.3,
                -1.3,
                -2.0,
            ),
        ]
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_tuple(
                inp,
            ),
        )

        tup_inp = (
            (
                True,
                False,
                True,
            ),
            (
                False,
                False,
                False,
            ),
        )
        scripted_fn = torch.jit.script(
            test_nested_tuple,
            example_inputs=[
                (tup_inp,),
            ],
        )
        inp = (
            (
                True,
                True,
                True,
            ),
            (
                False,
                False,
                True,
            ),
        )
        self.assertEqual(
            scripted_fn(
                inp,
            ),
            test_nested_tuple(
                inp,
            ),
        )

    def test_pdt_dict(self):
        # [FUNCTION DOC] Scripts two lookup functions using a string-keyed bool dictionary and an
        # int-keyed dictionary of bool lists as examples, and checks scripted results match eager
        # on new dictionaries.
        def test_dict(a):
            # [FUNCTION DOC] HELPER: Returns the value stored under the string key foo.
            return a["foo"]

        def test_dict_int_list(a):
            # [FUNCTION DOC] HELPER: Returns the value stored under the int key one.
            return a[1]

        make_global(test_dict, test_dict_int_list)

        str_bool_inp = {"foo": True, "bar": False}
        scripted_fn = torch.jit.script(test_dict, example_inputs=[(str_bool_inp,)])
        self.assertEqual(
            scripted_fn(
                {"foo": False, "bar": True},
            ),
            test_dict(
                {"foo": False, "bar": True},
            ),
        )

        str_list_inp = {0: [True, False], 1: [False, True]}
        scripted_fn = torch.jit.script(
            test_dict_int_list, example_inputs=[(str_list_inp,)]
        )
        self.assertEqual(
            scripted_fn(
                {0: [False, False], 1: [True, True]},
            ),
            test_dict_int_list(
                {0: [False, False], 1: [True, True]},
            ),
        )

    def test_any(self):
        # [FUNCTION DOC] Checks scripting when profiled examples show several different types for
        # the same argument (int, string, float, list, and for the second function also bool and
        # dictionary), so the argument is treated as having any type, and results match eager.
        def test_multiple_types(a):
            # [FUNCTION DOC] HELPER: Asserts the argument is not a bool and returns it.
            assert not isinstance(a, bool)  # noqa: S101
            return a

        def test_multiple_type_refinement(a):
            # [FUNCTION DOC] HELPER: Returns one for bool, one plus the value for int, one plus
            # the truncated value for float, and minus one for other types.
            if isinstance(a, bool):
                return 1
            elif isinstance(a, int):
                return 1 + a
            elif isinstance(a, float):
                return 1 + int(a)
            else:
                return -1

        make_global(test_multiple_types, test_multiple_type_refinement)

        scripted_fn = torch.jit.script(
            test_multiple_types, example_inputs=[(1,), ("abc",), (8.9,), ([3, 4, 5],)]
        )
        self.assertEqual(scripted_fn(10), test_multiple_types(10))
        self.assertEqual(scripted_fn("def"), test_multiple_types("def"))
        self.assertEqual(scripted_fn(7.89999), test_multiple_types(7.89999))
        self.assertEqual(scripted_fn([10, 11, 14]), test_multiple_types([10, 11, 14]))

        scripted_fn = torch.jit.script(
            test_multiple_type_refinement,
            example_inputs=[
                (1,),
                ("abc",),
                (8.9,),
                ([3, 4, 5],),
                (True,),
                ({"a": True},),
            ],
        )
        self.assertEqual(scripted_fn(10), test_multiple_type_refinement(10))
        self.assertEqual(scripted_fn("def"), test_multiple_type_refinement("def"))
        self.assertEqual(scripted_fn(7.89999), test_multiple_type_refinement(7.89999))
        self.assertEqual(
            scripted_fn([10, 11, 14]), test_multiple_type_refinement([10, 11, 14])
        )
        self.assertEqual(scripted_fn(False), test_multiple_type_refinement(False))
        self.assertEqual(
            scripted_fn({"abc": True, "def": False}),
            test_multiple_type_refinement({"abc": True, "def": False}),
        )

    def test_class_as_profiled_types(self):
        # [FUNCTION DOC] Scripts a function taking a number and an instance of a user class,
        # profiled on int and float numbers, and checks results against eager for an int and float.
        class UserDefinedClass:
            # [CLASS DOC] HELPER: Plain class with one method used as a profiled argument type.
            def fn(self, b) -> Any:
                # [FUNCTION DOC] HELPER: Asserts the argument is not None; returns the int if
                # positive else minus one, the float if positive else minus one float, else zero.
                assert b is not None  # noqa: S101
                if isinstance(b, int):
                    return b if b > 0 else -1
                elif isinstance(b, float):
                    return b if b > 0.0 else -1.0
                return 0

        def test_model(a, m):
            # [FUNCTION DOC] HELPER: Asserts the first argument is not a bool and returns the
            # result of calling the method on the second argument with it.
            assert not isinstance(a, bool)  # noqa: S101
            return m.fn(a)

        make_global(UserDefinedClass, test_model)

        user_class = UserDefinedClass()
        scripted_fn = torch.jit.script(
            test_model,
            example_inputs=[
                (
                    10,
                    user_class,
                ),
                (
                    10.9,
                    user_class,
                ),
            ],
        )
        self.assertEqual(
            scripted_fn(
                100,
                user_class,
            ),
            test_model(100, user_class),
        )
        self.assertEqual(
            scripted_fn(
                1.9,
                user_class,
            ),
            test_model(1.9, user_class),
        )

    def test_class_with_args_as_profiled_types(self):
        # [FUNCTION DOC] Scripts a function using an instance of a class with a constructor
        # argument as a profiled type, with an instance whose flag is false in the examples, then
        # checks a call with an instance whose flag is true matches eager.
        class ClassWithArgs:
            # [CLASS DOC] HELPER: Plain class storing a boolean flag in its constructor.
            def __init__(self, a: bool):
                # [FUNCTION DOC] HELPER: Stores the boolean flag on the instance.
                self.a = a

            def fn(self, b):
                # [FUNCTION DOC] HELPER: Returns the argument when the flag is true, otherwise
                # minus one.
                if self.a:
                    return b
                else:
                    return -1

        def test_model_with_args(a, m):
            # [FUNCTION DOC] HELPER: Asserts the first argument is not a bool and returns the
            # result of calling the method on the second argument with it.
            assert not isinstance(a, bool)  # noqa: S101
            return m.fn(a)

        make_global(ClassWithArgs, test_model_with_args)

        user_class = ClassWithArgs(False)
        scripted_fn = torch.jit.script(
            test_model_with_args,
            example_inputs=[
                (
                    10,
                    user_class,
                ),
                (
                    10.9,
                    user_class,
                ),
            ],
        )
        self.assertEqual(
            scripted_fn(
                100,
                ClassWithArgs(True),
            ),
            test_model_with_args(100, ClassWithArgs(True)),
        )

    def test_nn_parameter_as_arg(self):
        # [FUNCTION DOC] Scripts a module holding a parameter and calling a helper method with the
        # parameter and an int input, and checks the scripted result matches eager.
        class TestNNParameter(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a two by three parameter of ones and a method adding
            # a value to a given tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the parameter named inp, filled with ones.
                super().__init__()
                self.inp = torch.nn.Parameter(torch.ones(2, 3))

            def add_nn_parameter_with_int(self, x, y):
                # [FUNCTION DOC] HELPER: Returns the tensor addition of its two arguments.
                return torch.add(x, y)

            def forward(self, y):
                # [FUNCTION DOC] HELPER: Adds the input to the stored parameter via the helper.
                return self.add_nn_parameter_with_int(self.inp, y)

        make_global(TestNNParameter)
        pdt_model = TestNNParameter()
        scripted_fn = torch.jit.script(
            pdt_model,
            example_inputs={
                pdt_model: [
                    (10,),
                ],
            },
        )
        self.assertEqual(scripted_fn(20), pdt_model(20))

    def test_fx_tracing_with_typing(self):
        # [FUNCTION DOC] Scripts a module returning a named tuple with a list field, profiled on a
        # list of ints, and checks the scripted result matches eager.
        class FXModelOutput(NamedTuple):
            # [CLASS DOC] HELPER: Named tuple with a single field holding a list of ints.
            result: List[int]

        class FXModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward wraps its argument in the named tuple type.
            def forward(self, a) -> FXModelOutput:
                # [FUNCTION DOC] HELPER: Returns a named tuple whose result field is the argument.
                result = FXModelOutput(result=a)
                return result

        make_global(FXModel, FXModelOutput)
        pdt_model = FXModel()
        scripted_fn = torch.jit.script(
            pdt_model,
            example_inputs={
                pdt_model: [
                    (
                        [
                            10,
                            20,
                        ],
                    ),
                ],
            },
        )
        self.assertEqual(scripted_fn([20]), pdt_model([20]))

    def test_nonetype_as_optional_of_type(self):
        # [FUNCTION DOC] Scripts a function with examples of None combined with float, int and
        # tensor values, so the argument is inferred as optional, and checks results against eager
        # for a float, an int and a tensor input.
        def test_none(a) -> Any:
            # [FUNCTION DOC] HELPER: Returns zero for None; otherwise adds a one-element ones
            # tensor to the argument.
            if a is None:
                return 0
            else:
                return a + torch.ones(1)

        make_global(test_none)

        scripted_fn = torch.jit.script(test_none, example_inputs=[(None,), (10.6,)])
        self.assertEqual(
            scripted_fn(
                30.9,
            ),
            test_none(
                30.9,
            ),
        )

        scripted_fn = torch.jit.script(test_none, example_inputs=[(None,), (10,)])
        self.assertEqual(
            scripted_fn(
                2,
            ),
            test_none(
                2,
            ),
        )

        scripted_fn = torch.jit.script(
            test_none, example_inputs=[(None,), (torch.Tensor(1),)]
        )
        self.assertEqual(
            scripted_fn(
                torch.ones(1),
            ),
            test_none(
                torch.ones(1),
            ),
        )


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
