# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for torch.jit.isinstance, both when compiled with TorchScript (via
# checkScript, which also runs eagerly) and when called eagerly, including container element
# types, Optional, nesting, type refinement, error cases for missing contained types and
# handling of empty containers. Not runnable directly: running the file as a script raises an
# error that points to the umbrella test runner.

import os
import sys
import warnings
from typing import Any, Dict, List, Optional, Tuple

import torch


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase


# Tests for torch.jit.isinstance
class TestIsinstance(JitTestCase):
    # [CLASS DOC] JitTestCase with one test per kind of type check done by torch.jit.isinstance.
    def test_int(self):
        # [FUNCTION DOC] Runs checkScript on the helper with the integer 1.
        def int_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the Any-typed value is an int and not a float.
            assert torch.jit.isinstance(x, int)  # noqa: S101
            assert not torch.jit.isinstance(x, float)  # noqa: S101

        x = 1
        self.checkScript(int_test, (x,))

    def test_float(self):
        # [FUNCTION DOC] Runs checkScript on the helper with the float 1.0.
        def float_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the Any-typed value is a float and not an int.
            assert torch.jit.isinstance(x, float)  # noqa: S101
            assert not torch.jit.isinstance(x, int)  # noqa: S101

        x = 1.0
        self.checkScript(float_test, (x,))

    def test_bool(self):
        # [FUNCTION DOC] Runs checkScript on the helper with the boolean False.
        def bool_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the Any-typed value is a bool and not a float.
            assert torch.jit.isinstance(x, bool)  # noqa: S101
            assert not torch.jit.isinstance(x, float)  # noqa: S101

        x = False
        self.checkScript(bool_test, (x,))

    def test_list(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of three strings.
        def list_str_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a list of str, and is neither a list
            # of int nor a tuple of int.
            assert torch.jit.isinstance(x, List[str])  # noqa: S101
            assert not torch.jit.isinstance(x, List[int])  # noqa: S101
            assert not torch.jit.isinstance(x, Tuple[int])  # noqa: S101

        x = ["1", "2", "3"]
        self.checkScript(list_str_test, (x,))

    def test_list_tensor(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of three tensors.
        def list_tensor_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a list of tensors and not a
            # tuple of int.
            assert torch.jit.isinstance(x, List[torch.Tensor])  # noqa: S101
            assert not torch.jit.isinstance(x, Tuple[int])  # noqa: S101

        x = [torch.tensor([1]), torch.tensor([2]), torch.tensor([3])]
        self.checkScript(list_tensor_test, (x,))

    def test_dict(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a mapping from strings to ints.
        def dict_str_int_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a dict from str to int, and not a dict
            # from int to str or from str to str.
            assert torch.jit.isinstance(x, Dict[str, int])  # noqa: S101
            assert not torch.jit.isinstance(x, Dict[int, str])  # noqa: S101
            assert not torch.jit.isinstance(x, Dict[str, str])  # noqa: S101

        x = {"a": 1, "b": 2}
        self.checkScript(dict_str_int_test, (x,))

    def test_dict_tensor(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a mapping from an int to a tensor.
        def dict_int_tensor_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a dict from int to tensor.
            assert torch.jit.isinstance(x, Dict[int, torch.Tensor])  # noqa: S101

        x = {2: torch.tensor([2])}
        self.checkScript(dict_int_tensor_test, (x,))

    def test_tuple(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a tuple of a string, an int
        # and a string.
        def tuple_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a tuple of str, int, str and is not a
            # tuple of int, str, str nor a one-element tuple of str.
            assert torch.jit.isinstance(x, Tuple[str, int, str])  # noqa: S101
            assert not torch.jit.isinstance(x, Tuple[int, str, str])  # noqa: S101
            assert not torch.jit.isinstance(x, Tuple[str])  # noqa: S101

        x = ("a", 1, "b")
        self.checkScript(tuple_test, (x,))

    def test_tuple_tensor(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a tuple of two tensors.
        def tuple_tensor_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a tuple of two tensors.
            assert torch.jit.isinstance(x, Tuple[torch.Tensor, torch.Tensor])  # noqa: S101

        x = (torch.tensor([1]), torch.tensor([[2], [3]]))
        self.checkScript(tuple_tensor_test, (x,))

    def test_optional(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a tensor.
        def optional_test(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches an optional tensor and does not
            # match an optional string.
            assert torch.jit.isinstance(x, Optional[torch.Tensor])  # noqa: S101
            assert not torch.jit.isinstance(x, Optional[str])  # noqa: S101

        x = torch.ones(3, 3)
        self.checkScript(optional_test, (x,))

    def test_optional_none(self):
        # [FUNCTION DOC] Runs checkScript on the helper with None, which is a case where script
        # and eager results could differ (see the in-body TODO).
        def optional_test_none(x: Any):
            # [FUNCTION DOC] HELPER: Asserts that None matches an optional tensor. A
            # second check on an optional string is commented out because eager and
            # script disagree on it.
            assert torch.jit.isinstance(x, Optional[torch.Tensor])  # noqa: S101
            # assert torch.jit.isinstance(x, Optional[str])
            # TODO: above line in eager will evaluate to True while in
            #       the TS interpreter will evaluate to False as the
            #       first torch.jit.isinstance refines the 'None' type

        x = None
        self.checkScript(optional_test_none, (x,))

    def test_list_nested(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of two dicts of str to int.
        def list_nested(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a list of dict from str to int and not
            # a list of list of str.
            assert torch.jit.isinstance(x, List[Dict[str, int]])  # noqa: S101
            assert not torch.jit.isinstance(x, List[List[str]])  # noqa: S101

        x = [{"a": 1, "b": 2}, {"aa": 11, "bb": 22}]
        self.checkScript(list_nested, (x,))

    def test_dict_nested(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a dict from strings to
        # 3-tuples of strings.
        def dict_nested(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a dict from str to a 3-tuple of strs
            # and not a dict from str to a 3-tuple of ints.
            assert torch.jit.isinstance(x, Dict[str, Tuple[str, str, str]])  # noqa: S101
            assert not torch.jit.isinstance(x, Dict[str, Tuple[int, int, int]])  # noqa: S101

        x = {"a": ("aa", "aa", "aa"), "b": ("bb", "bb", "bb")}
        self.checkScript(dict_nested, (x,))

    def test_tuple_nested(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a tuple of a dict, a list of bools
        # and None.
        def tuple_nested(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches a deeply nested tuple type (dict
            # of str to string 3-tuple, list of bool, optional str) and fails to match three
            # other type descriptions.
            assert torch.jit.isinstance(  # noqa: S101
                x, Tuple[Dict[str, Tuple[str, str, str]], List[bool], Optional[str]]
            )
            assert not torch.jit.isinstance(x, Dict[str, Tuple[int, int, int]])  # noqa: S101
            assert not torch.jit.isinstance(x, Tuple[str])  # noqa: S101
            assert not torch.jit.isinstance(x, Tuple[List[bool], List[str], List[int]])  # noqa: S101

        x = (
            {"a": ("aa", "aa", "aa"), "b": ("bb", "bb", "bb")},
            [True, False, True],
            None,
        )
        self.checkScript(tuple_nested, (x,))

    def test_optional_nested(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of strings.
        def optional_nested(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches an optional list of str.
            assert torch.jit.isinstance(x, Optional[List[str]])  # noqa: S101

        x = ["a", "b", "c"]
        self.checkScript(optional_nested, (x,))

    def test_list_tensor_type_true(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of two random tensors of
        # different shapes.
        def list_tensor_type_true(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is a list of tensors.
            assert torch.jit.isinstance(x, List[torch.Tensor])  # noqa: S101

        x = [torch.rand(3, 3), torch.rand(4, 3)]
        self.checkScript(list_tensor_type_true, (x,))

    def test_tensor_type_false(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of ints.
        def list_tensor_type_false(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value is not a list of tensors.
            assert not torch.jit.isinstance(x, List[torch.Tensor])  # noqa: S101

        x = [1, 2, 3]
        self.checkScript(list_tensor_type_false, (x,))

    def test_in_if(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of ints.
        def list_in_if(x: Any):
            # [FUNCTION DOC] HELPER: Uses isinstance checks as branch conditions; asserts the
            # branch for a list of int is taken and the branch for a list of str is not.
            if torch.jit.isinstance(x, List[int]):
                assert True  # noqa: S101
            if torch.jit.isinstance(x, List[str]):
                assert not True  # noqa: S101

        x = [1, 2, 3]
        self.checkScript(list_in_if, (x,))

    def test_if_else(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a tuple of three strings.
        def list_in_if_else(x: Any):
            # [FUNCTION DOC] HELPER: Uses a tuple of three str check as a condition and asserts
            # the true branch is taken, not the else branch.
            if torch.jit.isinstance(x, Tuple[str, str, str]):
                assert True  # noqa: S101
            else:
                assert not True  # noqa: S101

        x = ("a", "b", "c")
        self.checkScript(list_in_if_else, (x,))

    def test_in_while_loop(self):
        # [FUNCTION DOC] Runs checkScript on the helper with a list of two dicts.
        def list_in_while_loop(x: Any):
            # [FUNCTION DOC] HELPER: Uses an isinstance check for a list of dict as a
            # while-loop condition together with a counter bounded at zero, and asserts
            # the loop body ran once.
            count = 0
            while torch.jit.isinstance(x, List[Dict[str, int]]) and count <= 0:
                count = count + 1
            assert count == 1  # noqa: S101

        x = [{"a": 1, "b": 2}, {"aa": 11, "bb": 22}]
        self.checkScript(list_in_while_loop, (x,))

    def test_type_refinement(self):
        # [FUNCTION DOC] Runs checkScript on the helper first with a list of tensors and then
        # with a dict of strings, to test that isinstance refines the type inside each branch.
        def type_refinement(obj: Any):
            # [FUNCTION DOC] HELPER: If the argument is a list of tensors, clamps each element
            # (so tensor methods type check) and flips a flag; if a dict of str to str,
            # concatenates the values, asserts the result is 111222 and flips the flag. Finally
            # asserts exactly one branch ran.
            hit = False
            if torch.jit.isinstance(obj, List[torch.Tensor]):
                hit = not hit
                for el in obj:
                    # perform some tensor operation
                    y = el.clamp(0, 0.5)  # noqa: F841
            if torch.jit.isinstance(obj, Dict[str, str]):
                hit = not hit
                str_cat = ""
                for val in obj.values():
                    str_cat = str_cat + val
                assert "111222" == str_cat  # noqa: S101
            assert hit  # noqa: S101

        x = [torch.rand(3, 3), torch.rand(4, 3)]
        self.checkScript(type_refinement, (x,))
        x = {"1": "111", "2": "222"}
        self.checkScript(type_refinement, (x,))

    def test_list_no_contained_type(self):
        # [FUNCTION DOC] Checks that using a bare list type without an element type raises a
        # RuntimeError with an explanatory message both when scripting the helper and when
        # calling it eagerly.
        def list_no_contained_type(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches the bare list type, which
            # is invalid.
            assert torch.jit.isinstance(x, List)  # noqa: S101

        x = ["1", "2", "3"]

        err_msg = (
            "Attempted to use List without a contained type. "
            r"Please add a contained type, e.g. List\[int\]"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            torch.jit.script(list_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            list_no_contained_type(x)

    def test_tuple_no_contained_type(self):
        # [FUNCTION DOC] Checks that using a bare tuple type without element types raises a
        # RuntimeError with an explanatory message both when scripting the helper and when
        # calling it eagerly.
        def tuple_no_contained_type(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches the bare tuple type,
            # which is invalid.
            assert torch.jit.isinstance(x, Tuple)  # noqa: S101

        x = ("1", "2", "3")

        err_msg = (
            "Attempted to use Tuple without a contained type. "
            r"Please add a contained type, e.g. Tuple\[int\]"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            torch.jit.script(tuple_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            tuple_no_contained_type(x)

    def test_optional_no_contained_type(self):
        # [FUNCTION DOC] Checks that using a bare optional type without a contained type raises
        # a RuntimeError with an explanatory message both when scripting the helper and when
        # calling it eagerly.
        def optional_no_contained_type(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches the bare optional type,
            # which is invalid.
            assert torch.jit.isinstance(x, Optional)  # noqa: S101

        x = ("1", "2", "3")

        err_msg = (
            "Attempted to use Optional without a contained type. "
            r"Please add a contained type, e.g. Optional\[int\]"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            torch.jit.script(optional_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            optional_no_contained_type(x)

    def test_dict_no_contained_type(self):
        # [FUNCTION DOC] Checks that using a bare dict type without key and value types raises a
        # RuntimeError with an explanatory message both when scripting the helper and when
        # calling it eagerly.
        def dict_no_contained_type(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches the bare dict type, which
            # is invalid.
            assert torch.jit.isinstance(x, Dict)  # noqa: S101

        x = {"a": "aa"}

        err_msg = (
            "Attempted to use Dict without contained types. "
            r"Please add contained type, e.g. Dict\[int, int\]"
        )

        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            torch.jit.script(dict_no_contained_type)
        with self.assertRaisesRegex(
            RuntimeError,
            err_msg,
        ):
            dict_no_contained_type(x)

    def test_tuple_rhs(self):
        # [FUNCTION DOC] Runs checkScript on the helper with an int and with a list of strings,
        # testing a tuple of types as the right-hand side.
        def fn(x: Any):
            # [FUNCTION DOC] HELPER: Asserts the value matches an int-or-list-of-str type tuple
            # and does not match two other type tuples.
            assert torch.jit.isinstance(x, (int, List[str]))  # noqa: S101
            assert not torch.jit.isinstance(x, (List[float], Tuple[int, str]))  # noqa: S101
            assert not torch.jit.isinstance(x, (List[float], str))  # noqa: S101

        self.checkScript(fn, (2,))
        self.checkScript(fn, (["foo", "bar", "baz"],))

    def test_nontuple_container_rhs_throws_in_eager(self):
        # [FUNCTION DOC] Checks that in eager mode a list or a set used as the type argument is
        # rejected with a RuntimeError saying it must be a type or a tuple of types.
        def fn1(x: Any):
            # [FUNCTION DOC] HELPER: Calls torch.jit.isinstance with a list of types as the type
            # argument, which raises in eager mode.
            assert torch.jit.isinstance(x, [int, List[str]])  # noqa: S101

        def fn2(x: Any):
            # [FUNCTION DOC] HELPER: Calls torch.jit.isinstance with a set of types as the type
            # argument, which raises in eager mode.
            assert not torch.jit.isinstance(x, {List[str], Tuple[int, str]})  # noqa: S101

        err_highlight = "must be a type or a tuple of types"

        with self.assertRaisesRegex(RuntimeError, err_highlight):
            fn1(2)

        with self.assertRaisesRegex(RuntimeError, err_highlight):
            fn2(2)

    def test_empty_container_throws_warning_in_eager(self):
        # [FUNCTION DOC] Calls the helper eagerly with an empty list and expects exactly one
        # recorded warning, then with an int and expects no warnings.
        def fn(x: Any):
            # [FUNCTION DOC] HELPER: Calls torch.jit.isinstance with a list of int type and
            # discards the result.
            torch.jit.isinstance(x, List[int])

        with warnings.catch_warnings(record=True) as w:
            x: List[int] = []
            fn(x)
            self.assertEqual(len(w), 1)

        with warnings.catch_warnings(record=True) as w:
            x: int = 2
            fn(x)
            self.assertEqual(len(w), 0)

    def test_empty_container_special_cases(self):
        # [FUNCTION DOC] Calls check_empty_containers with an empty tensor and a multi-element
        # tensor, verifying that neither raises an ambiguous truth value error.
        # Should not throw "Boolean value of Tensor with no values is
        # ambiguous" error
        torch._jit_internal.check_empty_containers(torch.Tensor([]))

        # Should not throw "Boolean value of Tensor with more than
        # one value is ambiguous" error
        torch._jit_internal.check_empty_containers(torch.rand(2, 3))


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
