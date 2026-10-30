# Owner(s): ["oncall: jit"]
# [MODULE DOC] Tests for TorchScript support of Python builtins (hasattr, del, torch._check) and
# for tensor properties and methods (property access, subscript assignment, item, implicit tensor
# to number conversion). Import time side effect: extends sys.path with the test directory.

import inspect
import os
import sys
import unittest
from typing import Dict, List

import torch
from torch.testing import FileCheck


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)
from torch.testing._internal.common_utils import raise_on_run_directly
from torch.testing._internal.jit_utils import JitTestCase, RUN_CUDA


class TestBuiltins(JitTestCase):
    # [CLASS DOC] Test case for builtin function support in TorchScript: hasattr with static
    # arguments, del statements on variables, lists and dictionaries, and torch._check forms.
    """
    Tests for TorchScript support of Python builtin functions.
    """

    def test_has_attr(self):
        # [FUNCTION DOC] Scripts a module looping over submodules, one with attribute a and one
        # with attribute b, collecting hasattr flags and the attribute values via static
        # refinement; compares scripted and eager with checkModule.
        class HasA(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute a.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute a to zero.
                super().__init__()
                self.a = 0

        class HasB(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with an integer attribute b.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Sets attribute b to one.
                super().__init__()
                self.b = 1

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a list of the two modules above and reporting
            # hasattr results.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a module list of one instance of each
                # helper module.
                super().__init__()
                self.mods = torch.nn.ModuleList([HasA(), HasB()])

            def forward(self):
                # [FUNCTION DOC] HELPER: Returns a list holding, for each submodule, the hasattr
                # flags for a and b as integers followed by each attribute that exists.
                # use a list to encode hasattr results
                l = torch.jit.annotate(List[int], [])
                for mod in self.mods:
                    l.append(int(hasattr(mod, "a")))
                    l.append(int(hasattr(mod, "b")))
                    # actually retrieve the attr to test static refinement
                    if hasattr(mod, "a"):
                        l.append(mod.a)
                    if hasattr(mod, "b"):
                        l.append(mod.b)
                return l

        self.checkModule(Mod(), ())

    def test_has_attr_invalid_args(self):
        # [FUNCTION DOC] Asserts scripting raises RuntimeError mentioning hasattr, highlighting
        # the name argument, when the attribute name is not static or the object is not a class
        # type (tensor).
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer whose forward calls hasattr with a
            # runtime name.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores a one by one linear layer.
                super().__init__()
                self.mod = torch.nn.Linear(1, 1)

            def forward(self, name):
                # [FUNCTION DOC] HELPER: Calls hasattr on the linear layer with a non constant
                # name (rejected).
                # not allowed, `name` must be static.
                return hasattr(self.mod, name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward calls hasattr on a tensor with a
            # runtime name.
            def forward(self, name):
                # [FUNCTION DOC] HELPER: Calls hasattr on a new random tensor with a non constant
                # name (rejected).
                # not allowed, `torch.rand` is not a class type
                return hasattr(torch.rand(2, 3), name)

        with self.assertRaisesRegexWithHighlight(RuntimeError, "hasattr", "name"):
            torch.jit.script(Mod())

    def test_del(self):
        # [FUNCTION DOC] Checks that del of a local works and compares with eager, then asserts
        # scripting raises RuntimeError mentioning an undefined value when a deleted variable
        # is used afterwards, when it is deleted only in one branch, and when an unknown name
        # is deleted.
        def fn(x: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Deletes a temporary and returns the input list.
            a = x * 2
            del a
            return x

        self.checkScript(fn, ([1, 2, 3],))

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @torch.jit.script
            def fn(x):
                # [FUNCTION DOC] HELPER: Deletes a variable then returns it (error expected at
                # compile time).
                a = x**2
                del a
                return a  # noqa: F821

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "a"):

            @torch.jit.script
            def fn(x):
                # [FUNCTION DOC] HELPER: Deletes a variable only in one branch then returns it
                # (error expected).
                a = x**2
                if a:
                    del a
                return a

        with self.assertRaisesRegexWithHighlight(RuntimeError, "undefined value", "b"):

            @torch.jit.script
            def fn(x):
                # [FUNCTION DOC] HELPER: Deletes a name that was never defined (error expected).
                a = x**2
                del b  # noqa: F821
                return a

    def test_del_multiple_operands(self):
        # [FUNCTION DOC] Checks del with several operands: on local names, on list elements and on
        # dictionary keys; the scripted results must equal eager results.
        def fn(x: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Unpacks three elements and deletes the three names at once.
            a, b, c = x[0], x[1], x[2]
            del a, b, c
            return x

        self.checkScript(fn, ([1, 2, 3],))

        def del_list_multiple_operands(x: List[int]) -> List[int]:
            # [FUNCTION DOC] HELPER: Deletes two elements of a list in one statement and returns
            # the list.
            del x[0], x[1]
            return x

        py_out = del_list_multiple_operands([0, 1, 2])
        jit_out = torch.jit.script(del_list_multiple_operands)([0, 1, 2])
        self.assertEqual(py_out, jit_out)

        def del_dict_multiple_operands(x: Dict[str, int]) -> Dict[str, int]:
            # [FUNCTION DOC] HELPER: Deletes two keys of a dictionary in one statement and
            # returns it.
            del x["hi"], x["there"]
            return x

        py_out = del_dict_multiple_operands({"hi": 5, "there": 6})
        jit_out = torch.jit.script(del_dict_multiple_operands)({"hi": 5, "there": 6})
        self.assertEqual(py_out, jit_out)

    def test_torch_check(self):
        # [FUNCTION DOC] Checks torch._check in scripted code with positional and keyword
        # condition and message, reversed keyword order and use in a loop: results equal the
        # input, message constants appear in graphs, and graphs contain a conditional leading to a
        # raise (plus a loop where used).
        """Test torch._check functionality with flexible argument handling"""

        def test_check_basic(x):
            # [FUNCTION DOC] HELPER: Checks a condition on the tensor sum without a message and
            # returns x.
            torch._check(x.sum().item() > -1000)
            return x

        def test_check_with_message(x):
            # [FUNCTION DOC] HELPER: Checks a condition with a positional message and returns x.
            torch._check(x.sum().item() > -1000, "Tensor sum must be reasonable")
            return x

        def test_check_with_kwarg_message(x):
            # [FUNCTION DOC] HELPER: Checks a condition with the message passed by keyword and
            # returns x.
            torch._check(
                x.sum().item() > -1000, message="Tensor sum must be reasonable"
            )
            return x

        def test_check_cond_kwarg(x):
            # [FUNCTION DOC] HELPER: Passes only the condition, by keyword, and returns x.
            torch._check(cond=x.sum().item() > -1000)
            return x

        def test_check_both_kwargs(x):
            # [FUNCTION DOC] HELPER: Passes condition and message both by keyword and returns x.
            torch._check(cond=x.sum().item() > -1000, message="Both as kwargs")
            return x

        def test_check_kwargs_reversed(x):
            # [FUNCTION DOC] HELPER: Passes message before condition, both by keyword, and
            # returns x.
            torch._check(message="Reversed order", cond=x.sum().item() > -1000)
            return x

        def test_check_in_loop(x):
            # [FUNCTION DOC] HELPER: Checks a condition for every element of the list form of
            # the tensor.
            sizes = torch.jit.annotate(List[int], x.tolist())
            for s in sizes:
                torch._check(s > -100)
            return x

        test_tensor = torch.tensor([1, 2, 3])

        # Test all variations
        self.checkScript(test_check_basic, (test_tensor,))
        self.checkScript(test_check_with_message, (test_tensor,))
        self.checkScript(test_check_with_kwarg_message, (test_tensor,))
        self.checkScript(test_check_cond_kwarg, (test_tensor,))
        self.checkScript(test_check_both_kwargs, (test_tensor,))
        self.checkScript(test_check_kwargs_reversed, (test_tensor,))
        self.checkScript(test_check_in_loop, (test_tensor,))

        # Test that the compiled functions work correctly
        scripted_basic = torch.jit.script(test_check_basic)
        scripted_with_message = torch.jit.script(test_check_with_message)
        scripted_with_kwarg = torch.jit.script(test_check_with_kwarg_message)
        scripted_cond_kwarg = torch.jit.script(test_check_cond_kwarg)
        scripted_both_kwargs = torch.jit.script(test_check_both_kwargs)
        scripted_kwargs_reversed = torch.jit.script(test_check_kwargs_reversed)
        scripted_in_loop = torch.jit.script(test_check_in_loop)

        # These should all succeed without throwing
        result1 = scripted_basic(test_tensor)
        result2 = scripted_with_message(test_tensor)
        result3 = scripted_with_kwarg(test_tensor)
        result4 = scripted_cond_kwarg(test_tensor)
        result5 = scripted_both_kwargs(test_tensor)
        result6 = scripted_kwargs_reversed(test_tensor)
        result7 = scripted_in_loop(test_tensor)

        # Results should be the same as input
        for result in [result1, result2, result3, result4, result5, result6, result7]:
            self.assertEqual(result, test_tensor)

        # Check that the message constants are present in the graphs
        FileCheck().check("Tensor sum must be reasonable").run(
            scripted_with_message.graph
        )
        FileCheck().check("Tensor sum must be reasonable").run(
            scripted_with_kwarg.graph
        )
        FileCheck().check("Both as kwargs").run(scripted_both_kwargs.graph)
        FileCheck().check("Reversed order").run(scripted_kwargs_reversed.graph)

        # Verify the graphs contain some computation (not just empty)
        basic_graph_str = str(scripted_basic.graph)
        self.assertTrue(
            len(basic_graph_str) > 100, "Basic graph should contain some computation"
        )

        # Verify the loop case contains a loop
        FileCheck().check("prim::Loop").run(scripted_in_loop.graph)

        for scripted_func in [
            scripted_basic,
            scripted_with_message,
            scripted_with_kwarg,
            scripted_cond_kwarg,
            scripted_both_kwargs,
            scripted_kwargs_reversed,
        ]:
            FileCheck().check("prim::If").check("prim::RaiseException").run(
                scripted_func.graph
            )

    def test_torch_check_invalid_args(self):
        # [FUNCTION DOC] Asserts scripting raises RuntimeError for invalid argument usage of
        # torch._check: too many arguments, unknown keyword, condition given twice, missing
        # condition, no arguments, and too many total arguments with keywords.
        """Test torch._check with invalid arguments"""

        # Test too many arguments
        with self.assertRaisesRegex(
            RuntimeError, "torch._check\\(\\) expects 1 or 2 arguments"
        ):

            @torch.jit.script
            def too_many_args(x):
                # [FUNCTION DOC] HELPER: Calls the check with three positional
                # arguments (rejected).
                torch._check(True, "msg", "extra")
                return x

        # Test invalid keyword argument
        with self.assertRaisesRegex(RuntimeError, "unexpected keyword argument"):

            @torch.jit.script
            def invalid_kwarg(x):
                # [FUNCTION DOC] HELPER: Calls the check with an unknown keyword
                # argument (rejected).
                torch._check(True, invalid_arg="msg")
                return x

        # Test duplicate cond argument (positional + keyword)
        with self.assertRaisesRegex(
            RuntimeError, "multiple values for argument 'cond'"
        ):

            @torch.jit.script
            def duplicate_cond(x):
                # [FUNCTION DOC] HELPER: Passes the condition positionally and as a
                # keyword (rejected).
                torch._check(True, cond=False)
                return x

        # Test missing required cond argument
        with self.assertRaisesRegex(RuntimeError, "missing required argument 'cond'"):

            @torch.jit.script
            def missing_cond(x):
                # [FUNCTION DOC] HELPER: Passes only a message keyword (rejected).
                torch._check(message="msg only")
                return x

        # Test no arguments at all
        with self.assertRaisesRegex(
            RuntimeError, "torch._check\\(\\) expects 1 or 2 arguments"
        ):

            @torch.jit.script
            def no_args(x):
                # [FUNCTION DOC] HELPER: Calls the check with no arguments (rejected).
                torch._check()
                return x

        # Test too many total arguments (positional + keyword)
        with self.assertRaisesRegex(
            RuntimeError, "torch._check\\(\\) expects 1 or 2 arguments"
        ):

            @torch.jit.script
            def too_many_total_args(x):
                # [FUNCTION DOC] HELPER: Passes two positionals plus the condition
                # keyword (rejected).
                torch._check(True, "msg", cond=False)
                return x


class TestTensorBuiltins(JitTestCase):
    # [CLASS DOC] Test case for tensor attribute, subscript assignment, item and scalar conversion
    # behavior in TorchScript.
    def test_tensor_properties(self):
        # [FUNCTION DOC] Enumerates the non-callable, non-underscore attributes of a float tensor
        # (skipping imag, which raises for non complex tensors), compiles an accessor for each,
        # and compares compiled results with eager values; grad_fn, output_nr and grad_dtype are
        # not compiled and dtype and layout are compiled but their values not compared.
        def should_keep(tensor, name):
            # [FUNCTION DOC] HELPER: True if the named attribute is not a routine and does not
            # start with an underscore.
            if inspect.isroutine(getattr(tensor, name)):
                return False
            if name.startswith("_"):
                return False
            return True

        tensor = torch.arange(4, dtype=torch.float).view(2, 2)
        keys = dir(tensor)

        # real and imag are only implemented for complex tensors.
        self.assertRaises(RuntimeError, lambda: should_keep(tensor, "imag"))
        keys.remove("imag")

        properties = [p for p in keys if should_keep(tensor, p)]

        code_template = """
        def fn(x):
            return x.{}
        """

        EQUALITY_MISMATCH = {
            # TorchScript doesn't have real enums so they return an int instead
            # of the actual value
            "dtype",
            "layout",
        }
        MISSING_PROPERTIES = {
            "grad_fn",
            # This is an undocumented property so it's not included
            "output_nr",
            # We don't plan to support grad_dtype in TorchScript
            "grad_dtype",
        }

        for p in properties:
            if p in MISSING_PROPERTIES:
                continue
            code = code_template.format(p)
            cu = torch.jit.CompilationUnit()
            cu.define(code)
            if p in EQUALITY_MISMATCH:
                continue
            self.assertEqual(getattr(tensor, p), cu.fn(tensor))

    def test_tensor_subscript_assign(self):
        # [FUNCTION DOC] Runs checkScript on five functions assigning into a zero tensor with a
        # tensor or integer index and a tensor or integer value, in uint8 and float32.
        def fn1(x):
            # [FUNCTION DOC] HELPER: Assigns a uint8 tensor value at a tensor index.
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = torch.tensor(2, dtype=torch.uint8)
            return a

        def fn2(x):
            # [FUNCTION DOC] HELPER: Assigns an integer at an integer index of a uint8 tensor.
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = 2
            return a

        def fn3(x):
            # [FUNCTION DOC] HELPER: Assigns an integer at a tensor index of a uint8 tensor.
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[torch.tensor(0)] = 2
            return a

        def fn4(x):
            # [FUNCTION DOC] HELPER: Assigns a uint8 tensor value at an integer index.
            a = torch.zeros_like(x, dtype=torch.uint8)
            a[0] = torch.tensor(2, dtype=torch.uint8)
            return a

        def fn5(x):
            # [FUNCTION DOC] HELPER: Assigns an integer at a tensor index of a float32 tensor.
            a = torch.zeros_like(x, dtype=torch.float32)
            a[torch.tensor(0)] = 2
            return a

        for fn in (fn1, fn2, fn3, fn4, fn5):
            self.checkScript(fn, (torch.zeros(2, dtype=torch.uint8),))

    @unittest.skipIf(not RUN_CUDA, "requires CUDA")
    def test_tensor_subscript_assign_device(self):
        # [FUNCTION DOC] CUDA only (skipped otherwise): tensor index assignment on a CUDA float
        # tensor via checkScript.
        def fn6(x):
            # [FUNCTION DOC] HELPER: Assigns an integer at a tensor index of a CUDA float tensor.
            a = torch.zeros_like(x, dtype=torch.float32, device="cuda")
            a[torch.tensor(0)] = 2
            return a

        self.checkScript(fn6, (torch.zeros(2, dtype=torch.float32, device="cuda"),))

    def test_tensor_item(self):
        # [FUNCTION DOC] Checks that casting the result of item to int and float yields a
        # tuple typed as int and float in the graph, and that float and int tensors both
        # pass checkScript.
        def test_scalar_cast(x):
            # [FUNCTION DOC] HELPER: Returns the item cast to both int and float.
            scalar = x.item()
            return int(scalar), float(scalar)

        graph = torch.jit.script(test_scalar_cast).graph
        FileCheck().check("(int, float) = prim::TupleConstruct").run(graph)
        self.checkScript(test_scalar_cast, (torch.tensor(1.0),))
        self.checkScript(test_scalar_cast, (torch.tensor(1),))

    def test_method_on_number(self):
        # [FUNCTION DOC] Asserts scripting a function that calls a method on an integer constant
        # raises RuntimeError saying the object has no attribute or method.
        def func():
            # [FUNCTION DOC] HELPER: Calls an add method on an integer constant (rejected).
            c = 1
            return c.add(1)

        with self.assertRaisesRegex(RuntimeError, "object has no attribute or method"):
            torch.jit.script(func)

    # testing implicit conversion of tensors to scalars to match function arguments
    def test_scalar_to_num_conversions(self):
        # [FUNCTION DOC] Checks implicit conversion of tensors to scalars: no implicit tensor to
        # number node for a plain function, the error location appears in the message for a bad
        # dimension tensor, and for several tensors scripted and eager versions either give equal
        # results or both raise.
        @torch.jit.script
        def multiple_defs(x):
            # [FUNCTION DOC] HELPER: Scripted function adding an integer constant to its input.
            c = 1
            x = x + c
            return x

        self.assertTrue("ImplicitTensorToNum" not in str(multiple_defs.graph))

        @torch.jit.script
        def tensor_to_int_script(x, tensor):
            # [FUNCTION DOC] HELPER: Scripted function using a tensor argument as the dimension
            # of unsqueeze.
            return x.unsqueeze(tensor)

        # location present in error message
        with self.assertRaisesRegex(RuntimeError, "x.unsqueeze"):
            tensor_to_int_script(torch.tensor([2]), torch.tensor([2, 2]))

        def tensor_to_int(x, tensor):
            # [FUNCTION DOC] HELPER: Eager counterpart using the tensor as the
            # unsqueeze dimension.
            return x.unsqueeze(tensor)

        @torch.jit.script
        def tensor_to_float_script(x, tensor):
            # [FUNCTION DOC] HELPER: Scripted function passing a tensor as the value of addcmul.
            return x.addcmul(tensor, tensor, value=tensor)

        def tensor_to_float(x, tensor):
            # [FUNCTION DOC] HELPER: Eager counterpart passing a tensor as the value of addcmul.
            return x.addcmul(tensor, tensor, value=tensor)

        x = torch.zeros(10)
        # float tensor, float tensor with grad, int tensor (can't set grad on int tensor)
        tensors = [
            torch.tensor(1.1),
            torch.tensor(1.1, requires_grad=True),
            torch.tensor(0),
            torch.tensor([2]),
        ]

        script_funs = [tensor_to_int_script, tensor_to_float_script]
        funs = [tensor_to_int, tensor_to_float]

        # return the result, or whether exception was thrown
        def test_func(func, x, tensor):
            # [FUNCTION DOC] HELPER: Calls the function; returns its result, or True if it raises
            # a runtime or type error.
            try:
                result = func(x, tensor)
            except RuntimeError:
                result = True
            except TypeError:
                result = True
            return result

        # assert result or exception equal for each (function, inputs)
        for tensor in tensors:
            for i in range(len(script_funs)):
                self.assertEqual(
                    test_func(script_funs[i], x, tensor), test_func(funs[i], x, tensor)
                )


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
