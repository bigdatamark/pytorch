# Owner(s): ["oncall: jit"]
# ruff: noqa: F841
# [MODULE DOC] Miscellaneous TorchScript tests: f-strings, keyword-only arguments, tuple item
# assignment errors, Future type expressions, Any refinement, hacked_twin index_put operators,
# mobile op-name export with module interfaces, list literal inference, legacy tensor
# constructors, BroadcastingList annotations, IR parsing of constants, stacked decorators and
# operator overload ordering. Import time: imports TestModuleInterface from the sibling jit test
# module so that its tests also run when this file is collected (the import is otherwise
# unused), and appends the test directory to sys.path.

import os
import sys
import unittest
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.nn as nn
import torch.testing._internal.jit_utils
from jit.test_module_interface import TestModuleInterface  # noqa: F401
from torch import jit
from torch.testing import FileCheck
from torch.testing._internal.common_utils import (
    freeze_rng_state,
    raise_on_run_directly,
    skipIfTorchDynamo,
)
from torch.testing._internal.jit_utils import JitTestCase, make_global, RUN_CUDA_HALF


# Make the helper files in test/ importable
pytorch_test_dir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(pytorch_test_dir)


class TestMisc(JitTestCase):
    # [CLASS DOC] JitTestCase grouping many small, unrelated TorchScript frontend, schema and
    # IR tests.
    def test_joined_str(self):
        # [FUNCTION DOC] Runs a plain function using f-strings under captured stdout in eager
        # mode, then scripts it and runs the original function again under capture. Asserts
        # equal outputs and equal captured text.
        # KNOWN ISSUE: the scripted version is never actually called (the eager function is
        #     called twice), and the checkScript line is commented out.
        def func(x):
            # [FUNCTION DOC] HELPER: Prints several formatted strings (some f-strings, one
            # plain) and returns the input tensor plus one.
            hello, test = "Hello", "test"
            print(f"{hello + ' ' + test}, I'm a {test}")
            print("format blank")
            hi = "hi"
            print(f"stuff before {hi}")
            print(f"{hi} stuff after")
            return x + 1

        x = torch.arange(4.0, requires_grad=True)
        # TODO: Add support for f-strings in string parser frontend
        # self.checkScript(func, [x], optimize=True, capture_output=True)

        with self.capture_stdout() as captured:
            out = func(x)

        scripted = torch.jit.script(func)
        with self.capture_stdout() as captured_script:
            out_script = func(x)

        self.assertEqual(out, out_script)
        self.assertEqual(captured, captured_script)

    def test_kwarg_support(self):
        # [FUNCTION DOC] Checks keyword-only argument handling. Scripting a module whose forward
        # has keyword-only arguments with a mismatched default raises NotSupportedError
        # mentioning a variable number of arguments. A valid keyword-only module is scripted;
        # calling it with no arguments raises RuntimeError about the missing argument, calling
        # it positionally raises an error about positional args, and calling it with keywords
        # returns the tuple of the two values.
        with self.assertRaisesRegex(
            torch.jit.frontend.NotSupportedError, "variable number of arguments"
        ):

            class M(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward has keyword-only arguments, one default
            # of the wrong type.
                def forward(self, *, n_tokens: int, device_name: str = 2):
                # [FUNCTION DOC] HELPER: Does nothing; exists only so scripting fails on
                # its signature.
                    pass

            torch.jit.script(M())

        class M(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with keyword-only arguments returning them as a tuple.
            def forward(self, *, n_tokens: int, device_name: str):
            # [FUNCTION DOC] HELPER: Returns the two keyword-only arguments as a tuple.
                return n_tokens, device_name

        sm = torch.jit.script(M())

        with self.assertRaisesRegex(
            RuntimeError, "missing value for argument 'n_tokens'"
        ):
            sm()

        with self.assertRaisesRegex(RuntimeError, "positional arg"):
            sm(3, "hello")

        self.assertEqual(sm(n_tokens=3, device_name="hello"), (3, "hello"))

    def test_tuple_subscripted_assign(self):
        # [FUNCTION DOC] Checks that scripting a function that assigns to, or augmented-
        # assigns, a tuple element raises RuntimeError mentioning subscripted assignment or
        # augmented assignment.
        with self.assertRaisesRegex(RuntimeError, "subscripted assignment"):

            @torch.jit.script
            def foo(a: Tuple[int, int]) -> None:
            # [FUNCTION DOC] HELPER: Scripted function attempting plain item assignment on a
            # tuple; never runs.
                a[0] = a[1]

        with self.assertRaisesRegex(RuntimeError, "augmented assignment"):

            @torch.jit.script
            def bar(a: Tuple[int, int]) -> None:
            # [FUNCTION DOC] HELPER: Scripted function attempting augmented item assignment on a
            # tuple; never runs.
                a[0] += a[1]

    def test_subexpression_List_Future(self):
        # [FUNCTION DOC] Scripts a function taking a list of futures and returning one future,
        # and checks with FileCheck that the graph shows the future-of-int type twice.
        @torch.jit.script
        def fn(x: List[torch.jit.Future[int]]) -> torch.jit.Future[int]:
            # [FUNCTION DOC] HELPER: Returns the first element of a list of futures.
            return x[0]

        FileCheck().check("Future[int]").check("Future[int]").run(fn.graph)

    def test_subexpression_Future_annotate(self):
        # [FUNCTION DOC] Scripts a function that builds an annotated empty list of futures and
        # indexes it; checks the graph contains the list-of-future type string.
        @torch.jit.script
        def fn() -> torch.jit.Future[int]:
            # [FUNCTION DOC] HELPER: Creates an empty list annotated as a list of futures and
            # returns its first element.
            x: List[torch.jit.Future[int]] = []
            return x[0]

        FileCheck().check("Future[int][]").run(fn.graph)

    def test_future_isinstance(self):
        # [FUNCTION DOC] Scripts a function that uses an isinstance check against the future
        # type on an Any argument and checks the graph mentions the future-of-int type.
        @torch.jit.script
        def fn(x: Any) -> torch.jit.Future[int]:
            # [FUNCTION DOC] HELPER: Asserts its Any argument is a future of int and returns it.
            assert isinstance(x, jit.Future[int])  # noqa: S101
            return x

        FileCheck().check("Future[int]").run(fn.graph)

    def test_str_refine_any(self):
        # [FUNCTION DOC] Scripts a function that refines an Any argument with an isinstance
        # check for str. Checks the scripted function returns the foo string for an int and the
        # same string for a string.
        def forward(x: Any) -> str:
            # [FUNCTION DOC] HELPER: Returns its argument if it is a string, otherwise the
            # literal foo string.
            if isinstance(x, str):
                return x
            return "foo"

        forward = torch.jit.script(forward)
        self.assertEqual(forward(1), "foo")
        self.assertEqual(forward("bar"), "bar")

    def test_subexpression_Tuple_int_int_Future(self):
        # [FUNCTION DOC] Scripts a function taking a tuple containing a future and returning a
        # sub-tuple; checks the graph shows both tuple types with the future element.
        @torch.jit.script
        def fn(
            x: Tuple[int, int, torch.jit.Future[int]],
        ) -> Tuple[int, torch.jit.Future[int]]:
            # [FUNCTION DOC] HELPER: Returns a tuple of the first and third items of the
            # input tuple.
            return x[0], x[2]

        FileCheck().check("(int, int, Future[int])").check("(int, Future[int])").run(
            fn.graph
        )

    def test_subexpression_Dict_int_Future(self):
        # [FUNCTION DOC] Scripts a function that looks up a future in a dictionary keyed by int;
        # checks the graph shows the dict type with future values and the future type.
        @torch.jit.script
        def fn(x: Dict[int, torch.jit.Future[int]], y: int) -> torch.jit.Future[int]:
            # [FUNCTION DOC] HELPER: Returns the dictionary entry for the given key.
            return x[y]

        FileCheck().check("Dict(int, Future(int))").check("Future[int]").run(fn.graph)

    def test_subexpression_Optional(self):
        # [FUNCTION DOC] Scripts a function taking an optional dictionary of futures and checks
        # the graph contains the optional dict-of-future type.
        @torch.jit.script
        def fn(
            x: Optional[Dict[int, torch.jit.Future[int]]],
        ) -> Optional[torch.jit.Future[int]]:
            # [FUNCTION DOC] HELPER: Returns the entry for key zero if the dictionary is given,
            # otherwise None.
            if x is not None:
                return x[0]
            else:
                return None

        FileCheck().check("Dict(int, Future(int))?").run(fn.graph)

    def test_if_returning_any(self):
        # [FUNCTION DOC] Uses checkScript to verify a function whose return type is Any can
        # return different types (a tensor product or a string) from different if branches.
        """
        Check that an if statement can return different
        types early from each branch when the return
        type of the function is Any.
        """

        def if_function(inp: torch.Tensor) -> Any:
            # [FUNCTION DOC] HELPER: Returns the squared tensor when the first dimension is one,
            # else a string.
            if inp.shape[0] == 1:
                return inp * inp
            else:
                return "str"

        self.checkScript(if_function, (torch.randn(5),))

    def test_hacked_twin(self):
        # [FUNCTION DOC] Compares the index_put hacked_twin overloads (functional and in-place)
        # against the regular index_put, using two identically generated data sets. Asserts
        # equal functional outputs and equal mutated inputs after the in-place calls.
        def gen_data():
            # [FUNCTION DOC] HELPER: Under a frozen random state, returns a random float vector,
            # random integer indices and random values, so repeated calls give identical data.
            with freeze_rng_state():
                return torch.randn(10), torch.randint(10, (20,)), torch.randn(20)

        (
            input,
            index,
            value,
        ) = gen_data()
        (
            input1,
            index1,
            value1,
        ) = gen_data()
        out1 = torch.ops.aten.index_put.hacked_twin(
            input, [index], value, accumulate=False
        )
        out2 = torch.index_put(input1, [index1], value1, accumulate=False)
        self.assertEqual(out1, out2)

        torch.ops.aten.index_put_.hacked_twin(input, [index], value, accumulate=False)
        torch.index_put_(input1, [index1], value1, accumulate=False)
        self.assertEqual(input, input1)

    def test_unsafe_hacked_twin(self):
        # [FUNCTION DOC] Same comparison as the previous test for the unsafe index_put
        # hacked_twin overload and the unsafe index overload. Also scripts two functions calling
        # the unsafe index_put op and asserts scripted and eager results match on cloned inputs.
        def gen_data():
            # [FUNCTION DOC] HELPER: Under a frozen random state, returns a random float vector,
            # random integer indices and random values, so repeated calls give identical data.
            with freeze_rng_state():
                return torch.randn(10), torch.randint(10, (20,)), torch.randn(20)

        (
            input,
            index,
            value,
        ) = gen_data()
        (
            input1,
            index1,
            value1,
        ) = gen_data()
        out1 = torch.ops.aten._unsafe_index_put.hacked_twin(
            input, [index], value, accumulate=False
        )
        out2 = torch.index_put(input1, [index1], value1, accumulate=False)
        self.assertEqual(out1, out2)

        torch.ops.aten._unsafe_index.Tensor_hacked_twin(input, [index])
        torch.index_put(input1, [index1], value1, accumulate=False)
        self.assertEqual(input, input1)

        def index_put_fn(input, index, value):
            # [FUNCTION DOC] HELPER: Returns the result of the unsafe index_put op with the
            # index wrapped in a list and accumulate off.
            return torch.ops.aten._unsafe_index_put(
                input, [index], value, accumulate=False
            )

        input2, index2, value2 = gen_data()
        script_index_put_fn = torch.jit.script(index_put_fn)
        expect = index_put_fn(input2.clone(), index2, value2)
        actual = script_index_put_fn(input2.clone(), index2, value2)
        self.assertEqual(expect, actual)

        def index_fn(input, index, value):
            # [FUNCTION DOC] HELPER: Same body as the previous helper (it also calls the unsafe
            # index_put op).
            # KNOWN ISSUE: despite its name it does not test the unsafe index op.
            return torch.ops.aten._unsafe_index_put(
                input, [index], value, accumulate=False
            )

        script_index_fn = torch.jit.script(index_fn)
        expect = index_fn(input2.clone(), index2, value2)
        actual = script_index_fn(input2.clone(), index2, value2)
        self.assertEqual(expect, actual)

    def test_export_opnames_interface(self):
        # [FUNCTION DOC] Verifies operator-name export for modules using a module interface when
        # the mobile interface-call export flag is enabled. Scripts a wrapper holding a
        # submodule typed as the interface and implemented by the multiply/reciprocal module;
        # checks the exported op names contain the mul and reciprocal ops. After replacing the
        # submodule with the add/multiply module checks the exported names contain the add and
        # mul scalar ops. Side effect: enables a global export flag and never resets it.
        @torch.jit.interface
        class OneTwoModule(nn.Module):
            # [CLASS DOC] HELPER: TorchScript interface declaring one, two and forward methods
            # on tensors.
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration taking two tensors; body
                # is a placeholder.
                pass

            def two(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface method declaration taking one tensor; body is
                # a placeholder.
                pass

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Interface forward declaration; body is a placeholder.
                pass

        class FooMod(nn.Module):
            # [CLASS DOC] HELPER: Interface implementation using addition and
            # scalar multiplication.
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the sum of the two inputs.
                return x + y

            def two(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the input multiplied by two.
                return 2 * x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Applies the doubling method then the add method with
                # the original input.
                return self.one(self.two(x), x)

        class BarMod(nn.Module):
            # [CLASS DOC] HELPER: Interface implementation using multiplication and reciprocal.
            def one(self, x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns the elementwise product of the two inputs.
                return x * y

            def two(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns two divided by the input, which lowers to the
                # reciprocal op.
                return 2 / x

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Multiplies the input with itself, then applies the
                # division method.
                return self.two(self.one(x, x))

        make_global(OneTwoModule)

        class M(nn.Module):
            # [CLASS DOC] HELPER: Wrapper module with a submodule annotated with the
            # interface type.
            sub: OneTwoModule

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Stores the multiply/reciprocal implementation as
                # the submodule.
                super().__init__()
                self.sub = BarMod()

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls forward on the submodule and returns the result.
                return self.sub.forward(x)

        def use_module_interface(mod_list: List[OneTwoModule], x: torch.Tensor):
            # [FUNCTION DOC] HELPER: Sums the forward results of the first two interface modules
            # in a list. It is defined but never called or scripted in this test.
            return mod_list[0].forward(x) + mod_list[1].forward(x)

        torch._C._enable_mobile_interface_call_export()
        scripted_M_mod = torch.jit.script(M())
        self.assertTrue(
            {"aten::mul.Scalar", "aten::mul.Tensor", "aten::reciprocal"}.issubset(
                set(torch.jit.export_opnames(scripted_M_mod))
            )
        )

        scripted_M_mod.sub = torch.jit.script(FooMod())
        self.assertTrue(
            {"aten::add.Tensor", "aten::mul.Scalar"}.issubset(
                set(torch.jit.export_opnames(scripted_M_mod))
            )
        )

    def test_math_inf(self):
        # [FUNCTION DOC] Checks that a function returning the imported infinity float constant
        # can be scripted with the same result as eager.
        from math import inf

        def foo():
            # [FUNCTION DOC] HELPER: Returns the infinity constant imported from the
            # math module.
            return inf

        self.checkScript(foo, ())

    def test_list_literal_infer(self):
        # [FUNCTION DOC] Checks list literal type inference. An empty list literal passed as an
        # int list argument scripts fine. Passing an annotated list of tensors, or a named empty
        # list variable, raises RuntimeError at script time. A returned empty list literal
        # defaults to a list of tensors, confirmed via the graph.
        def expects_intlist(x: List[int]):
            # [FUNCTION DOC] HELPER: Appends three to the given integer list and returns it.
            x.append(3)
            return x

        def foo():
            # [FUNCTION DOC] HELPER: Passes an empty list literal to the int list helper.
            return expects_intlist([])

        self.checkScript(foo, ())

        def annotated_list_fail():
            # [FUNCTION DOC] HELPER: Passes an empty list annotated as a tensor list where an
            # int list is expected; scripting it must fail. It also refers to an undefined
            # tensor name.
            return expects_intlist(torch.jit.annotate([], List[Tensor]))  # noqa: F821

        with self.assertRaises(RuntimeError):
            torch.jit.script(annotated_list_fail)

        def non_temporary_fail():
            # [FUNCTION DOC] HELPER: Passes a named empty list variable to the int list helper;
            # scripting it must fail since the variable type defaults to tensor list.
            a = []
            return expects_intlist(a)

        with self.assertRaises(RuntimeError):
            torch.jit.script(non_temporary_fail)

        @torch.jit.script
        def test_return():
            # [FUNCTION DOC] HELPER: Scripted function returning an empty list literal.
            return []

        FileCheck().check("Tensor[] = prim::ListConstruct").run(test_return.graph)

    def test_legacy_tensor_constructor(self):
        # [FUNCTION DOC] Tests legacy typed tensor constructors in TorchScript. Constructing
        # from a list for every dtype, from sizes, and with no arguments matches eager results
        # (the sized case is compared after filling both with ones). Mixed positional arguments
        # raise RuntimeError about non-integer positional arguments, and an unknown keyword
        # raises RuntimeError mentioning that keyword.
        # testing PyObject overload
        def test_all_dtypes():
            # [FUNCTION DOC] HELPER: Returns a tuple of single-element tensors built with each
            # legacy constructor.
            return (
                torch.BoolTensor([2]),
                torch.LongTensor([3]),
                torch.ByteTensor([4]),
                torch.CharTensor([5]),
                torch.DoubleTensor([6]),
                torch.FloatTensor([7]),
                torch.IntTensor([8]),
                torch.ShortTensor([1]),
                torch.HalfTensor([1]),
            )

        self.checkScript(test_all_dtypes, ())

        # now test empty overload
        def empty_overload():
            # [FUNCTION DOC] HELPER: Returns an uninitialized long tensor of shape two by three
            # by four.
            return torch.LongTensor(2, 3, 4)

        eager = empty_overload()
        jit = torch.jit.script(empty_overload)()
        eager[:] = 1
        jit[:] = 1
        self.assertEqual(eager, jit)

        def no_inputs():
            # [FUNCTION DOC] HELPER: Returns an empty double tensor from the no-
            # argument constructor.
            return torch.DoubleTensor()

        self.checkScript(no_inputs, ())

        # bad schema
        def multiple_args():
            # [FUNCTION DOC] HELPER: Calls the long constructor with an int and a list;
            # scripting must fail.
            return torch.LongTensor(1, [2])

        with self.assertRaisesRegex(
            RuntimeError, "multiple positional arguments that were not all integers"
        ):
            torch.jit.script(multiple_args)

        # kwarg bad schema
        def bad_kwarg():
            # [FUNCTION DOC] HELPER: Calls the long constructor with an invalid keyword
            # argument; scripting must fail.
            return torch.LongTensor(hello="1")

        with self.assertRaisesRegex(RuntimeError, "hello"):
            torch.jit.script(bad_kwarg)

    def test_broadcasting_list(self):
        # [FUNCTION DOC] Checks that the BroadcastingList annotation and the size-2 type alias
        # are accepted by the script compiler: a function summing two int elements gives eight
        # for an int four, and one summing two float elements gives nine for a float 4.5.
        """
        Test BroadcastingList and torch.nn._size_N_t alias
        """
        from torch._jit_internal import BroadcastingList2
        from torch.nn.common_types import _size_2_t

        def sum_i(x: _size_2_t) -> int:
            # [FUNCTION DOC] HELPER: Returns the sum of the first two items of the argument.
            return x[0] + x[1]

        def sum_f(x: BroadcastingList2[float]) -> float:
            # [FUNCTION DOC] HELPER: Returns the sum of the first two items of the argument.
            return x[0] + x[1]

        self.assertTrue(torch.jit.script(sum_i)(4) == 8)
        self.assertTrue(torch.jit.script(sum_f)(4.5) == 9.0)

    def test_parse_ir_annotate(self):
        # [FUNCTION DOC] Parses IR text with a constant holding an annotated empty int list,
        # builds a function from the graph and checks calling it returns an empty list.
        ir = """
        graph():
          %3 : int[] = prim::Constant[value=annotate(List[int], [])]()
          return (%3)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == [])

    def test_parse_ir_single_element_tensor_positive(self):
        # [FUNCTION DOC] Parses IR with a long tensor constant given as a single-element
        # value (zero), builds a function and checks the result has one element and is
        # one- dimensional.
        ir = """
        graph():
          %7 : Long(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={0}]()
          return (%7)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret.numel() == 1)
        self.assertTrue(len(ret.size()) == 1)

    def test_parse_ir_single_element_tensor_negative(self):
        # [FUNCTION DOC] Parses IR with a long tensor constant given as a single
        # negative element, builds a function and checks the result has one element and
        # is one- dimensional.
        ir = """
        graph():
          %7 : Long(1, strides=[1], requires_grad=0, device=cpu) = prim::Constant[value={-17}]()
          return (%7)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret.numel() == 1)
        self.assertTrue(len(ret.size()) == 1)

    @skipIfTorchDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_single_inf(self):
        # [FUNCTION DOC] Skipped under Dynamo. Parses IR with a float constant of infinity and
        # checks the function returns positive infinity.
        ir = """
        graph():
          %12 : float = prim::Constant[value=inf]()
          return (%12)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == float("inf"))

    @skipIfTorchDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_single_minus_inf(self):
        # [FUNCTION DOC] Skipped under Dynamo. Parses IR with a float constant of negative
        # infinity and checks the function returns negative infinity.
        ir = """
        graph():
          %12 : float = prim::Constant[value=-inf]()
          return (%12)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == float("-inf"))

    @skipIfTorchDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_bool_true(self):
        # [FUNCTION DOC] Skipped under Dynamo. Parses IR with a bool constant true and checks
        # the function returns true.
        ir = """
        graph():
          %12 : bool = prim::Constant[value=True]()
          return (%12)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == True)  # noqa: E712

    @skipIfTorchDynamo("The test case only test the parser. No need to wrap dynamo.")
    def test_parse_ir_bool_false(self):
        # [FUNCTION DOC] Skipped under Dynamo. Parses IR with a bool constant false and checks
        # the function returns false.
        ir = """
        graph():
          %12 : bool = prim::Constant[value=False]()
          return (%12)
        """
        graph = torch._C.parse_ir(ir, True)
        func = torch._C._create_function_from_graph("forward", graph)
        ret = func()
        self.assertTrue(ret == False)  # noqa: E712

    def test_script_many_decorators(self):
        # [FUNCTION DOC] Checks that a function wrapped in five identity decorators can still be
        # scripted (source is found through the decorators) and gives the same result as eager
        # for an unsqueeze.
        def no_op_decorator(f):
            # [FUNCTION DOC] HELPER: Identity decorator returning the wrapped
            # function unchanged.
            return f

        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        @no_op_decorator
        def foo(x, dim: int):
            # [FUNCTION DOC] HELPER: Returns the input with a new dimension inserted at the
            # given position.
            return x.unsqueeze(dim)

        x = torch.randn(
            1,
        )
        expected = foo(x, 0)
        scripted = torch.jit.script(foo)
        actual = scripted(x, 0)
        torch.testing.assert_close(expected, actual)

    @unittest.skipIf(not RUN_CUDA_HALF, "need CUDA half support")
    def test_pow_multiple_dtype(self):
        # [FUNCTION DOC] Skipped unless CUDA half support exists. Regression test for an issue
        # about power with mixed dtypes: compares eager and scripted runs (the scripted function
        # is run four times to pass profiling) of a half-precision sigmoid raised to a float
        # power on a CUDA tensor.
        # https://github.com/pytorch/pytorch/issues/75476
        def fn(p: torch.Tensor, gamma: float = 2.0) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies sigmoid to the tensor and raises it to the given
            # float power (default two).
            p = torch.sigmoid(p)
            result = p**gamma
            return result

        x = torch.rand((2, 2), dtype=torch.half, device="cuda")

        ref = fn(x)

        script_fn = torch.jit.script(fn)
        for _ in range(4):
            res = script_fn(x)

        self.assertEqual(ref, res)

    def test_jit_get_operation_order(self):
        # [FUNCTION DOC] Verifies that the operator lookup returns aten overloads before
        # TorchScript-added ones. A scalar add of an int and float must not give a complex
        # dtype. In the overload names for the add operator, the first complex entry must come
        # after the first Scalar entry. Prints the override names.
        # See https://github.com/pytorch/pytorch/pull/107138.
        # Depending on order of operator registration, you can get different
        # order of overloads in the JIT operator registry.
        # This is to verify that the order of operators returned by
        # _jit_get_operation always puts aten ops first (i.e. by sorting
        # to put them first)

        # Make sure that this chooses a "scalar" overload not a "complex" overload
        ret = torch.ops.aten.add(4, 3.3)
        self.assertFalse("complex" in str(ret.dtype))

        # "Scalar" overload is a normal aten op; "complex" is added by torchscript.
        # We want "Scalar" to come before "complex".
        op, override_names = torch._C._jit_get_operation("aten::add")
        print(override_names)
        complex_indices = [
            i for i, name in enumerate(override_names) if name == "complex"
        ]
        Scalar_indices = [
            i for i, name in enumerate(override_names) if name == "Scalar"
        ]

        self.assertTrue(len(complex_indices) > 0)
        self.assertTrue(len(Scalar_indices) > 0)
        self.assertTrue(complex_indices[0] > Scalar_indices[0])


if __name__ == "__main__":
    raise_on_run_directly("test/test_jit.py")
