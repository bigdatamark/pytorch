# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo's generic getitem handling, which mirrors CPython's subscript
# dispatch: mapping subscript first, then sequence item access, then class subscript, plus
# explicit __getitem__ method calls. Covers list, tuple, range, torch.Size, dict, defaultdict,
# tensor, namedtuple, typing objects, mappingproxy, nn containers, user-defined containers,
# opaque objects, triton kernels, deque, sequence-protocol iteration and containment,
# list/tuple constructors and Python 3.15 qualified-name error messages, followed by a second
# class testing setitem and delitem. Most tests compile with the eager backend and fullgraph
# and compare against eager results; unsupported cases expect Unsupported. Import time has no
# side effects.
"""Tests for generic_getitem: CPython PyObject_GetItem dispatch in Dynamo.

Tests are organized by dispatch branch:
  - Branch 1 (mp_subscript): list, tuple, range, size, dict, defaultdict, tensor, etc.
  - Branch 2 (sq_item via pysequence_getitem): deque (natural), reversed str/bytes
  - Branch 3 (__class_getitem__): type subscript
  - Explicit __getitem__ dunder calls

See TODO(follow-up) comments on each mp_subscript_impl override for remaining
CPython behavioral gaps.
"""

import collections
import operator
import sys
import types
import typing
import unittest

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch._dynamo.variables.base import VariableTracker
from torch._dynamo.variables.constant import ConstantVariable
from torch._dynamo.variables.lists import BaseListVariable, DequeVariable, RangeVariable
from torch._library.opaque_object import (
    CustomClassBase,
    MemberType,
    register_custom_class,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON, HAS_GPU


class GetItemTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case holding all getitem dispatch tests. No state; each test builds its
    # function locally and uses _compile or a direct compile call, expecting results equal to
    # eager execution or a specific error.
    def _compile(self, fn, *args):
        # [FUNCTION DOC] HELPER: Compile the given function with the eager backend in
        # fullgraph mode, call it with the given positional arguments and return its result.
        # Raises if compilation hits a graph break.
        return torch.compile(fn, backend="eager", fullgraph=True)(*args)

    # --- BaseListVariable (ListVariable) ---

    def test_list_int_index(self):
        # [FUNCTION DOC] Indexing a list with constants 0, 1, 2 inside a compiled function
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a list of three tensors and
            # returns the elements at indices 0, 1 and 2.
            items = [x, x + 1, x + 2]
            return (
                operator.getitem(items, 0),
                operator.getitem(items, 1),
                operator.getitem(items, 2),
            )

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_getitem_compiled_directly(self):
        # [FUNCTION DOC] Compile the getitem operator function itself directly and check it
        # returns the first and last element of a plain list of ints.
        compiled = torch.compile(operator.getitem, backend="eager", fullgraph=True)
        items = [10, 20, 30]
        self.assertEqual(compiled(items, 0), 10)
        self.assertEqual(compiled(items, 2), 30)

    def test_list_slice(self):
        # [FUNCTION DOC] Slicing a list with a full slice, a leading slice and a single-
        # element slice matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a list of three tensors and
            # returns three slices of it.
            items = [x, x + 1, x + 2]
            full = operator.getitem(items, slice(None))
            partial = operator.getitem(items, slice(0, 2))
            single = operator.getitem(items, slice(1, 2))
            return full, partial, single

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_negative_index(self):
        # [FUNCTION DOC] Negative index on a list in a compiled function matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the last element of a three-
            # tensor list using index minus one.
            items = [x, x + 1, x + 2]
            return operator.getitem(items, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_bool_index(self):
        # [FUNCTION DOC] Boolean values used as list indices (False as 0, True as 1)
        # match eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns elements of a three-tensor
            # list selected with False and True.
            items = [x, x + 1, x + 2]
            return (
                operator.getitem(items, False),
                operator.getitem(items, True),
            )

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_invalid_index_type(self):
        # [FUNCTION DOC] Indexing a list with a string must raise Unsupported when compiled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a three-tensor list with a
            # string key, which is invalid.
            items = [x, x + 1, x + 2]
            return operator.getitem(items, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_list_index_via_index_dunder(self):
        # [FUNCTION DOC] A custom object defining __index__ is accepted as a list index and
        # matches eager.
        """Custom __index__ object used as list index — _PyIndex_Check + nb_index_impl."""

        class Idx:
            # [FUNCTION DOC] HELPER: Class whose __index__ method lets instances act as
            # integer indices. This one returns 2.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the integer 2 as the index value.
                return 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a three-tensor list with an
            # instance of the custom index class.
            items = [x, x + 1, x + 2]
            return operator.getitem(items, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- BaseListVariable (TupleVariable) ---

    def test_tuple_int_index(self):
        # [FUNCTION DOC] Indexing a tuple with constant 0 matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the first element of a
            # three-tensor tuple.
            items = (x, x + 1, x + 2)
            return operator.getitem(items, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_negative_index(self):
        # [FUNCTION DOC] Negative index on a tuple matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the last element of a three-
            # tensor tuple.
            items = (x, x + 1, x + 2)
            return operator.getitem(items, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_slice(self):
        # [FUNCTION DOC] Slicing a tuple with full, leading and single-element slices
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a three-tensor tuple and
            # returns three slices of it.
            items = (x, x + 1, x + 2)
            full = operator.getitem(items, slice(None))
            partial = operator.getitem(items, slice(0, 2))
            single = operator.getitem(items, slice(1, 2))
            return full, partial, single

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_bool_index(self):
        # [FUNCTION DOC] A boolean False used as a tuple index matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the first element of a
            # three-tensor tuple selected by False.
            items = (x, x + 1, x + 2)
            return operator.getitem(items, False)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_invalid_index_type(self):
        # [FUNCTION DOC] Indexing a tuple with a string must raise Unsupported when compiled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a three-tensor tuple with a
            # string key, which is invalid.
            items = (x, x + 1, x + 2)
            return operator.getitem(items, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    def test_tuple_index_via_index_dunder(self):
        # [FUNCTION DOC] A custom object with __index__ is accepted as a tuple index and
        # matches eager.
        class Idx:
            # [FUNCTION DOC] HELPER: Class whose __index__ method lets instances act as
            # integer indices. This one returns 2.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the integer 2 as the index value.
                return 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a three-tensor tuple with an
            # instance of the custom index class.
            items = (x, x + 1, x + 2)
            return operator.getitem(items, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- RangeVariable ---

    def test_range_int_index(self):
        # [FUNCTION DOC] Indexing a range object with a positive constant matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the element at index 3 of a
            # stepped range to the input tensor.
            r = range(0, 10, 2)
            return x + operator.getitem(r, 3)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_negative_index(self):
        # [FUNCTION DOC] Negative index on a range object matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the last element of a stepped
            # range to the input tensor.
            r = range(0, 10, 2)
            return x + operator.getitem(r, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_slice(self):
        # [FUNCTION DOC] Slicing a range yields a range whose first element is usable;
        # compiled result matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Slices a stepped range and adds the
            # first element of the slice to the input tensor.
            r = range(0, 10, 2)
            result = operator.getitem(r, slice(1, 3))
            return x + result[0]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_bool_index(self):
        # [FUNCTION DOC] A boolean True used as a range index matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the element at position
            # selected by True from a stepped range to the input.
            r = range(0, 10, 2)
            return x + operator.getitem(r, True)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_index_via_index_dunder(self):
        # [FUNCTION DOC] A custom __index__ object is accepted as a range index and
        # matches eager.
        class Idx:
            # [FUNCTION DOC] HELPER: Class whose __index__ method lets instances act as
            # integer indices. This one returns 2.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the integer 2 as the index value.
                return 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the range element selected by
            # an instance of the custom index class to the input.
            r = range(0, 10, 2)
            return x + operator.getitem(r, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_range_invalid_index_type(self):
        # [FUNCTION DOC] Indexing a range with a string must raise Unsupported when compiled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a stepped range with a
            # string key, which is invalid.
            r = range(0, 10, 2)
            return x + operator.getitem(r, "a")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- SizeVariable ---

    def test_size_int_index(self):
        # [FUNCTION DOC] Integer index on a tensor's size object matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the first dimension size of the
            # input tensor to the tensor.
            s = x.size()
            return x + operator.getitem(s, 0)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_negative_index(self):
        # [FUNCTION DOC] Negative index on a size object matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the last dimension size of the
            # input tensor to the tensor.
            s = x.size()
            return x + operator.getitem(s, -1)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_slice(self):
        # [FUNCTION DOC] Slicing a size object gives a size whose first element is usable;
        # compiled result matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Slices the size of the input to its
            # first dimension and adds that value to the input.
            s = x.size()
            result = operator.getitem(s, slice(0, 1))
            return x + result[0]

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_bool_index(self):
        # [FUNCTION DOC] A boolean False used as a size index matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the dimension size selected by
            # False to the input.
            s = x.size()
            return x + operator.getitem(s, False)

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_index_via_index_dunder(self):
        # [FUNCTION DOC] A custom __index__ object is accepted as a size index and
        # matches eager.
        class Idx:
            # [FUNCTION DOC] HELPER: Class whose __index__ method lets instances act as
            # integer indices. This one returns 1.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the integer 1 as the index value.
                return 1

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds the dimension size selected by
            # the custom index object to the input.
            s = x.size()
            return x + operator.getitem(s, Idx())

        x = torch.randn(4, 8)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_size_invalid_index_type(self):
        # [FUNCTION DOC] Indexing a size object with a string must raise Unsupported
        # when compiled.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes the size of the input with a
            # string key, which is invalid.
            s = x.size()
            return x + operator.getitem(s, "a")

        x = torch.randn(4, 8)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- ConstDictVariable ---

    def test_dict_str_key(self):
        # [FUNCTION DOC] String key lookup in a dict built inside the compiled function
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a dict with two tensor values
            # and returns the value for key a.
            d = {"a": x, "b": x + 1}
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_int_key(self):
        # [FUNCTION DOC] Integer key lookup in a dict matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a dict keyed by 0 and 1 and
            # returns the value for key 1.
            d = {0: x, 1: x + 1}
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_missing_key(self):
        # [FUNCTION DOC] Looking up a missing key in a dict during tracing must
        # raise Unsupported.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Looks up a key that is absent from a
            # one-entry dict.
            d = {"a": x}
            return operator.getitem(d, "missing")

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- DefaultDictVariable ---

    def test_defaultdict_existing_key(self):
        # [FUNCTION DOC] Lookup of an existing key in a defaultdict matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a defaultdict whose factory
            # adds 99 to the input, stores a value under key a and returns it.
            d = collections.defaultdict(lambda: x + 99)
            d["a"] = x
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_defaultdict_missing_key(self):
        # [FUNCTION DOC] Looking up a missing key in a defaultdict (which inserts a default
        # list) works and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a defaultdict of lists,
            # looks up an absent key purely for its side effect and returns the input.
            d = collections.defaultdict(list)
            operator.getitem(d, "new")
            return x

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TensorVariable ---

    def test_tensor_int_index(self):
        # [FUNCTION DOC] Integer index of a tensor matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns row 0 of the input tensor.
            return operator.getitem(x, 0)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_slice(self):
        # [FUNCTION DOC] Slice index of a tensor matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the first two rows of the
            # input tensor.
            return operator.getitem(x, slice(0, 2))

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_tuple_index(self):
        # [FUNCTION DOC] Tuple index combining an integer and a slice on a tensor
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes the input with a tuple of an
            # integer and a slice.
            return operator.getitem(x, (0, slice(1, 3)))

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_getitem_torch_function_mode(self):
        # [FUNCTION DOC] A TorchFunctionMode that adds one to the result of tensor getitem
        # must be honored when compiling with the eager backend (no fullgraph); compiled
        # output equals eager.
        """TorchFunctionMode intercepts tensor __getitem__ and can modify behavior."""

        class AddOneMode(torch.overrides.TorchFunctionMode):
            # [CLASS DOC] HELPER: TorchFunctionMode that runs every intercepted function and
            # adds one to the result when the function is the tensor getitem.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Run the intercepted function with its arguments; if
                # it is the tensor getitem method, return the result plus one, otherwise the
                # result unchanged.
                result = func(*args, **(kwargs or {}))
                if func is torch.Tensor.__getitem__:
                    return result + 1
                return result

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Enters the mode and returns row 0 of
            # the input tensor.
            with AddOneMode():
                return operator.getitem(x, 0)

        x = torch.randn(4, 4)
        expected = fn(x)
        compiled = torch.compile(fn, backend="eager")(x)
        self.assertEqual(expected, compiled)

    def test_tensor_getitem_torch_function_subclass(self):
        # [FUNCTION DOC] A tensor subclass whose __torch_function__ doubles the result of
        # getitem must be honored when compiled; compiled output equals eager.
        """Tensor subclass with __torch_function__ intercepts __getitem__."""

        class ScaledTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose __torch_function__ multiplies the
            # result by two when the intercepted function is the tensor getitem method.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Delegate to the base implementation, then double the
                # result if the function is the tensor getitem method; otherwise return the
                # base result.
                result = super().__torch_function__(func, types, args, kwargs or {})
                if func is torch.Tensor.__getitem__:
                    return result * 2
                return result

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns row 0 of the input tensor.
            return operator.getitem(x, 0)

        x = ScaledTensor(torch.randn(4, 4))
        expected = fn(x)
        compiled = torch.compile(fn, backend="eager")(x)
        self.assertEqual(expected, compiled)

    def test_tensor_negative_index(self):
        # [FUNCTION DOC] Negative integer index on a tensor matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the last row of the
            # input tensor.
            return operator.getitem(x, -1)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_bool_index(self):
        # [FUNCTION DOC] A boolean True index on a tensor (adds a leading dimension)
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes the input tensor with True.
            return operator.getitem(x, True)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tensor_ellipsis_index(self):
        # [FUNCTION DOC] Ellipsis index on a tensor matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes the input tensor with the
            # Ellipsis object.
            return operator.getitem(x, ...)

        x = torch.randn(4, 4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- NamedTupleVariable (via UserDefinedTupleVariable) ---

    def test_namedtuple_int_index(self):
        # [FUNCTION DOC] Integer index on the namedtuple result of the topk function
        # matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Takes the top two values of a vector
            # and returns element 1 (the indices) of the result.
            result = torch.topk(x, 2)
            return operator.getitem(result, 1)

        x = torch.randn(10)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_namedtuple_values_index(self):
        # [FUNCTION DOC] Index 0 on the namedtuple result of topk (the values) matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Takes the top two values of a vector
            # and returns element 0 of the result.
            result = torch.topk(x, 2)
            return operator.getitem(result, 0)

        x = torch.randn(10)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- TypingVariable ---

    def test_typing_subscript(self):
        # [FUNCTION DOC] Subscripting the builtin list type with int inside a compiled
        # function does not break the graph; result matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a parameterized list type
            # (unused) and returns the input plus one.
            t = list[int]  # noqa: F841
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_typing_subscript_dict(self):
        # [FUNCTION DOC] Subscripting typing.Dict via the getitem operator with constant types
        # compiles and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Subscripts typing.Dict with a tuple
            # of two types, discarding the result, and returns the input plus one.
            operator.getitem(typing.Dict, (str, int))  # noqa: UP006
            return x + 1

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_typing_subscript_non_constant(self):
        # [FUNCTION DOC] Subscripting a type with a traced tensor value (non-constant) must
        # raise Unsupported.
        """Subscripting a typing construct with a traced (non-constant) value should graph break."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Subscripts the list type with the
            # input tensor.
            return operator.getitem(list, x)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- MappingProxyVariable ---

    def test_mappingproxy_getitem(self):
        # [FUNCTION DOC] Lookup in a mappingproxy wrapping a dict matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Wraps a dict in a read-only mapping
            # proxy and adds its value for key a to the input.
            d = {"a": 1, "b": 2}
            proxy = types.MappingProxyType(d)
            return x + operator.getitem(proxy, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- NNModuleVariable (ModuleList) ---

    def test_nn_module_list_int_index(self):
        # [FUNCTION DOC] Integer index of a ModuleList attribute inside forward compiles with
        # fullgraph and matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleList of three Linear layers.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its ModuleList of three 4x4
                # Linear layers.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(3)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Fetch layer 1 of the ModuleList through the getitem
                # operator and apply it to the input.
                return operator.getitem(self.layers, 1)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- NNModuleVariable (ModuleDict) ---

    def test_nn_module_dict_str_key(self):
        # [FUNCTION DOC] String key lookup in a ModuleDict attribute inside forward compiles
        # with fullgraph and matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleDict with one Linear layer named fc.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its ModuleDict with a 4x4
                # Linear layer under fc.
                super().__init__()
                self.layers = torch.nn.ModuleDict({"fc": torch.nn.Linear(4, 4)})

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Fetch the layer under key fc through the getitem
                # operator and apply it to the input.
                return operator.getitem(self.layers, "fc")(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    # --- NNModuleVariable (Sequential) ---

    def test_nn_sequential_int_index(self):
        # [FUNCTION DOC] Integer index of a Sequential attribute compiles with fullgraph and
        # matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Sequential of a Linear layer followed
            # by ReLU.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its Sequential container.
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Fetch element 0 of the Sequential through the getitem
                # operator and apply it to the input.
                return operator.getitem(self.seq, 0)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    def test_nn_module_list_negative_index(self):
        # [FUNCTION DOC] Negative index of a ModuleList attribute compiles with fullgraph and
        # matches eager.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleList of three Linear layers.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its ModuleList of three 4x4
                # Linear layers.
                super().__init__()
                self.layers = torch.nn.ModuleList(
                    [torch.nn.Linear(4, 4) for _ in range(3)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Fetch the last layer of the ModuleList by negative
                # index and apply it to the input.
                return operator.getitem(self.layers, -1)(x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager", fullgraph=True)
        self.assertEqual(model(x), compiled(x))

    def test_nn_sequential_slice(self):
        # [FUNCTION DOC] Slicing a Sequential in forward gives a sub-Sequential that is
        # applied; compiled without fullgraph because slicing an nn module triggers a restart
        # of analysis. Checks results equal eager and at least one frame was compiled with a
        # CompileCounter backend.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a Sequential of Linear, ReLU and Linear.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its three-
                # element Sequential.
                super().__init__()
                self.seq = torch.nn.Sequential(
                    torch.nn.Linear(4, 4),
                    torch.nn.ReLU(),
                    torch.nn.Linear(4, 4),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Slice the Sequential to its first two elements
                # through the getitem operator and apply the sub-Sequential to the input.
                sub = operator.getitem(self.seq, slice(0, 2))
                return sub(x)

        model = Model()
        x = torch.randn(4)
        # Slice on nn.Module triggers convert_to_unspecialized → RestartAnalysis,
        # so don't use fullgraph=True.
        cnt = torch._dynamo.testing.CompileCounter()
        compiled = torch.compile(model, backend=cnt)
        self.assertEqual(model(x), compiled(x))
        self.assertGreaterEqual(cnt.frame_count, 1)

    def test_nn_module_dict_missing_key(self):
        # [FUNCTION DOC] A missing key in a ModuleDict raises KeyError while tracing, which
        # surfaces as Unsupported under fullgraph compile.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module holding a ModuleDict with one Linear layer named fc.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its ModuleDict.
                super().__init__()
                self.layers = torch.nn.ModuleDict({"fc": torch.nn.Linear(4, 4)})

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Fetch a nonexistent key from the ModuleDict and apply
                # the result to the input.
                return operator.getitem(self.layers, "nonexistent")(x)

        model = Model()
        x = torch.randn(4)
        # ModuleDict["nonexistent"] raises KeyError at trace time, which
        # surfaces as Unsupported (observed exception, no handler in compiled fn).
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            torch.compile(model, backend="eager", fullgraph=True)(x)

    # --- UserDefinedObjectVariable ---

    def test_user_defined_object_getitem(self):
        # [FUNCTION DOC] A user class with a __getitem__ method that indexes its item list is
        # indexable inside a compiled function.
        class Container:
            # [CLASS DOC] HELPER: Plain container that stores a list and forwards subscripting
            # to it.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the given items list on the instance.
                self.items = items

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored item at the given key.
                return self.items[key]

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates the container with two
            # tensors and returns element 0.
            c = Container([x, x + 1])
            return operator.getitem(c, 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_object_without_getitem(self):
        # [FUNCTION DOC] Indexing an object whose class has no __getitem__ must raise
        # Unsupported when compiled.
        class NoGetItem:
            # [CLASS DOC] HELPER: Empty class without subscript support.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates an instance and indexes it
            # with 0.
            obj = NoGetItem()
            return operator.getitem(obj, 0)

        x = torch.randn(4)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            self._compile(fn, x)

    # --- UserDefinedListVariable ---

    def test_user_defined_list_getitem(self):
        # [FUNCTION DOC] Indexing an instance of a plain list subclass matches eager.
        class MyList(list):
            # [CLASS DOC] HELPER: Empty list subclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds the list subclass with three
            # tensors and returns element 1.
            items = MyList([x, x + 1, x + 2])
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefinedTupleVariable ---

    def test_user_defined_tuple_getitem(self):
        # [FUNCTION DOC] Indexing an instance of a plain tuple subclass matches eager.
        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Empty tuple subclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds the tuple subclass with three
            # tensors and returns element 1.
            items = MyTuple((x, x + 1, x + 2))
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefinedDictVariable ---

    def test_user_defined_dict_getitem(self):
        # [FUNCTION DOC] Indexing an instance of a plain dict subclass matches eager.
        class MyDict(dict):
            # [CLASS DOC] HELPER: Empty dict subclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds the dict subclass from
            # keywords a and b and returns the value for key a.
            d = MyDict(a=x, b=x + 1)
            return operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_missing(self):
        # [FUNCTION DOC] A dict subclass defining __missing__ returns the fallback value for
        # an absent key; matches eager.
        class MyDict(dict):
            # [CLASS DOC] HELPER: Dict subclass whose __missing__ always returns 42.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Return the constant 42 for any missing key.
                return 42

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates the subclass with key a only
            # and adds the value looked up for absent key b to the input.
            d = MyDict(a=1)
            return x + operator.getitem(d, "b")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_custom_missing(self):
        # [FUNCTION DOC] A dict subclass whose __missing__ stores the new key (value equal to
        # the current size) and returns it; matches eager.
        class DefaultDict(dict):
            # [CLASS DOC] HELPER: Dict subclass whose __missing__ inserts the key mapped to
            # the current number of entries and returns that value.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Insert the missing key with the current dict length
                # as value and return it.
                self[key] = len(self)
                return self[key]

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Fills one key, looks up absent key b
            # triggering __missing__, and adds the value to the input.
            d = DefaultDict()
            d["a"] = 1
            val = operator.getitem(d, "b")
            return x + val

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_collections_counter_getitem(self):
        # [FUNCTION DOC] Lookup in collections.Counter for an existing key matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a Counter from a mapping and
            # adds the count of key a to the input.
            c = collections.Counter({"a": 1, "b": 2})
            return x + operator.getitem(c, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- UserDefined* with overridden __getitem__ ---

    def test_user_defined_dict_overridden_getitem(self):
        # [FUNCTION DOC] A dict subclass overriding __getitem__ must have that override used
        # (adds 100) rather than the base dict lookup; matches eager.
        """Dict subclass with custom __getitem__ should NOT delegate to _base_vt."""

        class MyDict(dict):
            # [CLASS DOC] HELPER: Dict subclass whose __getitem__ adds 100 to the base
            # lookup result.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the base dict lookup result plus 100.
                return super().__getitem__(key) + 100

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates the subclass with keys a and
            # b and adds the value looked up for a to the input.
            d = MyDict(a=1, b=2)
            return x + operator.getitem(d, "a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_list_overridden_getitem(self):
        # [FUNCTION DOC] A list subclass overriding __getitem__ must have the override used
        # (doubles the result); matches eager.
        """List subclass with custom __getitem__ should NOT delegate to _base_vt."""

        class MyList(list):
            # [CLASS DOC] HELPER: List subclass whose __getitem__ doubles the base
            # lookup result.
            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the base list lookup result times two.
                return super().__getitem__(key) * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds the subclass with three
            # tensors and returns element 1.
            items = MyList([x, x + 1, x + 2])
            return operator.getitem(items, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_counter_missing_key(self):
        # [FUNCTION DOC] Counter returns zero for a missing key via its __missing__ method;
        # matches eager.
        """Counter.__missing__ returns 0 for missing keys."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a Counter and adds the count
            # for an absent key to the input.
            c = collections.Counter({"a": 1, "b": 2})
            return x + operator.getitem(c, "missing")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_getattr_dict_subscript(self):
        # [FUNCTION DOC] Subscripting an object's __dict__ attribute (to fetch a submodule
        # from _modules) compiles without fullgraph and matches eager.
        """obj.__dict__["key"] → GetAttrVariable → DunderDictVariable."""

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with one Linear layer that calls it through the
            # instance dict and modules mapping.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module and its 4x4 Linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Look up the linear layer through the instance
                # __dict__ and its modules mapping and apply it to the input.
                return self.__dict__["_modules"]["linear"](x)

        model = Model()
        x = torch.randn(4)
        compiled = torch.compile(model, backend="eager")
        self.assertEqual(model(x), compiled(x))

    # --- CustomClassObjectVariable ---

    def test_opaque_object_getitem(self):
        # [FUNCTION DOC] Subscripting a registered opaque custom class object (members
        # registered as inlined or real-value) works under fullgraph and matches eager.
        # Registers two classes with the custom class registry as a side effect.
        class OpaqueScaler(CustomClassBase):
            # [CLASS DOC] HELPER: Opaque custom class holding a scale value.
            def __init__(self, scale):
                # [FUNCTION DOC] HELPER: Store the scale value.
                self.scale = scale

            def apply(self, x):
                # [FUNCTION DOC] HELPER: Multiply the argument by the stored scale and
                # return it.
                return x * self.scale

        class OpaqueContainer(CustomClassBase):
            # [CLASS DOC] HELPER: Opaque custom class holding a list of items and
            # supporting subscripting.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the items list.
                self.items = items

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: Return the stored item at the given index.
                return self.items[idx]

        register_custom_class(
            OpaqueScaler,
            typ="symbolic",
            members={
                "scale": MemberType.USE_REAL,
                "apply": MemberType.INLINED,
            },
        )
        register_custom_class(
            OpaqueContainer,
            typ="symbolic",
            members={
                "items": MemberType.USE_REAL,
                "__getitem__": MemberType.INLINED,
            },
        )

        def fn(x, c):
            # [FUNCTION DOC] HELPER: Function under test. Fetches element 0 from the opaque
            # container and applies it to the input.
            scaler = operator.getitem(c, 0)
            return scaler.apply(x)

        x = torch.randn(4)
        c = OpaqueContainer([OpaqueScaler(2.0), OpaqueScaler(3.0)])
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, c), compiled(x, c))

    # --- TritonKernelVariable ---

    @unittest.skipUnless(HAS_GPU and HAS_CUDA_AND_TRITON, "requires gpu and triton")
    def test_triton_kernel_getitem_grid(self):
        # [FUNCTION DOC] Subscripting a triton kernel with a launch grid gives a launcher;
        # when called, results under fullgraph match eager. Skipped without a GPU and triton.
        from torch.testing._internal.triton_utils import add_kernel

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Function under test. Allocates a zero output, builds a
            # one-element grid, subscripts the add kernel with it, launches it with block size
            # 256 and returns the output.
            output = torch.zeros_like(x)
            n_elements = output.numel()
            grid = (n_elements // 256,)
            bound = operator.getitem(add_kernel, grid)
            bound(x, y, output, n_elements, BLOCK_SIZE=256)
            return output

        x = torch.randn(256, device="cuda")
        y = torch.randn(256, device="cuda")
        compiled = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x, y), compiled(x, y))

    # ===================================================================
    # CPython behavioral gaps — expectedFailure until implemented
    # ===================================================================

    # GAP 2: dict_subscript calls _PyObject_HashFast → TypeError for unhashable keys.
    # TODO: ConstDictVariable.mp_subscript_impl should check tp_hash and raise TypeError
    # for unhashable keys, matching CPython's dict_subscript (Objects/dictobject.c:3680).
    @unittest.expectedFailure
    def test_dict_unhashable_key(self):
        # [FUNCTION DOC] EXPECTED FAILURE: indexing a dict with an unhashable list key should
        # raise TypeError when compiled, but Dynamo does not yet.
        # KNOWN ISSUE: Dynamo does not raise the TypeError CPython raises for unhashable keys;
        # the test is marked expectedFailure.
        """dict[unhashable] should raise TypeError, not KeyError or silent failure."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a dict and indexes it with a
            # list key.
            d = {0: x, 1: x + 1}
            return operator.getitem(d, [0])

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            self._compile(fn, x)

    # ===================================================================
    # Branch 1 (mp_subscript): str/bytes
    # ConstantVariable.mp_subscript_impl for str/bytes.
    # CPython: Objects/unicodeobject.c:13809 (unicode_subscript)
    # CPython: Objects/bytesobject.c (bytes_subscript)
    # ===================================================================

    def test_str_subscript(self):
        # [FUNCTION DOC] Indexing a string constant matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Takes the first character of a
            # string and adds its length to the input.
            s = "hello"
            c = operator.getitem(s, 0)
            return x + len(c)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_bytes_subscript(self):
        # [FUNCTION DOC] Indexing a bytes constant yields an integer and matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Takes the first byte of a bytes
            # object and adds it to the input.
            b = b"hello"
            val = operator.getitem(b, 0)
            return x + val

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # ===================================================================
    # Explicit __getitem__ dunder call path tests
    # Exercises: obj.__getitem__(key) → LOAD_ATTR + CALL, which may
    # route through call_method → mp_subscript_impl rather than
    # generic_getitem → mp_subscript_impl.
    # ===================================================================

    def test_list_dunder_getitem(self):
        # [FUNCTION DOC] Calling __getitem__ explicitly on a list matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Calls the list __getitem__ method
            # with 1.
            items = [x, x + 1, x + 2]
            return items.__getitem__(1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_dict_dunder_getitem(self):
        # [FUNCTION DOC] Calling __getitem__ explicitly on a dict matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Calls the dict __getitem__ method
            # with key a.
            d = {"a": x, "b": x + 1}
            return d.__getitem__("a")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_user_defined_dict_missing_dunder_getitem(self):
        # [FUNCTION DOC] A dict subclass with __missing__ gives the fallback when __getitem__
        # is called as a method.
        """__missing__ fallback must work via __getitem__ method call, not just operator.getitem."""

        class MyDict(dict):
            # [CLASS DOC] HELPER: Dict subclass whose __missing__ always returns 42.
            def __missing__(self, key):
                # [FUNCTION DOC] HELPER: Return the constant 42 for any missing key.
                return 42

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Calls __getitem__ with absent key b
            # on the subclass and adds the result to the input.
            d = MyDict(a=1)
            return x + d.__getitem__("b")

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # ===================================================================
    # Branch 3: __class_getitem__ (type subscript)
    # Exercises: MyClass[int] → type.__getitem__ → __class_getitem__
    # In Python 3.10+, type.__getitem__ sets mp_subscript on type objects,
    # so this goes through Branch 1 of generic_getitem.
    # ===================================================================

    def test_class_getitem_builtin(self):
        # [FUNCTION DOC] Subscripting the builtin list type yields a generic alias whose
        # arguments can be read; matches eager.
        """list[int] → GenericAlias via type.__getitem__."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates the alias for list of int
            # and adds the number of alias arguments to the input.
            alias = list[int]
            return x + len(alias.__args__)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_class_getitem_custom(self):
        # [FUNCTION DOC] A class with __class_getitem__ returning its argument can be
        # subscripted in compiled code; matches eager.
        """Custom class with __class_getitem__."""

        class MyGeneric:
            # [CLASS DOC] HELPER: Class whose __class_getitem__ returns the subscript
            # argument unchanged.
            def __class_getitem__(cls, item):
                # [FUNCTION DOC] HELPER: Return the given subscript item.
                return item

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Subscripts the class with int and
            # adds one to the input if the result is int, else zero.
            result = MyGeneric[int]
            return x + (1 if result is int else 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_not_subscriptable(self):
        # [FUNCTION DOC] Subscripting an int constant raises TypeError at run time with the
        # eager backend (not fullgraph).
        """Non-subscriptable object should raise TypeError."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes the int 42 with 0.
            return operator.getitem(42, 0)

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(x)

    # ===================================================================
    # Branch 2: sq_item via pysequence_getitem
    #
    # CPython's PyObject_GetItem branch 2: types with sq_item but no
    # mp_subscript go through _PyIndex_Check → PySequence_GetItem → sq_item.
    #
    # Sub-sections:
    #   (a) Natural dispatch — deque (only sq_item, no mp_subscript)
    #   (b) reversed() → sq_item — str/bytes lack __reversed__, so
    #       reversed() falls back to pysequence_getitem naturally
    # ===================================================================

    # --- (a) Natural dispatch: deque ---
    # CPython's deque only has sq_item (Modules/_collectionsmodule.c:1888),
    # not mp_subscript. generic_getitem dispatches to sq_item_impl directly.

    def test_deque_int_index(self):
        # [FUNCTION DOC] Positive integer index on a deque matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns element 1 of a three-
            # tensor deque.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, 1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_negative_index(self):
        # [FUNCTION DOC] Negative index on a deque (sequence-style item access) matches eager.
        """pysequence_getitem wraps negative indices via sq_length before sq_item."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns the last element of a three-
            # tensor deque.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, -1)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_bool_index(self):
        # [FUNCTION DOC] A boolean True index on a deque matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns element 1 of a deque
            # selected by True.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, True)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_index_via_index_dunder(self):
        # [FUNCTION DOC] A custom __index__ object works as a deque index; matches eager.
        class Idx:
            # [FUNCTION DOC] HELPER: Class whose __index__ method lets instances act as
            # integer indices. This one returns 2.
            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the integer 2 as the index value.
                return 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a three-tensor deque with
            # the custom index object.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, Idx())

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_deque_slice_should_reject(self):
        # [FUNCTION DOC] Slicing a deque must raise TypeError when compiled with the
        # eager backend.
        """deque[0:2] → TypeError — sq_item does not accept slices."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Slices a deque with 0 to 2.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, slice(0, 2))

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(x)

    def test_deque_invalid_index_type(self):
        # [FUNCTION DOC] Indexing a deque with a string must raise TypeError when compiled
        # with the eager backend.
        """deque['a'] → TypeError, matching CPython's branch 2 _PyIndex_Check."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a deque with a string.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, "a")

        x = torch.randn(4)
        with self.assertRaises(TypeError):
            torch.compile(fn, backend="eager")(x)

    def test_deque_out_of_range(self):
        # [FUNCTION DOC] An out-of-range deque index must raise IndexError when compiled with
        # the eager backend.
        """deque[100] → IndexError('deque index out of range')."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Indexes a three-element deque
            # with 100.
            d = collections.deque([x, x + 1, x + 2])
            return operator.getitem(d, 100)

        x = torch.randn(4)
        with self.assertRaises(IndexError):
            torch.compile(fn, backend="eager")(x)

    def test_deque_reversed(self):
        # [FUNCTION DOC] The reversed builtin on a deque works and yields the last element
        # first; matches eager.
        """reversed(deque) uses __reversed__ (deque has it since 3.8)."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a list from reversed
            # iteration of a deque and returns its first element.
            d = collections.deque([x, x + 1, x + 2])
            items = list(reversed(d))
            return items[0]

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- (b) iter/list/tuple/reversed on deque ---
    # deque has __iter__, so these go through the normal iteration path
    # (unpack_var_sequence), not the sq_item fallback in builtin.py.

    def test_iter_deque(self):
        # [FUNCTION DOC] Using iter on a deque and building a list from it matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns a list made from an iterator
            # over a three-tensor deque.
            d = collections.deque([x, x + 1, x + 2])
            return list(iter(d))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_deque(self):
        # [FUNCTION DOC] Building a list from a deque matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns a list made from a three-
            # tensor deque.
            d = collections.deque([x, x + 1, x + 2])
            return list(d)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_deque(self):
        # [FUNCTION DOC] Building a tuple from a deque matches eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Returns a tuple made from a three-
            # tensor deque.
            d = collections.deque([x, x + 1, x + 2])
            return tuple(d)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- (d) generic_getiter sequence protocol fallback ---
    # User-defined class with __getitem__ + __len__ but no __iter__:
    # CPython gives it both mp_subscript and sq_item via slot wrappers.
    # generic_getiter (#178462) detects no tp_iter, falls back to
    # pysequence_check → sequence_iterator polyfill → __getitem__.

    def test_iter_seqonly(self):
        # [FUNCTION DOC] iter over a class with only __getitem__ and __len__ uses the sequence
        # protocol; the loop sums the elements; matches eager.
        """iter(SeqOnly) → generic_getiter → sequence_iterator → __getitem__."""

        class SeqOnly:
            # [CLASS DOC] HELPER: Class with only __getitem__ (delegating to a list) and
            # __len__, no __iter__, so iteration falls back to the sequence protocol.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the given items list.
                self.items = items

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the stored item at index i.
                return self.items[i]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of stored items.
                return len(self.items)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Iterates over the sequence-only
            # object in a loop, sums elements and adds the total to the input.
            s = SeqOnly([1, 2, 3])
            total = 0
            for item in s:
                total += item
            return x + total

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_seqonly(self):
        # [FUNCTION DOC] Building a list from a sequence-protocol-only object works;
        # matches eager.
        """list(SeqOnly) → generic_getiter → sequence_iterator → __getitem__."""

        class SeqOnly:
            # [CLASS DOC] HELPER: Class with only __getitem__ (delegating to a list) and
            # __len__, no __iter__, so iteration falls back to the sequence protocol.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the given items list.
                self.items = items

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the stored item at index i.
                return self.items[i]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of stored items.
                return len(self.items)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a list from the sequence-only
            # object and adds the sum of its items to the input.
            s = SeqOnly([1, 2, 3])
            return x + sum(list(s))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_list_constructor_rejects_next_only(self):
        # [FUNCTION DOC] list of an object that only defines __next__ must raise TypeError
        # saying the object is not iterable; the test compares the compiled flag to eager.
        class IterNextOnly:
            # [CLASS DOC] HELPER: Class that defines __next__ over a stored list but
            # no __iter__.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the items and set the read position to zero.
                self.items = items
                self.index = 0

            def __next__(self):
                # [FUNCTION DOC] HELPER: Return the next item or raise StopIteration at the
                # end, advancing the position.
                if self.index >= len(self.items):
                    raise StopIteration
                item = self.items[self.index]
                self.index += 1
                return item

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a list from the
            # object and returns True if a TypeError mentioning not iterable is raised,
            # else False.
            try:
                list(IterNextOnly([1, 2, 3]))
            except TypeError as exc:
                return "object is not iterable" in str(exc)
            return False

        self.assertEqual(fn(), self._compile(fn))

    def test_list_constructor_rejects_iter_without_next(self):
        # [FUNCTION DOC] list of an object whose __iter__ returns itself but which has no
        # __next__ raises TypeError; compiled message string must equal eager and be a str.
        class IterNoNext:
            # [CLASS DOC] HELPER: Class whose __iter__ returns itself although it defines
            # no __next__.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a list from the
            # object and returns the TypeError message text.
            try:
                list(IterNoNext())
            except TypeError as exc:
                return str(exc)

        out = self._compile(fn)
        self.assertTrue(type(out) is str)
        self.assertEqual(fn(), out)

    def test_list_constructor_rejects_keywords(self):
        # [FUNCTION DOC] Calling list with an unsupported keyword argument raises TypeError
        # mentioning no keyword arguments; compiled flag equals eager.
        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Calls list with an unsupported
            # keyword and returns True if the TypeError message mentions that list takes no
            # keyword arguments, else False.
            try:
                list(unsupported_arg=[])
            except TypeError as exc:
                return "list() takes no keyword" in str(exc)
            return False

        self.assertEqual(fn(), self._compile(fn))

    def test_list_constructor_rejects_too_many_args(self):
        # [FUNCTION DOC] Calling list with two positional arguments raises TypeError about at
        # most one argument; compiled flag equals eager.
        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Calls list with two lists and
            # returns True if the TypeError message mentions expected at most one argument,
            # else False.
            try:
                list([], [])
            except TypeError as exc:
                return "expected at most 1 argument" in str(exc)
            return False

        self.assertEqual(fn(), self._compile(fn))

    def test_tuple_constructor_preserves_exact_tuple_identity(self):
        # [FUNCTION DOC] Calling tuple on an exact tuple returns the same object; compiled
        # result equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a tuple, calls tuple on it
            # and returns whether the result is the identical object.
            items = (x, x + 1)
            return tuple(items) is items

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_tuple_constructor_converts_torch_size_to_tuple(self):
        # [FUNCTION DOC] Calling tuple on a torch.Size gives a plain tuple of the same
        # contents that is a new object; compiled result equals eager.
        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Converts a Size of 2 and 3 to a
            # tuple and returns whether the type is exactly tuple, the contents are 2 and 3,
            # and the result is not the Size itself.
            size = torch.Size([2, 3])
            result = tuple(size)
            return type(result) is tuple and result == (2, 3) and result is not size

        self.assertEqual(fn(), self._compile(fn))

    def test_tuple_constructor_rejects_next_only(self):
        # [FUNCTION DOC] tuple of an object that only defines __next__ raises TypeError about
        # not iterable; compiled flag equals eager.
        class IterNextOnly:
            # [CLASS DOC] HELPER: Class that defines __next__ over a stored list but
            # no __iter__.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the items and set the read position to zero.
                self.items = items
                self.index = 0

            def __next__(self):
                # [FUNCTION DOC] HELPER: Return the next item or raise StopIteration at the
                # end, advancing the position.
                if self.index >= len(self.items):
                    raise StopIteration
                item = self.items[self.index]
                self.index += 1
                return item

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a tuple from the
            # object and returns True if a TypeError mentioning not iterable is raised,
            # else False.
            try:
                tuple(IterNextOnly([1, 2, 3]))
            except TypeError as exc:
                return "object is not iterable" in str(exc)
            return False

        self.assertEqual(fn(), self._compile(fn))

    def test_tuple_constructor_rejects_iter_without_next(self):
        # [FUNCTION DOC] tuple of an object whose __iter__ returns itself but lacks __next__
        # raises TypeError; compiled message is a str equal to eager.
        class IterNoNext:
            # [CLASS DOC] HELPER: Class whose __iter__ returns itself although it defines
            # no __next__.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a tuple from the
            # object and returns the TypeError message text, or False if no error occurs.
            try:
                tuple(IterNoNext())
            except TypeError as exc:
                return str(exc)
            return False

        out = self._compile(fn)
        self.assertTrue(type(out) is str)
        self.assertEqual(fn(), out)

    def test_tuple_constructor_rejects_keywords(self):
        # [FUNCTION DOC] Calling tuple with an unsupported keyword raises TypeError mentioning
        # no keyword arguments; compiled flag equals eager.
        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Calls tuple with an unsupported
            # keyword and returns True if the message says tuple takes no keyword arguments,
            # else False.
            try:
                tuple(unsupported_arg=[])
            except TypeError as exc:
                return "tuple() takes no keyword" in str(exc)
            return False

        self.assertEqual(fn(), self._compile(fn))

    def test_tuple_constructor_rejects_too_many_args(self):
        # [FUNCTION DOC] Calling tuple with two positional arguments raises TypeError about at
        # most one argument; compiled flag equals eager.
        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Calls tuple with two lists and
            # returns True if the message mentions expected at most one argument, else False.
            try:
                tuple([], [])
            except TypeError as exc:
                return "expected at most 1 argument" in str(exc)
            return False

        self.assertEqual(fn(), self._compile(fn))

    def test_tuple_seqonly(self):
        # [FUNCTION DOC] Building a tuple from a sequence-protocol-only object works;
        # matches eager.
        """tuple(SeqOnly) → generic_getiter → sequence_iterator → __getitem__."""

        class SeqOnly:
            # [CLASS DOC] HELPER: Class with only __getitem__ (delegating to a list) and
            # __len__, no __iter__, so iteration falls back to the sequence protocol.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the given items list.
                self.items = items

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the stored item at index i.
                return self.items[i]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of stored items.
                return len(self.items)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a tuple from the sequence-
            # only object and adds the sum of its items to the input.
            s = SeqOnly([1, 2, 3])
            return x + sum(tuple(s))

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # --- (e) Sequence protocol fallback (in operator) ---
    # A user-defined class with __getitem__ + __len__ but no __iter__
    # exercises the sequence protocol in CPython and the __getitem__
    # fallback in Dynamo's CONTAINS_OP polyfill.

    def test_contains_getitem_fallback(self):
        # [FUNCTION DOC] The in operator on an object with __getitem__ and __len__ but no
        # __iter__ finds a present item; matches eager.
        """'in' operator works on types with __getitem__ but no __iter__."""

        class SeqOnly:
            # [CLASS DOC] HELPER: Class with only __getitem__ (delegating to a list) and
            # __len__, no __iter__, used for the containment operator fallback.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the given items list.
                self.items = items

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the stored item at index i.
                return self.items[i]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of stored items.
                return len(self.items)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds 10 to the input if 2 is
            # contained in the sequence-only object, else 0.
            s = SeqOnly([1, 2, 3])
            return x + (10 if 2 in s else 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    def test_contains_getitem_fallback_missing(self):
        # [FUNCTION DOC] The in operator returns False for an item absent from a sequence-
        # protocol-only object; matches eager.
        """'in' returns False when item is not found via __getitem__ iteration."""

        class SeqOnly:
            # [CLASS DOC] HELPER: Class with only __getitem__ (delegating to a list) and
            # __len__, no __iter__, used for the containment operator fallback.
            def __init__(self, items):
                # [FUNCTION DOC] HELPER: Store the given items list.
                self.items = items

            def __getitem__(self, i):
                # [FUNCTION DOC] HELPER: Return the stored item at index i.
                return self.items[i]

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of stored items.
                return len(self.items)

        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds 10 to the input if 99 is
            # contained in the sequence-only object, else 0.
            s = SeqOnly([1, 2, 3])
            return x + (10 if 99 in s else 0)

        x = torch.randn(4)
        self.assertEqual(fn(x), self._compile(fn, x))

    # ===================================================================
    # sq_item_impl override checks
    # These types have mp_subscript so Branch 1 always wins at runtime.
    # Verify the defensive sq_item_impl overrides exist and aren't the
    # base class fallback (which calls unimplemented).
    # ===================================================================

    def test_sq_item_impl_overrides(self):
        # [FUNCTION DOC] Checks that the sq_item_impl method of base list, range, constant and
        # deque variable trackers differs from the base VariableTracker one. Fails the
        # assertion with a message naming the class when not overridden.
        """All sequence types override sq_item_impl from the base class."""
        base = VariableTracker.sq_item_impl
        for cls in (BaseListVariable, RangeVariable, ConstantVariable, DequeVariable):
            self.assertIsNot(
                cls.sq_item_impl,
                base,
                lambda msg: f"{msg}\n{cls.__name__} must override sq_item_impl",
            )

    # tests for python_qualified_name (exposed via __iter__ error messages in python 3.15+)
    @unittest.skipIf(
        sys.version_info < (3, 15), "Error messages in < 3.15 don't have module name"
    )
    def test_qualname_custom_mod(self):
        # [FUNCTION DOC] Under Python 3.15 and later, the error message for an iterator-less
        # __iter__ return uses the custom module name in the qualified class name; compares
        # the compiled message with an expected string. Skipped on earlier versions.
        class IterNoNext:
            # [CLASS DOC] HELPER: Class whose __iter__ returns itself, with its module name
            # set to test_mod.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self

            __module__ = "test_mod"

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a list from the
            # object and returns the TypeError message text.
            try:
                list(IterNoNext())
            except TypeError as exc:
                return str(exc)

        self.assertEqual(
            self._compile(fn),
            "test_mod.GetItemTests.test_qualname_custom_mod.<locals>.IterNoNext.__iter__() must return an iterator, not test_mod.GetItemTests.test_qualname_custom_mod.<locals>.IterNoNext",
        )

    @unittest.skipIf(
        sys.version_info < (3, 15), "Error messages in < 3.15 don't have module name"
    )
    def test_qualname_custom_mod_builtins(self):
        # [FUNCTION DOC] As for the custom module test, but with module name builtins: the
        # expected message omits the module prefix. Skipped before Python 3.15.
        class IterNoNext:
            # [CLASS DOC] HELPER: Class whose __iter__ returns itself, with its module name
            # set to builtins.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self

            __module__ = "builtins"

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a list from the
            # object and returns the TypeError message text.
            try:
                list(IterNoNext())
            except TypeError as exc:
                return str(exc)

        self.assertEqual(
            self._compile(fn),
            "GetItemTests.test_qualname_custom_mod_builtins.<locals>.IterNoNext.__iter__() must return an iterator, not GetItemTests.test_qualname_custom_mod_builtins.<locals>.IterNoNext",
        )

    @unittest.skipIf(
        sys.version_info < (3, 15), "Error messages in < 3.15 don't have module name"
    )
    def test_qualname_custom_mod_main(self):
        # [FUNCTION DOC] As for the custom module test, but with module name __main__: the
        # expected message omits the module prefix. Skipped before Python 3.15.
        class IterNoNext:
            # [CLASS DOC] HELPER: Class whose __iter__ returns itself, with its module name
            # set to __main__.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return the instance itself.
                return self

            __module__ = "__main__"

        def fn():
            # [FUNCTION DOC] HELPER: Function under test. Tries to build a list from the
            # object and returns the TypeError message text.
            try:
                list(IterNoNext())
            except TypeError as exc:
                return str(exc)

        self.assertEqual(
            self._compile(fn),
            "GetItemTests.test_qualname_custom_mod_main.<locals>.IterNoNext.__iter__() must return an iterator, not GetItemTests.test_qualname_custom_mod_main.<locals>.IterNoNext",
        )


class SetDelItemTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for setitem and delitem handling, via both subscript assignment or
    # deletion statements and the operator module functions. No state.
    """Tests for call_setitem / call_delitem: operator.setitem / operator.delitem
    dispatch (also the STORE_SUBSCR / DELETE_SUBSCR bytecode paths)."""

    def _compile(self, fn, *args):
        # [FUNCTION DOC] HELPER: Compile the given function with the eager backend in
        # fullgraph mode, call it with the given positional arguments and return its result.
        return torch.compile(fn, backend="eager", fullgraph=True)(*args)

    def test_list_setitem(self):
        # [FUNCTION DOC] Assigning list items via a subscript statement and the setitem
        # operator is traced; compiled result equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a list, assigns index 0 by
            # statement and index 2 via the operator, and returns the list with the input
            # plus one.
            lst = [1, 2, 3]
            lst[0] = 99  # STORE_SUBSCR
            operator.setitem(lst, 2, 7)
            return lst, x + 1

        self.assertEqual(self._compile(fn, torch.zeros(1)), fn(torch.zeros(1)))

    def test_list_delitem(self):
        # [FUNCTION DOC] Deleting list items via a statement and the delitem operator is
        # traced; compiled result equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Creates a four-element list, deletes
            # index 0 by statement and index 1 via the operator, and returns it with the input
            # plus one.
            lst = [1, 2, 3, 4]
            del lst[0]  # DELETE_SUBSCR
            operator.delitem(lst, 1)
            return lst, x + 1

        self.assertEqual(self._compile(fn, torch.zeros(1)), fn(torch.zeros(1)))

    def test_dict_setitem_delitem(self):
        # [FUNCTION DOC] Setting and deleting dict keys via statements and operator functions
        # is traced; compiled result equals eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Adds keys b and c, then deletes keys
            # a and b, returning the dict with the input plus one.
            d = {"a": 1}
            d["b"] = 2  # STORE_SUBSCR
            operator.setitem(d, "c", 3)
            del d["a"]  # DELETE_SUBSCR
            operator.delitem(d, "b")
            return d, x + 1

        self.assertEqual(self._compile(fn, torch.zeros(1)), fn(torch.zeros(1)))

    def test_tensor_setitem(self):
        # [FUNCTION DOC] Item assignment on a tensor clone via a statement and the setitem
        # operator matches eager.
        def fn(t):
            # [FUNCTION DOC] HELPER: Function under test. Clones the input, sets index 0 to
            # 5.0 by statement and index 1 to 9.0 via the operator, and returns it.
            t = t.clone()
            t[0] = 5.0  # STORE_SUBSCR
            operator.setitem(t, 1, 9.0)
            return t

        self.assertEqual(self._compile(fn, torch.arange(3.0)), fn(torch.arange(3.0)))

    def test_tensor_delitem_raises(self):
        # [FUNCTION DOC] Deleting an item of a tensor raises TypeError saying deletion is
        # unsupported; the compiled error text must contain that message.
        # Tensor.mp_ass_subscript_impl rejects deletion (value is None).
        # [FUNCTION DOC] HELPER: Function under test. Tries to delete element 0 of the input
        # and returns the error text, or the string no error.
        def fn(t):
            try:
                del t[0]
                return "no error"
            except TypeError as e:
                return str(e)

        result = self._compile(fn, torch.arange(3.0))
        self.assertIn("Tensor does not support deleting items", result)

    def test_numpy_setitem_sequence_rhs(self):
        # [FUNCTION DOC] Regression test: assigning a Python sequence to a numpy array slice
        # or element broadcasts like numpy instead of failing in the tensor path; compiled
        # result equals eager.
        # Regression: numpy setitem with a Python sequence RHS. numpy broadcasts
        # the sequence; the tensor setitem path rejects it ("can't assign a
        # tuple/list to a torch.LongTensor"). Mirrors scipy _broadcast_shapes.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Builds a 2x2 int numpy array,
            # assigns tuples and lists to row slices, to row 0 and, via the operator, to row
            # 1, and returns the array with the input plus one.
            a = np.ones((2, 2), dtype=int)
            for row in a:
                row[1:] = (7,)  # STORE_SUBSCR, tuple RHS
            a[0] = [3, 4]  # list RHS
            operator.setitem(a, 1, (5, 6))  # operator.setitem, tuple RHS
            return a, x + 1

        self.assertEqual(self._compile(fn, torch.zeros(1)), fn(torch.zeros(1)))

    def test_numpy_delitem_raises(self):
        # [FUNCTION DOC] Deleting an element of a numpy array raises the same ValueError numpy
        # gives eagerly; compiled message must contain that it cannot delete array elements.
        # NumpyNdarrayVariable.mp_ass_subscript_impl rejects deletion with the
        # same ValueError numpy raises eagerly.
        def fn(x):
            # [FUNCTION DOC] HELPER: Function under test. Tries to delete element 0 of a numpy
            # array and returns the error text, or the string no error.
            a = np.ones(3, dtype=int)
            try:
                del a[0]
                return "no error"
            except ValueError as e:
                return str(e)

        result = self._compile(fn, torch.zeros(1))
        self.assertIn("cannot delete array elements", result)


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
