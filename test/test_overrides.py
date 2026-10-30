# Owner(s): ["module: __torch_function__"]
# [MODULE DOC] Tests for the __torch_function__ override protocol and torch function modes.
# Defines pure-python instrumented functions, several tensor-like and tensor-subclass fixtures
# with dispatch tables, a Wrapper container, and many regression test classes. At import it
# exits the whole process silently on builds whose CPU capability env var is default or avx2,
# builds the override tables from the testing-overrides registry (raising if any overridable
# function is untested), attaches one generated test per override to the main test class, and
# instantiates device-generic redispatch op tests.

import sys
import torch
import numpy as np
import inspect
import functools
import pprint
import pickle
import collections
import unittest
import os

from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    TEST_WITH_CROSSREF,
    TEST_WITH_TORCHDYNAMO,
    skipIfTorchDynamo,
)
from torch.testing._internal.common_subclass import RedispatchTensor
from torch._dynamo.utils import clone_input
from torch.testing._internal.inductor_utils import clone_preserve_strides_offset
from torch.overrides import (
    handle_torch_function,
    has_torch_function,
    get_ignored_functions,
    get_overridable_functions,
    get_testing_overrides,
    resolve_name,
    is_tensor_method_or_property,
    TorchFunctionMode,
    _get_current_function_mode,
    _get_current_function_mode_stack,
    BaseTorchFunctionMode,
    redispatch_function
)
from torch.testing._internal.common_device_type import (
    ops,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.opinfo.core import SampleInput
from torch.utils._mode_utils import all_same_mode
from torch.utils._pytree import tree_map

Tensor = torch.Tensor

if os.getenv("ATEN_CPU_CAPABILITY") in ("default", "avx2"):
    # This test is not supported on ARM
    print(
        "Skipping due to failing when cuda build runs on non cuda machine, "
        + "see https://github.com/pytorch/pytorch/pull/150059 for example"
    )
    sys.exit()

# The functions below simulate the pure-python torch functions in the
# torch.functional namespace. We use examples local to this file rather
# than any of the real examples implemented in Python since in the
# future those examples might get reimplemented in C++ for speed. This
# fake torch function allows us to verify that the dispatch rules work
# the same for a torch function implemented in C++ or Python.

def foo(a, b, c=None):
    # [FUNCTION DOC] HELPER: Instrumented python function with an optional third argument.
    # If any argument has a torch function override it is routed through
    # handle_torch_function; otherwise returns the sum of the first two plus the third when
    # it is given.
    """A function multiple arguments and an optional argument"""
    if has_torch_function((a, b, c)):
        return handle_torch_function(foo, (a, b, c), a, b, c=c)
    if c is not None:
        return a + b + c
    return a + b

def bar(a):
    # [FUNCTION DOC] HELPER: Single-argument instrumented function. Dispatches through
    # handle_torch_function when the argument overrides torch function, otherwise returns
    # the argument unchanged.
    """A function with one argument"""
    if has_torch_function((a,)):
        return handle_torch_function(bar, (a,), a)
    return a

def baz(a, b):
    # [FUNCTION DOC] HELPER: Two-argument instrumented function. Dispatches through
    # handle_torch_function when either argument overrides torch function, otherwise returns
    # their sum.
    """A function with multiple arguments"""
    if has_torch_function((a, b)):
        return handle_torch_function(baz, (a, b), a, b)
    return a + b

def quux(a):
    # [FUNCTION DOC] HELPER: Single-argument instrumented function used to check that errors
    # raised by user overrides propagate. Dispatches via handle_torch_function when
    # overridden, otherwise returns the argument.
    """Used to test that errors raised in user implementations get propagated"""
    if has_torch_function((a,)):
        return handle_torch_function(quux, (a,), a)
    return a

# HANDLED_FUNCTIONS_DIAGONAL is a dispatch table that
# DiagonalTensor.__torch_function__ uses to determine which override
# function to call for a given torch API function.  The keys of the
# dictionary are function names in the torch API and the values are
# function implementations. Implementations are added to
# HANDLED_FUNCTION_DIAGONAL by decorating a python function with
# implements_diagonal. See the overrides immediately below the definition
# of DiagonalTensor for usage examples.
HANDLED_FUNCTIONS_DIAGONAL = {}

def implements_diagonal(torch_function):
    # [FUNCTION DOC] HELPER: Decorator factory that registers an override for a torch API
    # function in the dispatch table for DiagonalTensor. Takes the torch function to
    # override and returns a decorator.
    """Register a torch function override for DiagonalTensor.

    This decorator takes a function in the torch API as a
    parameter. Applying this decorator to a function adds that function
    as the registered override for the torch function passed as a
    parameter to the decorator. See DiagonalTensor.__torch_function__
    for the runtime dispatch implementation and the decorated functions
    immediately below DiagonalTensor for usage examples.
    """
    @functools.wraps(torch_function)
    def decorator(func):
        # [FUNCTION DOC] HELPER: Decorator returned by the factory: stores the decorated
        # implementation in the diagonal dispatch table keyed by the torch function, and
        # returns the implementation unchanged.
        HANDLED_FUNCTIONS_DIAGONAL[torch_function] = func
        return func
    return decorator

class DiagonalTensor:
    # [CLASS DOC] Fixture that is not a Tensor subclass but implements the override
    # protocol. Represents a square diagonal matrix with a size and one repeated value,
    # stored privately. Overrides are looked up in the class attribute handled_functions so
    # subclasses can swap in their own table; unknown functions yield NotImplemented.
    """A class with __torch_function__ and a specific diagonal representation

    This class has limited utility and is mostly useful for verifying that the
    dispatch mechanism works as expected. It is based on the `DiagonalArray
    example`_ in the NumPy documentation.

    Note that this class does *not* inherit from ``torch.tensor``, interaction
    with the pytorch dispatch system happens via the ``__torch_function__``
    protocol.

    ``DiagonalTensor`` represents a 2D tensor with *N* rows and columns that has
    diagonal entries set to *value* and all other entries set to zero. The
    main functionality of ``DiagonalTensor`` is to provide a more compact
    string representation of a diagonal tensor than in the base tensor class:

    >>> d = DiagonalTensor(5, 2)
    >>> d
    DiagonalTensor(N=5, value=2)
    >>> d.tensor()
    tensor([[2., 0., 0., 0., 0.],
            [0., 2., 0., 0., 0.],
            [0., 0., 2., 0., 0.],
            [0., 0., 0., 2., 0.],
            [0., 0., 0., 0., 2.]])

    Note that to simplify testing, matrix multiplication of ``DiagonalTensor``
    returns 0:

    >>> torch.mm(d, d)
    0

    .. _DiagonalArray example:
        https://numpy.org/devdocs/user/basics.dispatch.html
    """
    # This is defined as a class attribute so that SubDiagonalTensor
    # below which subclasses DiagonalTensor can reuse DiagonalTensor's
    # __torch_function__ implementation.
    handled_functions = HANDLED_FUNCTIONS_DIAGONAL

    def __init__(self, N, value):
        # [FUNCTION DOC] HELPER: Stores the matrix size and the diagonal value.
        self._N = N
        self._i = value

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string showing the class name, size,
        # and value.
        return f"DiagonalTensor(N={self._N}, value={self._i})"

    def __array__(self):
        # [FUNCTION DOC] HELPER: Returns a numpy array equal to the stored value times
        # an identity matrix of the stored size.
        return self._i * np.eye(self._N)

    def tensor(self):
        # [FUNCTION DOC] Returns the dense float torch tensor equal to the stored value
        # times an identity matrix.
        return self._i * torch.eye(self._N)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Class-level override hook. Defaults kwargs to an empty mapping,
        # returns NotImplemented if the function is absent from the class dispatch
        # table, otherwise calls the registered implementation with the arguments.
        if kwargs is None:
            kwargs = {}
        if func not in cls.handled_functions:
            return NotImplemented
        return cls.handled_functions[func](*args, **kwargs)

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality by exact type, size and value; objects of a
        # different type compare unequal.
        return type(other) is type(self) and self._N == other._N and self._i == other._i

@implements_diagonal(torch.mean)
def mean(mat):
    # [FUNCTION DOC] HELPER: Override for mean: returns the stored value divided by the size
    # as a float.
    return float(mat._i) / mat._N

@implements_diagonal(torch.mm)
def diagonal_mm(mat1, mat2):
    # [FUNCTION DOC] HELPER: Override for mm: ignores its inputs and returns zero.
    return 0

@implements_diagonal(torch.div)
def diagonal_div(input, other, out=None):
    # [FUNCTION DOC] HELPER: Override for div: ignores its inputs and returns minus one.
    return -1

@implements_diagonal(torch.add)
def add(mat1, mat2):
    # [FUNCTION DOC] HELPER: Override for add: always raises ValueError, used to test error
    # propagation.
    raise ValueError

@implements_diagonal(foo)
def diagonal_foo(a, b, c=None):
    # [FUNCTION DOC] HELPER: Override for the instrumented foo function: returns minus one.
    return -1

@implements_diagonal(bar)
def diagonal_bar(a):
    # [FUNCTION DOC] HELPER: Override for the instrumented bar function: returns minus one.
    return -1

@implements_diagonal(quux)
def diagonal_quux(a):
    # [FUNCTION DOC] HELPER: Override for the instrumented quux function: always raises
    # ValueError.
    raise ValueError

# The dispatch table for SubTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB = {}

def implements_sub(torch_function):
    # [FUNCTION DOC] HELPER: Decorator factory registering overrides in the dispatch table
    # for SubTensor.
    "Register a torch function override for SubTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        # [FUNCTION DOC] HELPER: Decorator that stores the implementation in the
        # SubTensor dispatch table keyed by the torch function and returns it unchanged.
        HANDLED_FUNCTIONS_SUB[torch_function] = func
        return func
    return decorator

class SubTensor(torch.Tensor):
    # [CLASS DOC] Tensor subclass fixture whose __torch_function__ looks up the module-level
    # SubTensor dispatch table. Used to test dispatch precedence and subclass behavior;
    # unregistered functions give NotImplemented.
    """A subclass of torch.Tensor use for testing __torch_function__ dispatch

    This class has the property that matrix multiplication returns zero:

    >>> s = SubTensor([[1, 1], [1, 1]])
    >>> torch.mm(s, s)
    0
    >>> t = torch.tensor([[1, 1], [1, 1]])
    >>> torch.mm(s, t)
    0
    >>> torch.mm(t, s)
    0
    >>> torch.mm(t, t)
    tensor([[2, 2],
            [2, 2]])

    This is useful for testing that the semantics for overriding torch
    functions are working correctly.
    """
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Class-level override hook: defaults kwargs to an empty mapping,
        # returns NotImplemented when the function is not in the SubTensor table, else
        # calls the registered implementation.
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_SUB:
            return NotImplemented
        return HANDLED_FUNCTIONS_SUB[func](*args, **kwargs)

class SubTensor2(torch.Tensor):
    # [CLASS DOC] HELPER: Plain Tensor subclass with no overrides, used for subclass
    # propagation tests.
    pass

class SubSubTensor2(SubTensor2):
    # [CLASS DOC] HELPER: Subclass of the previous plain Tensor subclass, used for subclass-
    # of-subclass propagation tests.
    pass

class SubTensor3(torch.Tensor):
    # [CLASS DOC] HELPER: Second unrelated plain Tensor subclass, used to check that
    # unrelated class trees do not merge.
    pass

@implements_sub(torch.mean)
def sub_mean(mat):
    # [FUNCTION DOC] HELPER: SubTensor override for mean: returns zero.
    return 0

@implements_sub(torch.mm)
def sub_mm(mat1, mat2):
    # [FUNCTION DOC] HELPER: SubTensor override for mm: returns minus one.
    return -1

@implements_sub(bar)
def sub_bar(mat):
    # [FUNCTION DOC] HELPER: SubTensor override for bar: returns one.
    return 1

@implements_sub(torch.div)
def sub_div(input, other, out=None):
    # [FUNCTION DOC] HELPER: SubTensor override for div: returns NotImplemented so dispatch
    # continues to other types.
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_SUB_DIAGONAL = {}

def implements_sub_diagonal(torch_function):
    # [FUNCTION DOC] HELPER: Decorator factory registering overrides in the dispatch table
    # for SubDiagonalTensor.
    "Register a torch function override for SubDiagonalTensor"
    @functools.wraps(torch_function)
    def decorator(func):
        # [FUNCTION DOC] HELPER: Decorator that stores the implementation in the
        # SubDiagonalTensor table keyed by the torch function and returns it unchanged.
        HANDLED_FUNCTIONS_SUB_DIAGONAL[torch_function] = func
        return func
    return decorator

class SubDiagonalTensor(DiagonalTensor):
    # [CLASS DOC] Subclass of DiagonalTensor that uses its own dispatch table (via the
    # handled_functions class attribute) and its own repr, to test overrides on a subclass
    # of a class that already defines the protocol.
    """A subclass of ``DiagonalTensor`` to test custom dispatch

    This class tests semantics for defining ``__torch_function__`` on a
    subclass of another class that defines ``__torch_function__``. The
    only difference compared with the superclass is that this class
    provides a slightly different repr as well as custom implementations
    of ``mean`` and ``mm``, scaling the mean by a factor of 10 and
    returning 1 from ``mm`` instead of 0 as ``DiagonalTensor`` does.
    """
    handled_functions = HANDLED_FUNCTIONS_SUB_DIAGONAL

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns a readable string showing the SubDiagonalTensor
        # class name, size and value.
        return f"SubDiagonalTensor(N={self._N}, value={self._i})"


@implements_sub_diagonal(torch.mean)
def sub_diagonal_mean(mat):
    # [FUNCTION DOC] HELPER: SubDiagonalTensor override for mean: ten times the value
    # divided by the size.
    return 10 * float(mat._i) / mat._N

@implements_sub_diagonal(bar)
def sub_diagonal_bar(mat):
    # [FUNCTION DOC] HELPER: SubDiagonalTensor override for bar: returns zero.
    return 0

@implements_sub_diagonal(torch.mm)
def sub_diagonal_mm(mat1, mat2):
    # [FUNCTION DOC] HELPER: SubDiagonalTensor override for mm: returns one.
    return 1

@implements_sub_diagonal(torch.div)
def sub_diagonal_div(input, other, out=None):
    # [FUNCTION DOC] HELPER: SubDiagonalTensor override for div: returns NotImplemented.
    return NotImplemented

@implements_sub_diagonal(foo)
def sub_diagonal_foo(a, b, c=None):
    # [FUNCTION DOC] HELPER: SubDiagonalTensor override for foo: returns NotImplemented.
    return NotImplemented

# The dispatch table for SubDiagonalTensor's __torch_function__ implementation.
HANDLED_FUNCTIONS_TENSOR_LIKE = {}


# Note: _triggered wrapper
# Dict that wraps the implementations from get_testing_overrides into another
# function with a _triggered slot/flag. The triggered flag is set when the
# implementation is called.
WRAPPED_TRIGGERED_IMPLS = {}


def triggered_wrapper(f):
    # [FUNCTION DOC] HELPER: Wraps an implementation so that calling it sets a triggered
    # attribute on the wrapper, initially false, then forwards to the original. Returns the
    # wrapper.
    @functools.wraps(f)
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Inner wrapper: marks itself as triggered and returns
        # the original function's result for the given arguments.
        wrapped._triggered = True
        return f(*args, **kwargs)

    wrapped._triggered = False
    return wrapped

def implements_tensor_like(torch_function):
    # [FUNCTION DOC] HELPER: Decorator factory registering overrides in the dispatch table
    # for TensorLike.
    "Register a torch function override for TensorLike"
    @functools.wraps(torch_function)
    def decorator(func):
        # [FUNCTION DOC] HELPER: Decorator that stores the implementation in the
        # TensorLike table keyed by the torch function and returns it unchanged.
        HANDLED_FUNCTIONS_TENSOR_LIKE[torch_function] = func
        return func
    return decorator

def generate_tensor_like_torch_implementations():
    # [FUNCTION DOC] Builds the override tables from the registry of testing overrides.
    # First collects every overridable function that has no testing override (ignoring two
    # known monkeypatched names) and raises AssertionError listing them. Then wraps each
    # override with a triggered flag, records it in the wrapped-implementations mapping, and
    # registers it for SubTensor if it is a Tensor method or property, else for TensorLike.
    # Mutates module-level tables; called once at import.
    untested_funcs = []
    testing_overrides = get_testing_overrides()
    # test/test_cpp_api_parity.py monkeypatches torch.nn to have a new
    # function sample_functional.  Depending on what order you run pytest
    # collection, this may trigger the error here.  This is a hack to fix
    # the problem.  A more proper fix is to make the "not tested" check
    # a test on its own, and to make sure the monkeypatch is only installed
    # for the span of the relevant test (and deleted afterwards)
    testing_ignore = {"sample_functional", "autocast"}
    for namespace, funcs in get_overridable_functions().items():
        for func in funcs:
            if func not in testing_overrides and func.__name__ not in testing_ignore:
                untested_funcs.append(f"{namespace}.{func.__name__}")
    msg = (
        "The following functions are not tested for __torch_function__ "
        "support, please ensure there is an entry in the dict returned by "
        "torch.overrides.get_testing_overrides for this function or if a "
        "__torch_function__ override does not make sense, add an entry to "
        "the tuple returned by torch._overrides.get_ignored_functions.\n\n{}"
    )
    if len(untested_funcs) != 0:
        raise AssertionError(msg.format(pprint.pformat(untested_funcs)))
    for func, override in testing_overrides.items():
        # decorate the overrides with implements_tensor_like if it's not a
        # torch.Tensor method
        wrapped = triggered_wrapper(override)
        # See note: "_triggered wrapper"
        WRAPPED_TRIGGERED_IMPLS[func] = wrapped
        if is_tensor_method_or_property(func):
            implements_sub(func)(wrapped)
        else:
            implements_tensor_like(func)(wrapped)

generate_tensor_like_torch_implementations()

class TensorLike:
    # [CLASS DOC] Non-Tensor fixture whose __torch_function__ dispatches through the
    # TensorLike table filled from the testing overrides, used to check the full torch API
    # can be overridden.
    """A class that overrides the full torch API

    This class is used to explicitly test that the full torch.tensor API
    can be overridden with a class that defines __torch_function__.
    """
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Class-level override hook: defaults kwargs to an empty mapping,
        # returns NotImplemented for functions not in the TensorLike table, else calls
        # the registered wrapped implementation.
        if kwargs is None:
            kwargs = {}

        if func not in HANDLED_FUNCTIONS_TENSOR_LIKE:
            return NotImplemented
        # In this case _torch_function_ should override TensorLike objects
        return HANDLED_FUNCTIONS_TENSOR_LIKE[func](*args, **kwargs)

class TestTorchFunctionOverride(TestCase):
    # [CLASS DOC] Main test class for the override protocol. Hand-written tests below check
    # dispatch semantics; one extra test per overridable function is attached to it at
    # import by the generator function.
    def test_dtype_override(self):
        # [FUNCTION DOC] Checks that an object defining __torch_function__ is called
        # when it is passed as a dtype to a view, and its return value is returned.
        class MyDtype:
            # [CLASS DOC] HELPER: Fixture whose override hook always returns four.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Override hook that ignores its
                # arguments and returns four.
                return 4

        self.assertEqual(torch.empty(4).view(MyDtype()), 4)

    def test_mean_semantics(self):
        # [FUNCTION DOC] Checks single-argument override dispatch for DiagonalTensor,
        # SubTensor and SubDiagonalTensor through mean and the instrumented bar
        # function, comparing against each class's registered values.
        """Test that a function with one argument can be overridden"""
        t1 = DiagonalTensor(5, 2)
        t2 = SubTensor([[1, 2], [1, 2]])
        t3 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.mean(t1), 0.4)
        self.assertEqual(bar(t1), -1)
        self.assertEqual(torch.mean(t2), 0)
        self.assertEqual(bar(t2), 1)
        self.assertEqual(torch.mean(t3), 4.0)
        self.assertEqual(bar(t3), 0)

    def test_has_torch_function_non_sequence(self):
        # [FUNCTION DOC] Checks that has_torch_function raises TypeError mentioning a
        # sequence when given a non-sequence object.
        with self.assertRaisesRegex(TypeError, "expected a sequence"):
            has_torch_function(object())

    def test_mm_semantics(self):
        # [FUNCTION DOC] Checks multi-argument dispatch for mm across DiagonalTensor,
        # SubTensor, plain tensors and SubDiagonalTensor: argument order matters for
        # unrelated classes, subclasses take precedence over their parents.
        """Test that a function with multiple arguments can be overridden"""
        t1 = DiagonalTensor(5, 2)
        t2 = torch.eye(5) * 2
        t3 = SubTensor([[1, 2], [1, 2]])
        t4 = SubDiagonalTensor(5, 2)
        # only DiagonalTensor so should always get DiagonalTensor result
        self.assertEqual(torch.mm(t1, t1), 0)
        # tensor and DiagonalTensor, always return DiagonalTensor result
        self.assertEqual(torch.mm(t1, t2), 0)
        self.assertEqual(torch.mm(t2, t1), 0)
        # only SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t3), -1)
        # tensor and SubTensor so should always get SubTensor result
        self.assertEqual(torch.mm(t3, t2), -1)
        self.assertEqual(torch.mm(t2, t3), -1)
        # DiagonalTensor and SubTensor are unrelated classes so the result
        # depends on which argument appears first
        self.assertEqual(torch.mm(t3, t1), -1)
        self.assertEqual(torch.mm(t1, t3), 0)
        # SubDiagonalTensor should take precedence over DiagonalTensor
        # but should behave otherwise the same as DiagonalTensor
        self.assertEqual(torch.mm(t4, t4), 1)
        self.assertEqual(torch.mm(t4, t1), 1)
        self.assertEqual(torch.mm(t1, t4), 1)
        self.assertEqual(torch.mm(t4, t2), 1)
        self.assertEqual(torch.mm(t2, t4), 1)
        self.assertEqual(torch.mm(t3, t4), -1)
        self.assertEqual(torch.mm(t4, t3), 1)

    def test_precedence_semantics(self):
        # [FUNCTION DOC] Checks dispatch precedence rules with div, foo, mul and baz:
        # implementations returning NotImplemented fall through to the next type, a
        # registered one is used, and TypeError is raised when every type returns
        # NotImplemented or none is registered.
        """Test semantics for __torch_function__ for functions that take
        multiple arguments

        For functions that take multiple arguments, the appropriate
        __torch_function__ implementation to call is determined by
        examining the types of the arguments. The precedence order is
        left-to-right in the argument list, except subclasses are always
        checked before superclasses. The first result of calling the
        implementations in precedence order that is not NotImplemented
        is returned to the user. If all implementations return
        NotImplemented, a TypeError is raised.

        All cases are tested with functions implemented in C++ and
        either foo or baz, which are python functions defined above that
        are instrumented to obey the same dispatch rules as the
        functions in torch.functional.
        """
        # DiagonalTensor has a valid override and SubDiagonal has an
        # override that returns NotImplemented so we should call the
        # DiagonalTensor implementation, returning -1
        t1 = DiagonalTensor(5, 2)
        t2 = SubDiagonalTensor(5, 2)
        self.assertEqual(torch.div(t1, t2), -1)
        self.assertEqual(torch.div(t2, t1), -1)
        self.assertEqual(foo(t1, t2), -1)
        self.assertEqual(foo(t2, t1), -1)

        # SubTensor has an implementation that returns NotImplemented as
        # well so it should behave exactly like SubDiagonalTensor in the
        # test above
        t3 = SubTensor([[1, 2], [1, 2]])
        self.assertEqual(torch.div(t1, t3), -1)
        self.assertEqual(torch.div(t3, t1), -1)
        self.assertEqual(foo(t1, t3), -1)
        self.assertEqual(foo(t3, t1), -1)

        # div between SubTensor and SubDiagonalTensor should raise
        # TypeError since both have an implementation that
        # explicitly returns NotImplemented
        with self.assertRaises(TypeError):
            torch.div(t2, t3)
        with self.assertRaises(TypeError):
            torch.div(t3, t2)
        with self.assertRaises(TypeError):
            foo(t2, t3)
        with self.assertRaises(TypeError):
            foo(t3, t2)

        # none of DiagonalTensor, SubdiagonalTensor, or SubTensor have a
        # mul or a baz implementation so all ops should raise TypeError
        with self.assertRaises(TypeError):
            torch.mul(t1, t1)
        with self.assertRaises(TypeError):
            torch.mul(t1, t2)
        with self.assertRaises(TypeError):
            torch.mul(t1, t3)
        with self.assertRaises(TypeError):
            torch.mul(t2, t1)
        with self.assertRaises(TypeError):
            torch.mul(t2, t2)
        with self.assertRaises(TypeError):
            torch.mul(t2, t3)
        with self.assertRaises(TypeError):
            torch.mul(t3, t1)
        with self.assertRaises(TypeError):
            torch.mul(t3, t2)
        with self.assertRaises(TypeError):
            torch.mul(t3, t3)
        with self.assertRaises(TypeError):
            baz(t1, t1)
        with self.assertRaises(TypeError):
            baz(t1, t2)
        with self.assertRaises(TypeError):
            baz(t1, t3)
        with self.assertRaises(TypeError):
            baz(t2, t1)
        with self.assertRaises(TypeError):
            baz(t2, t2)
        with self.assertRaises(TypeError):
            baz(t2, t3)
        with self.assertRaises(TypeError):
            baz(t3, t1)
        with self.assertRaises(TypeError):
            baz(t3, t2)
        with self.assertRaises(TypeError):
            baz(t3, t3)

    def test_user_implementation_raises(self):
        # [FUNCTION DOC] Checks that ValueError raised inside user overrides propagates
        # out of both a torch function and the instrumented quux function.
        """Test that errors raised in user implementations propagate correctly"""
        t1 = DiagonalTensor(5, 2)
        t2 = DiagonalTensor(5, 2)
        with self.assertRaises(ValueError):
            torch.add(t1, t2)
        with self.assertRaises(ValueError):
            quux(t1)

    def test_tensor_subclass_propagation(self):
        # [FUNCTION DOC] Checks Tensor subclass propagation rules: results of arithmetic
        # and indexing keep the most derived subclass regardless of operand order, and
        # mixing unrelated subclasses raises TypeError.
        """this test exercises the functionality described in
        docs/source/notes/extending.md#subclassing-torch-tensor"""
        t1 = torch.tensor([5])
        t2 = torch.tensor([6])

        s1 = SubTensor2([5])
        s2 = SubTensor2([6])

        ss1 = SubSubTensor2([5])
        ss2 = SubSubTensor2([6])

        sn1 = SubTensor3([5])
        sn2 = SubTensor3([6])

        # Check that leaf subclass is kept regardless of order
        self.assertTrue(isinstance(s1 + t2, SubTensor2))
        self.assertTrue(isinstance(t1 + s2, SubTensor2))
        self.assertTrue(isinstance(s1 + s2, SubTensor2))

        # Check indexing subclass is kept
        self.assertTrue(isinstance(s1[0], SubTensor2))

        # Check case for subclass of subclass.
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + s2, SubSubTensor2))
        self.assertTrue(isinstance(s1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1 + t2, SubSubTensor2))
        self.assertTrue(isinstance(t1 + ss2, SubSubTensor2))
        self.assertTrue(isinstance(ss1[0], SubSubTensor2))

        # Make sure unrelated class trees are not merged.
        with self.assertRaises(TypeError):
            s1 + sn2
        with self.assertRaises(TypeError):
            sn1 + s2

    def test_base(self):
        # [FUNCTION DOC] Regression check for a view-base issue: a subclass instance
        # built from a tensor is a view whose base is the original tensor.
        # https://github.com/szagoruyko/pytorchviz/issues/65
        class DummyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass used for the view-base
            # check.
            pass

        a = torch.ones(1)
        c = DummyTensor(a)
        self.assertTrue(c._is_view())
        self.assertTrue(c._base is a)

    def test_grad(self):
        # [FUNCTION DOC] Checks that for autograd.grad a non-Tensor override receives
        # the inputs and outputs as one-element tuples, and returns its sentinel result.
        # Previously, Tensor-like objects that did not subclass from Tensor
        # did not get wrapped into unary tuples before being passed into
        # handle_torch_function, in contradiction with how Tensor-likes
        # were handled
        #
        # NB: this asserts that the arguments get normalized into a tuple
        # before entering the torch function handler; it could go the
        # other way but beware https://github.com/pytorch/pytorch/issues/76037

        class Dummy:
            # [CLASS DOC] HELPER: Fixture whose override hook asserts the normalized
            # inputs and outputs and returns minus one.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook that unpacks the first
                # two arguments, asserts both equal the one-element tuple
                # holding the test object (using the enclosing test case), and
                # returns minus one.
                inputs, outputs = args
                self.assertEqual(inputs, (x,))
                self.assertEqual(outputs, (x,))
                return -1

        x = Dummy()
        self.assertEqual(torch.autograd.grad(x, x), -1)

    def test_pow_rpow(self):
        # [FUNCTION DOC] Checks that a power operation between a subclass returning
        # NotImplemented and a subclass overriding only reverse power resolves to the
        # latter, returning minus one.
        class NothingImplemented(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose override hook returns
            # NotImplemented for everything.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook that always returns
                # NotImplemented.
                return NotImplemented

        class RPowOnly(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass that handles only reverse power
            # and returns NotImplemented otherwise.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook returning minus one for
                # reverse power, NotImplemented otherwise.
                if func is torch.Tensor.__rpow__:
                    return -1
                return NotImplemented

        self.assertEqual(NothingImplemented() ** RPowOnly(), -1)

    def test_torch_function_in_lists(self):
        # [FUNCTION DOC] Checks that objects with an override hook nested in integer
        # list arguments (padding) trigger their hook, including multiple such objects.
        """Test that __torch_function__ is called for objects inside lists"""

        class IntLike:
            # [CLASS DOC] HELPER: Int-like fixture that records whether its override
            # hook was called.
            """Object that can be used in int lists"""
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value and sets the called
                # flag to false.
                self.value = value
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook that records the call and
                # returns a result chosen by function name (the input for pad,
                # ones for layer norm, otherwise a constant tensor).
                self.torch_function_called = True
                # Return a result that makes the operation succeed
                if func.__name__ == 'pad':
                    # For pad, return the input with shape adjusted
                    return args[0]
                elif func.__name__ == 'layer_norm':
                    # For layer_norm, return normalized tensor
                    return torch.ones_like(args[0])
                elif func.__name__ == 'tensordot':
                    # For tensordot, return appropriate shape
                    return torch.tensor(42.0)
                # Fallback
                return torch.tensor(42.0)

        # Test with F.pad which takes int list
        import torch.nn.functional as F
        x = torch.randn(2, 3)
        obj = IntLike(1)

        # pad takes [left, right, top, bottom] as padding
        _ = F.pad(x, [1, obj, 0, 0])
        self.assertTrue(obj.torch_function_called,
                        "torch_function should be called for object in int list")

        # Test multiple objects in list
        obj1 = IntLike(1)
        obj2 = IntLike(2)
        _ = F.pad(x, [obj1, obj2, 0, 0])
        self.assertTrue(obj1.torch_function_called or obj2.torch_function_called,
                        "torch_function should be called for at least one object")

    def test_torch_function_in_float_lists(self):
        # [FUNCTION DOC] Checks that an override object inside a float-list argument
        # (normalized shape of layer norm) has its hook called.
        """Test that __torch_function__ is called for objects inside float lists"""

        class FloatLike:
            # [CLASS DOC] HELPER: Float-like fixture recording whether its override
            # hook was called.
            """Object that can be used in float lists"""
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value as a float and sets
                # the called flag to false.
                self.value = float(value)
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook recording the call and
                # returning ones for layer norm or a constant tensor otherwise.
                self.torch_function_called = True
                # Return appropriate result
                if func.__name__ == 'layer_norm':
                    return torch.ones_like(args[0])
                return torch.tensor(42.0)

        import torch.nn.functional as F
        x = torch.randn(2, 3, 4)
        obj = FloatLike(4.0)

        # layer_norm takes normalized_shape as int/float list
        _ = F.layer_norm(x, [3, obj])
        self.assertTrue(obj.torch_function_called,
                        "torch_function should be called for object in float list")

    def test_torch_function_in_scalar_lists(self):
        # [FUNCTION DOC] Checks that scalar-like override objects inside a list passed
        # to stack have their hook invoked.
        """Test that __torch_function__ is called for scalar objects inside lists"""

        class ScalarLike:
            # [CLASS DOC] HELPER: Scalar-like fixture that records hook calls and
            # supports float and int conversion.
            """Object that can be used as a scalar in lists"""
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value and sets the called
                # flag to false.
                self.value = value
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook recording the call and
                # returning a scalar tensor of the stored value.
                self.torch_function_called = True
                # Return a scalar tensor
                return torch.tensor(self.value)

            def __float__(self):
                # [FUNCTION DOC] HELPER: Converts the stored value to a float.
                return float(self.value)

            def __int__(self):
                # [FUNCTION DOC] HELPER: Converts the stored value to an int.
                return int(self.value)

        # Test with a function that takes scalar lists
        # Using torch.as_tensor which can take scalar lists
        obj1 = ScalarLike(1.0)
        obj2 = ScalarLike(2.0)

        # Create a tensor with scalar list containing torch function objects
        # Use a different operation that should trigger torch_function
        _ = torch.stack([obj1, obj2])
        self.assertTrue(obj1.torch_function_called or obj2.torch_function_called,
                        "torch_function should be called for scalar objects in list")

    def test_torch_function_precedence_in_lists(self):
        # [FUNCTION DOC] Checks that when two override objects are in one list, the
        # first one's hook runs first and, on NotImplemented, the second one's hook runs
        # next.
        """Test precedence when multiple torch function objects are in a list"""

        call_order = []

        class HighPriority:
            # [CLASS DOC] HELPER: Fixture whose hook records high and returns
            # NotImplemented.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook appending a high marker
                # to the shared call order list and returning NotImplemented.
                call_order.append('high')
                # Delegate to lower priority
                return NotImplemented

        class LowPriority:
            # [CLASS DOC] HELPER: Fixture whose hook records low and returns a valid
            # result.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook appending a low marker to
                # the shared call order list and returning the input for pad or
                # a constant tensor.
                call_order.append('low')
                # Return valid result
                if func.__name__ == 'pad':
                    return args[0]
                return torch.tensor(42.0)

        import torch.nn.functional as F
        x = torch.randn(2, 3)

        high = HighPriority()
        low = LowPriority()

        # Test with both objects in list
        call_order.clear()
        _ = F.pad(x, [1, high, low, 0])

        # High priority should be called first
        self.assertEqual(call_order[0], 'high',
                         "Higher priority torch_function should be called first")
        self.assertEqual(call_order[1], 'low',
                         "Lower priority torch_function should be called after NotImplemented")

    def test_torch_function_mixed_lists(self):
        # [FUNCTION DOC] Checks that a list mixing plain integers and one override
        # object calls the object's hook exactly once.
        """Test lists with mix of regular values and torch function objects"""

        class CountingInt:
            # [CLASS DOC] HELPER: Fixture counting hook calls in a class-level
            # counter and usable as an index.
            call_count = 0

            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the integer value.
                self.value = value

            @classmethod
            def reset(cls):
                # [FUNCTION DOC] HELPER: Class method resetting the shared call
                # counter to zero.
                cls.call_count = 0

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook incrementing the shared
                # counter and returning the input for pad or a constant tensor.
                CountingInt.call_count += 1
                # Return valid result
                if func.__name__ == 'pad':
                    return args[0]
                return torch.tensor(42.0)

            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value so the object
                # works as an integer index.
                return self.value

        import torch.nn.functional as F
        x = torch.randn(2, 3)

        obj = CountingInt(2)
        CountingInt.reset()

        # Mix regular ints with torch function object
        _ = F.pad(x, [1, obj, 0, 0])

        self.assertEqual(CountingInt.call_count, 1,
                         "torch_function should be called exactly once for mixed list")

    def test_torch_function_empty_lists(self):
        # [FUNCTION DOC] Checks that an empty list is accepted as a size argument for
        # view, giving a zero-dimensional result.
        """Test that empty lists work correctly"""

        # This should work without calling any torch_function
        x = torch.randn(1)  # Single element tensor

        # Functions that accept empty lists should still work
        # torch.stack with empty list of tensors would fail,
        # but empty size lists should work
        result = x.view([])  # Empty list means scalar
        self.assertEqual(result.shape, torch.Size([]),
                         "Empty list should work for size arguments")

    def test_torch_function_not_first_in_list(self):
        # [FUNCTION DOC] Checks the override hook runs for pad when the override object
        # is second, third or last in the padding list.
        """Test that torch_function is called even when object is not first in list"""

        class IntLikeNotFirst:
            # [CLASS DOC] HELPER: Index-like fixture recording whether its hook was
            # called.
            """Object with torch_function that won't be first in list"""
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value and sets the called
                # flag to false.
                self.value = value
                self.torch_function_called = False

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook recording the call and
                # returning the first argument.
                self.torch_function_called = True
                # Return input tensor for pad
                return args[0]

            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value for use as an
                # index.
                return self.value

        import torch.nn.functional as F
        x = torch.randn(2, 3)

        # Test with torch_function object as second item
        obj_second = IntLikeNotFirst(2)
        _ = F.pad(x, [1, obj_second, 0, 0])
        self.assertTrue(obj_second.torch_function_called,
                        "torch_function should be called when object is second in list")

        # Test with torch_function object as third item
        obj_third = IntLikeNotFirst(1)
        _ = F.pad(x, [1, 1, obj_third, 0])
        self.assertTrue(obj_third.torch_function_called,
                        "torch_function should be called when object is third in list")

        # Test with torch_function object as last item
        obj_last = IntLikeNotFirst(1)
        _ = F.pad(x, [1, 1, 1, obj_last])
        self.assertTrue(obj_last.torch_function_called,
                        "torch_function should be called when object is last in list")

    def test_torch_function_nested_tuple_getitem(self):
        # [FUNCTION DOC] Checks that indexing with a tuple that nests override objects
        # inside an inner tuple calls the hook with the item-access function, and not
        # the size function.
        """Test that torch_function is called with getitem for TF objects inside nested tuples"""

        called_functions = []

        class TorchFunctionObj:
            # [CLASS DOC] HELPER: Fixture recording the names of functions its hook
            # is called with in a shared list.
            """Object with torch_function that tracks which functions are called"""
            def __init__(self, value):
                # [FUNCTION DOC] HELPER: Stores the value.
                self.value = value

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook appending the function
                # name to the shared list; returns the first argument for item
                # access, else a constant tensor.
                called_functions.append(func.__name__)
                # For getitem, return the tensor unchanged
                if func.__name__ == '__getitem__':
                    return args[0]
                # Return a simple result for other functions
                return torch.tensor(42.0)

            def __index__(self):
                # [FUNCTION DOC] HELPER: Returns the stored value for use as an
                # index.
                return self.value

        # Create a tensor to index
        x = torch.randn(5, 5, 5)

        # Create torch function objects - these will be INSIDE the nested structure
        tf_obj1 = TorchFunctionObj(0)
        tf_obj2 = TorchFunctionObj(1)

        # Clear the called functions list
        called_functions.clear()

        # Test with tuple of tuple where TF objects are only on the INSIDE
        # The outer structure is regular tuples, but inner elements have __torch_function__
        # This tests the recursive detection logic added in the recent commit
        x[(0, (tf_obj1, tf_obj2))]

        # Assert that torch_function was called
        self.assertTrue(len(called_functions) > 0,
                        "torch_function should be called for TF objects inside nested tuples")

        # Assert that getitem was called, not size
        self.assertIn('__getitem__', called_functions,
                      "getitem should be called for tuple indexing with torch function objects inside")

        self.assertNotIn('size', called_functions,
                         "size should not be called - we should use getitem, not convert to advanced indexing")


def generate_tensor_like_override_tests(cls):
    # [FUNCTION DOC] Populates the given test class with one generated test per entry in the
    # testing overrides registry. The test name is built from the module (or Tensor for
    # methods and properties, with special handling of descriptor getters) and function
    # name. Mutates the class by setting attributes.
    from torch.testing._internal.generated.annotated_fn_args import annotated_args

    def test_generator(func, override):
        # [FUNCTION DOC] HELPER: Builds a test function for one torch function and its
        # override. Picks sample-instance generation by whether the function is a Tensor
        # method or property, builds positional and keyword arguments from the annotated
        # argument schema (or from the override's argspec as fallback), and returns the
        # test.
        # If func corresponds to a torch.Tensor method or property.
        if is_tensor_method_or_property(func):
            # Generate an instance by using SubTensor,
            def instance_gen():
                # [FUNCTION DOC] HELPER: Returns a one-element SubTensor, used
                # when the function is a Tensor method or property.
                return SubTensor([5])
        else:
            # Otherwise, TensorLike.
            def instance_gen():
                # [FUNCTION DOC] HELPER: Returns a fresh TensorLike instance,
                # used for non-method functions.
                return TensorLike()

        # FIXME The following code does not support kwonly args without defaults.
        # The fix is easy, as one just needs to save these args when generating the variable
        # annotated_args. The problem is that, if one does so, one finds a number
        # of functions that have problematic signatures in native_functions.yaml.
        # Fixing these would be BC breaking, so hence this terrible hack
        # https://github.com/pytorch/pytorch/issues/67008
        kwargs = {}
        if hasattr(func, "__name__") and "linalg_solve_triangular" in func.__name__:
            kwargs = {"upper": True}

        func_args = []
        is_method = is_tensor_method_or_property(func)

        def _simple_type_parser(func, arg_name, arg_type):
            # [FUNCTION DOC] HELPER: Maps a schema simple type name to a sample
            # argument value (tensor instances, lists, scalars, flags, dtype,
            # empty strings and similar); raises RuntimeError for unsupported type
            # names.
            # KNOWN ISSUE: the integer-array branch reads a size from the
            # enclosing loop variable of the schema argument rather than from its
            # own parameters, so it only works when called from that loop.
            # Guess valid input to aten function based on type of argument
            if arg_type == "Tensor":
                return instance_gen()
            elif arg_type == "TensorList" or arg_type == "ITensorListRef":
                return [instance_gen(), instance_gen()]
            elif arg_type == "c10::List<::std::optional<Tensor>>":
                return [instance_gen(), instance_gen()]
            elif arg_type == "IntArrayRef" or arg_type == "SymIntArrayRef":
                size = arg.get("size", 2)
                if size == 1:
                    return 1
                else:
                    return [1] * size
            elif arg_type == "Scalar":
                return 3.5
            elif arg_type == "bool":
                return False
            elif arg_type.startswith("int"):
                return 0
            elif arg_type == "Stream":
                return torch.Stream()
            elif arg_type.startswith("float") or arg_type == "double":
                return 1.0
            elif arg_type in {"Generator", "MemoryFormat", "TensorOptions"}:
                return None
            elif arg_type == "ScalarType":
                return torch.float32
            elif arg_type == "c10::string_view":
                return ""
            elif arg_type in ("std::string_view", "::std::string_view"):
                return ""
            elif arg_type == "SymInt":
                # TODO: generate actual SymbolicInt
                return 1
            else:
                raise RuntimeError(
                    f"Unsupported argument type {arg_type} for {arg_name} of function {func}"
                )

        # Special case; this doesn't have a schema but takes a list
        if func is torch.sym_sum:
            func_args.append([TensorLike(), TensorLike()])
        elif func in annotated_args:
            for arg in annotated_args[func]:
                # Guess valid input to aten function based on type of argument
                t = arg["simple_type"]
                t = t.removesuffix("?")
                if t == "Tensor" and is_method and arg["name"] == "self":
                    # See "Note: properties and __get__"
                    func = func.__get__(instance_gen())
                    continue
                arg_to_add = _simple_type_parser(func, arg["name"], t)
                if "is_kwarg_only" in arg and arg["is_kwarg_only"] == str(True):
                    kwargs[arg["name"]] = arg_to_add
                else:
                    func_args.append(arg_to_add)
        else:
            args = inspect.getfullargspec(override)
            try:
                func_args = inspect.getfullargspec(func)
                # Remove annotations from argspec
                func_args = type(func_args)(**{**func_args, 'annotations': None})
                if func_args != args:
                    raise RuntimeError(f"Override for {func} doesn't match its argspec.\n"
                                       + f"Original: {inspect.signature(func)}\n"
                                       + f"Override: {inspect.signature(override)}")
            except TypeError:
                pass
            nargs = len(args.args)
            if args.defaults is not None:
                nargs -= len(args.defaults)
            func_args = [instance_gen() for _ in range(nargs)]
            if args.varargs is not None:
                func_args += [instance_gen(), instance_gen()]

        def test(self):
            # [FUNCTION DOC] Generated test body: calls the function with the
            # prepared arguments. For non-methods or None results asserts the
            # override's triggered flag was set; otherwise asserts the result is
            # minus one.
            ret = func(*func_args, **kwargs)
            # ret is None for certain protocols, e.g., `__weakref__` and `__setitem__`
            # This is currently the best check but doesn't work for, for example,
            # Tensor.__add__ because it redirects to Tensor.add.
            # See note "_triggered wrapper"
            if not is_method or ret is None:
                self.assertTrue(WRAPPED_TRIGGERED_IMPLS[func]._triggered)
                return

            self.assertEqual(ret, -1)

        return test

    for func, override in get_testing_overrides().items():
        test_method = test_generator(func, override)
        if func.__name__ == "__get__":
            # Note: properties and __get__
            # __get__ is part of the descriptor protocol.
            # https://docs.python.org/3/howto/descriptor.html
            # This is used for properties of the form
            # torch.Tensor.<property>, with the method __get__
            # In this case we get the property name in two ways:

            # This case for properties defined in C.
            module = getattr(
                func.__self__,
                "__qualname__",
                None
            )

            # This one for properties defined in Python.
            if module is None:
                module = "Tensor." + func.__self__.fget.__name__

            # Unfortunately I couldn't find a way to unify these two cases
            # and there is no way for general descriptors.
        elif is_tensor_method_or_property(func):
            module = "Tensor"
        else:
            module = func.__module__
        if module:
            name = 'test_{}_{}'.format(module.replace('.', '_'), func.__name__)
        else:
            name = f'test_{func.__name__}'
        test_method.__name__ = name
        setattr(cls, name, test_method)

generate_tensor_like_override_tests(TestTorchFunctionOverride)

class Wrapper:
    # [CLASS DOC] Test double wrapping a value (typically a Tensor) that is not a Tensor
    # subclass; records attribute names accessed and torch functions called, unwraps
    # arguments before calling, and rewraps tensor results. Overloads arithmetic and
    # comparison operators to route through the override hook.
    "Basic data container that knows how to unwrap itself"
    def __init__(self, data):
        # [FUNCTION DOC] HELPER: Stores the wrapped data, plus empty sets for used
        # attributes and used calls, directly in the instance dictionary to bypass the
        # custom attribute hooks.
        self.__dict__["_data"] = data
        self.__dict__["used_attrs"] = set()
        self.__dict__["used_calls"] = set()

    def __getattr__(self, name):
        # [FUNCTION DOC] HELPER: Attribute lookup: returns own stored attributes
        # directly; otherwise records the name, reads it from the wrapped data and, for
        # callable non-device values, returns a function that routes the call through
        # __torch_function__ (adding self as an argument unless it is a classmethod or
        # staticmethod). Non-callables are rewrapped.
        if name in self.__dict__:
            return self.__dict__[name]
        self.used_attrs.add(name)

        val = getattr(self._data, name)

        # If it's a method
        if not isinstance(val, torch.device) and callable(val):
            c = getattr(type(self._data), name)
            # Don't append self to args if classmethod/staticmethod
            if c is val:
                return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=a, kwargs=kw))
            # Otherwise append self to args
            return lambda *a, **kw: wrap(self.__torch_function__(c, (Wrapper,), args=(self,) + a, kwargs=kw))

        return wrap(val)

    def __setattr__(self, name, value):
        # [FUNCTION DOC] HELPER: Attribute assignment: updates own stored attributes
        # if present, records the name, and always sets the unwrapped value on the
        # wrapped data as well.
        # KNOWN ISSUE: there is no return after updating own attributes, so internal
        # fields are also copied onto the wrapped data.
        if name in self.__dict__:
            self.__dict__[name] = value

        self.used_attrs.add(name)
        setattr(self._data, name, unwrap(value))

    def __setitem__(self, key, value):
        # [FUNCTION DOC] HELPER: Item assignment on the wrapped data after unwrapping
        # key and value.
        self._data[unwrap(key)] = unwrap(value)

    def __getitem__(self, key):
        # [FUNCTION DOC] HELPER: Item access on the wrapped data after unwrapping the
        # key; result is rewrapped.
        return wrap(self._data[unwrap(key)])

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Class-level override hook: gathers instances of the class among
        # the arguments (including inside sequences), raises AssertionError if none,
        # records the function in each one's used calls, unwraps arguments, calls the
        # function and rewraps the result.
        if kwargs is None:
            kwargs = {}
        # Find an instance of this class in the arguments
        args_of_this_cls = []
        for a in args:
            if isinstance(a, cls):
                args_of_this_cls.append(a)
            elif isinstance(a, collections.abc.Sequence):
                args_of_this_cls.extend(el for el in a if isinstance(el, cls))
        if len(args_of_this_cls) <= 0:
            raise AssertionError("expected args_of_this_cls to be non-empty")
        for a in args_of_this_cls:
            a.used_calls.add(func)
        args = unwrap(tuple(args))
        kwargs = {k: unwrap(v) for k, v in kwargs.items()}

        return wrap(func(*args, **kwargs))

    def __add__(self, other):
        # [FUNCTION DOC] HELPER: Addition routed through the override hook with torch
        # add.
        return self.__torch_function__(torch.add, (Wrapper,), (self, other))

    def __mul__(self, other):
        # [FUNCTION DOC] HELPER: Multiplication routed through the override hook with
        # torch mul.
        return self.__torch_function__(torch.mul, (Wrapper,), (self, other))

    def __sub__(self, other):
        # [FUNCTION DOC] HELPER: Subtraction routed through the override hook with torch
        # sub.
        return self.__torch_function__(torch.sub, (Wrapper,), (self, other))

    def __truediv__(self, other):
        # [FUNCTION DOC] HELPER: True division routed through the override hook with
        # torch true_divide.
        return self.__torch_function__(torch.true_divide, (Wrapper,), (self, other))

    def __floordiv__(self, other):
        # [FUNCTION DOC] HELPER: Floor division routed through the override hook with
        # torch floor_divide.
        return self.__torch_function__(torch.floor_divide, (Wrapper,), (self, other))

    def __ge__(self, other):
        # [FUNCTION DOC] HELPER: Greater-or-equal comparison routed through the override
        # hook with torch ge.
        return self.__torch_function__(torch.ge, (Wrapper,), (self, other))

    def __gt__(self, other):
        # [FUNCTION DOC] HELPER: Greater-than comparison routed through the override
        # hook with torch gt.
        return self.__torch_function__(torch.gt, (Wrapper,), (self, other))

    def __lt__(self, other):
        # [FUNCTION DOC] HELPER: Less-than comparison routed through the override hook
        # with torch lt.
        return self.__torch_function__(torch.lt, (Wrapper,), (self, other))

    def __le__(self, other):
        # [FUNCTION DOC] HELPER: Less-or-equal comparison routed through the override
        # hook with torch le.
        return self.__torch_function__(torch.le, (Wrapper,), (self, other))

    def __eq__(self, other):
        # [FUNCTION DOC] HELPER: Equality routed through the override hook with torch
        # eq, returning a wrapped tensor.
        return self.__torch_function__(torch.eq, (Wrapper,), (self, other))

    def __ne__(self, other):
        # [FUNCTION DOC] HELPER: Inequality routed through the override hook with torch
        # ne.
        return self.__torch_function__(torch.ne, (Wrapper,), (self, other))

    def __bool__(self):
        # [FUNCTION DOC] HELPER: Truth value routed through the override hook with the
        # Tensor bool method.
        return self.__torch_function__(torch.Tensor.__bool__, (Wrapper,), (self,))

    def __int__(self):
        # [FUNCTION DOC] HELPER: Integer conversion routed through the override hook
        # with the Tensor int method.
        return self.__torch_function__(torch.Tensor.__int__, (Wrapper,), (self,))

    def __len__(self):
        # [FUNCTION DOC] HELPER: Returns the length of the wrapped data.
        return len(self._data)


# unwrap inputs if necessary
def unwrap(v):
    # [FUNCTION DOC] HELPER: Recursively unwraps lists and tuples (preserving the container
    # type); returns the data of a Wrapper, or the value itself otherwise.
    if type(v) in {tuple, list}:
        return type(v)(unwrap(vi) for vi in v)

    return v._data if isinstance(v, Wrapper) else v

# wrap inputs if necessary
def wrap(v):
    # [FUNCTION DOC] HELPER: Recursively wraps lists and tuples (preserving the container
    # type); wraps Tensors in Wrapper, other values unchanged.
    if type(v) in {tuple, list}:
        return type(v)(wrap(vi) for vi in v)

    return Wrapper(v) if isinstance(v, torch.Tensor) else v

class TestEinsumOverride(TestCase):
    # [CLASS DOC] Regression tests for einsum with wrapper operands.
    "Regression test for gh-38479"
    def test_wrapper(self):
        # [FUNCTION DOC] Checks einsum of two wrapped tensors equals the outer product,
        # and that the older list-of-operands call form matches bilinear.
        x = Wrapper(torch.randn(5))
        y = Wrapper(torch.randn(4))
        self.assertEqual(torch.einsum('i,j->ij', x, y)._data,
                         torch.ger(x, y)._data)

        # in the old einsum interface, `operands` is a list
        a = Wrapper(torch.randn(2, 3))
        b = Wrapper(torch.randn(5, 3, 7))
        c = Wrapper(torch.randn(2, 7))
        self.assertEqual(torch.einsum('ik,jkl,il->ij', [a, b, c])._data,
                         torch.nn.functional.bilinear(a, c, b)._data)

class TestGradCheckOverride(TestCase):
    # [CLASS DOC] Tests that wrapped tensors work with gradient checking.
    "Test that wrappers work with gradcheck."
    def test_gradcheck(self):
        # [FUNCTION DOC] Runs gradcheck and gradgradcheck on wrapped double tensors in
        # fast and slow modes, then asserts the exact sets of attributes accessed and
        # torch functions called.
        from torch.testing._internal.common_utils import gradcheck, gradgradcheck

        def run_test(fast_mode):
            # [FUNCTION DOC] HELPER: Runs one gradcheck pass for the given fast-mode
            # flag and compares used attributes and calls against the expected sets
            # (with extra entries in fast mode).
            a = wrap(torch.tensor(5.0, dtype=torch.double))
            b = wrap(torch.tensor(6.0, dtype=torch.double))

            a.requires_grad = True
            b.requires_grad = True

            gradcheck(torch.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)
            gradgradcheck(torch.add, (a, b), raise_exception=False, check_batched_grad=False, fast_mode=fast_mode)

            total_used_attrs = a.used_attrs.union(b.used_attrs)
            total_used_calls = a.used_calls.union(b.used_calls)

            # These attributes (and the functions below) may change
            # if the gradcheck implementation changes. It's best to
            # aim for attributes that may be commonly present on other
            # Tensor-likes.
            expected_used_attrs = {
                'data',
                'dtype',
                'is_floating_point',
                'is_sparse',
                'layout',
                'new_zeros',
                'numel',
                'requires_grad',
                'requires_grad_',
                'size',
                'stride',
            }
            if fast_mode:
                expected_used_attrs.add('is_complex')
                expected_used_attrs.add('device')
            self.assertEqual(expected_used_attrs, total_used_attrs)

            expected_used_calls = {
                torch.Tensor.new_zeros,
                torch.Tensor.size,
                torch.Tensor.is_floating_point,
                torch.Tensor.numel,
                torch.Tensor.stride,
                torch.Tensor.requires_grad_,
                torch.autograd.grad,
                torch.add,
            }
            if fast_mode:
                expected_used_calls.add(torch.Tensor.is_complex)
            self.assertEqual(expected_used_calls, total_used_calls)
        run_test(fast_mode=True)
        run_test(fast_mode=False)

class TestNamedTuple(TestCase):
    # [CLASS DOC] Regression test for named tuple results with Tensor subclasses.
    """ Regression test for gh-47090 """
    def test_max(self):
        # [FUNCTION DOC] Checks that max along a dimension on a subclass instance
        # returns the same named tuple type and values as on a plain tensor.
        x = torch.tensor([1, 2])
        xs = x.as_subclass(SubTensor2)
        r = torch.max(x, dim=0)
        rs = torch.max(xs, dim=0)
        self.assertEqual(type(r), type(rs))
        self.assertEqual(r, rs)

class TestGradNewOnesOverride(TestCase):
    # [CLASS DOC] Regression test for new_ones on Tensor subclasses.
    """ Regression test for gh-47069 """
    def test_newones(self):
        # [FUNCTION DOC] Checks that new_ones on a subclass instance returns a subclass
        # instance.
        t = torch.tensor([1, 2]).as_subclass(SubTensor2)
        n = t.new_ones((1, 2))
        self.assertEqual(type(n), SubTensor2)

class TestPickle(TestCase):
    # [CLASS DOC] Regression test for pickling Tensor subclasses.
    "Regression test for gh-47051"
    def test_pickle(self):
        # [FUNCTION DOC] Checks that a subclass instance with an extra attribute round-
        # trips through pickle keeping type and attribute.
        t = torch.tensor([1]).as_subclass(SubTensor2)
        t.abcd = "e"
        t2 = pickle.loads(pickle.dumps(t))
        self.assertIs(type(t2), SubTensor2)
        self.assertEqual(t2.abcd, "e")

class TestBroadcastAllOverride(TestCase):
    # [CLASS DOC] Test for broadcasting helper support of wrapper objects.
    """ test for gh-37141 """
    def test_broadcast_all(self):
        # [FUNCTION DOC] Checks that broadcast_all returns wrapped outputs, with
        # broadcast values, when given wrapped or mixed wrapped and plain inputs.
        from torch.distributions.utils import broadcast_all
        a = torch.tensor([1.2, 3.4, 5.6])
        a_w = Wrapper(a)
        b = torch.tensor(5.0)
        b_w = Wrapper(b)
        c = torch.tensor([5.0, 5.0, 5.0])

        o_1 = broadcast_all(a_w, b_w)
        self.assertTrue(isinstance(o_1[0], Wrapper))
        self.assertTrue(isinstance(o_1[1], Wrapper))
        self.assertEqual(o_1[0]._data, a)
        self.assertEqual(o_1[1]._data, c)

        o_2 = broadcast_all(a_w, b)
        self.assertTrue(isinstance(o_2[0], Wrapper))
        self.assertTrue(isinstance(o_2[1], Wrapper))
        self.assertEqual(o_2[0]._data, a)
        self.assertEqual(o_2[1]._data, c)

class TestWrapTorchFunction(TestCase):
    # [CLASS DOC] Test for the decorator that adds override support to a function.
    def test_wrap_torch_function(self):
        # [FUNCTION DOC] Checks that a function decorated with a dispatcher returning
        # its tensor arguments calls the override of a non-Tensor argument and returns
        # its result.
        class A:
            # [CLASS DOC] HELPER: Fixture whose override hook returns minus one.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs):
                # [FUNCTION DOC] HELPER: Override hook returning minus one.
                return -1

        def dispatcher(a):
            # [FUNCTION DOC] HELPER: Dispatcher returning the single argument in a
            # tuple.
            return (a,)

        @torch.overrides.wrap_torch_function(dispatcher)
        def f(a):
            # [FUNCTION DOC] HELPER: Function wrapped with the override decorator;
            # returns its argument.
            return a

        self.assertEqual(f(A()), -1)

class TestIndexing(TestCase):
    # [CLASS DOC] Regression tests for override dispatch on indexing and index assignment.
    """ Regression tests for gh-46277 """
    def test_getitem(self):
        # [FUNCTION DOC] Checks indexing a tensor with a non-Tensor override object
        # returns the hook's result and leaves the tensor unchanged.
        class A:
            # [CLASS DOC] HELPER: Fixture whose override hook returns minus one.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook returning minus one.
                return -1

        t = torch.tensor([5])
        self.assertEqual(t[A()], -1)
        self.assertEqual(t, torch.tensor([5]))

    def test_getitem_subclass(self):
        # [FUNCTION DOC] Checks indexing a tensor with a Tensor-subclass override
        # object, alone or in a tuple, returns the hook's result and leaves the tensor
        # unchanged.
        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose override hook returns minus
            # one.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook returning minus one.
                return -1

        t = torch.tensor([5])
        self.assertEqual(t[A()], -1)
        self.assertEqual(t[5, A()], -1)
        self.assertEqual(t, torch.tensor([5]))

    def test_setitem(self):
        # [FUNCTION DOC] Checks that assigning with an override object as index (alone
        # or in a tuple) triggers the hook with the setitem method and leaves the tensor
        # unchanged.
        triggered = set()

        class A:
            # [CLASS DOC] HELPER: Fixture whose hook records called functions in a
            # shared set and returns minus one.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook adding the function to
                # the shared set and returning minus one.
                triggered.add(func)
                return -1

        t = torch.tensor([5])
        t[A()] = 1
        t[5, A()] = 1
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, torch.tensor([5]))

    def test_setitem_val(self):
        # [FUNCTION DOC] Checks that assigning an override object as the value triggers
        # the hook with the setitem method and leaves the tensor unchanged.
        triggered = set()

        class A:
            # [CLASS DOC] HELPER: Fixture whose hook records called functions in a
            # shared set and returns minus one.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook adding the function to
                # the shared set and returning minus one.
                triggered.add(func)
                return -1

        t = torch.tensor([5])
        t[0] = A()
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, torch.tensor([5]))

    def test_setitem_subclass(self):
        # [FUNCTION DOC] Checks setitem with a Tensor-subclass override object as index
        # triggers the hook and leaves the tensor unchanged.
        triggered = set()

        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose hook records called
            # functions in a shared set and returns minus one.
            @classmethod
            def __torch_function__(cls, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook adding the function to
                # the shared set and returning minus one.
                triggered.add(func)
                return -1

        t = torch.tensor([5])
        t[A()] = 1
        t[5, A()] = 1
        self.assertIn(Tensor.__setitem__, triggered)
        self.assertEqual(t, torch.tensor([5]))


class TestIterator(TestCase):
    # [CLASS DOC] Regression test for iterating over Tensor subclasses.
    # Regression test for gh-54457
    def test_iterator(self):
        # [FUNCTION DOC] Checks that iterating over a subclass instance yields subclass
        # instances.
        t = torch.tensor([5, 6, 7]).as_subclass(SubTensor2)
        it = iter(t)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)
        self.assertIs(type(next(it)), SubTensor2)


class TestRNN(TestCase):
    # [CLASS DOC] Regression test for recurrent networks with wrapper inputs.
    # Regression test for gh-55868
    def test_rnn(self):
        # [FUNCTION DOC] Runs an RNN module forward on a wrapped input and checks it
        # completes.
        model = torch.nn.RNN(10, 20, 2)
        input = Wrapper(torch.randn(1, 5, 10))
        model(input)


class TestDisabledTorchFunction(TestCase):
    # [CLASS DOC] Regression test for parameters not blocking override dispatch.
    # Regression test for gh-64687
    def test_parameter_does_not_prevent_dispatch(self):
        # [FUNCTION DOC] Checks that mixing a Parameter and an override object in add
        # and in linear (in either position) dispatches to the override object.
        class MyTensor:
            # [CLASS DOC] HELPER: Fixture whose hook returns a marker string.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook returning a marker
                # string.
                return "called"

        t1 = MyTensor()
        t2 = torch.nn.Parameter(torch.rand(2, 2))
        self.assertEqual(torch.add(t2, t1), "called")

        inp = torch.rand(10, 10)
        self.assertEqual(torch.nn.functional.linear(inp, t1, t2), "called")
        self.assertEqual(torch.nn.functional.linear(inp, t2, t1), "called")

class TestResolveName(TestCase):
    # [CLASS DOC] Tests for name resolution of overridable functions.
    def test_resolve_name(self):
        # [FUNCTION DOC] For every overridable function, evaluates the resolved dotted
        # name and checks it yields the same function.
        for cs in get_overridable_functions().values():
            for c in cs:
                self.assertEqual(
                    eval(torch.overrides.resolve_name(c)),
                    c,
                    msg=lambda msg: f"{msg}\n{c}, {torch.overrides.resolve_name(c)}"
                )

class TestTorchFunctionWarning(TestCase):
    # [CLASS DOC] Tests that override hooks run without warnings on standalone classes and
    # Tensor subclasses.
    def test_torch_function_standalone_class(self):
        # [FUNCTION DOC] Checks dropout and abs on a non-Tensor override object return
        # the hook's constant tensor.
        class StandaloneTorchFunctionClass:
            # [CLASS DOC] HELPER: Fixture whose hook returns a constant tensor.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook returning a constant
                # tensor regardless of inputs.
                # Return a simple tensor for testing
                return torch.tensor(42.0)
        a = StandaloneTorchFunctionClass()
        # Test that torch_function works without warnings
        result1 = torch.nn.functional.dropout(a)
        result2 = torch.abs(a)
        self.assertEqual(result1, torch.tensor(42.0))
        self.assertEqual(result2, torch.tensor(42.0))

    def test_torch_function_tensor_subclass(self):
        # [FUNCTION DOC] Checks dropout and abs on a Tensor-subclass override object
        # return the hook's constant tensor.
        class TensorSubclassTorchFunctionClass(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass whose hook returns a constant
            # tensor.
            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook returning a constant
                # tensor regardless of inputs.
                # Return a simple tensor for testing
                return torch.tensor(99.0)
        b = TensorSubclassTorchFunctionClass()
        # Test that torch_function works without warnings
        result1 = torch.nn.functional.dropout(b)
        result2 = torch.abs(b)
        self.assertEqual(result1, torch.tensor(99.0))
        self.assertEqual(result2, torch.tensor(99.0))

class TestDisabledUserWarnings(TestCase):
    # [CLASS DOC] Tests that override-registry helper queries emit no user warnings.
    def test_no_implicit_user_warning_for_deprecated_functions(self):
        # [FUNCTION DOC] Asserts no warning is emitted by queries for ignored functions,
        # testing overrides, overridable functions, name resolution and the Tensor
        # method check.
        self.assertNotWarn(get_ignored_functions)
        self.assertNotWarn(get_testing_overrides)
        self.assertNotWarn(get_overridable_functions)
        self.assertNotWarn(lambda: resolve_name(torch.Tensor.add))
        self.assertNotWarn(lambda: is_tensor_method_or_property(torch.Tensor.add))

@unittest.skipIf(TEST_WITH_CROSSREF, "not run with crossref")
class TestTorchFunctionMode(TestCase):
    # [CLASS DOC] Tests for torch function modes; skipped under the cross-reference test
    # configuration.
    def test_basic(self):
        # [FUNCTION DOC] Checks that a mode returning minus one intercepts factory
        # functions, tensor ops, python-side functions and the instrumented bar function
        # within its context.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode whose hook returns minus one for everything.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Mode hook returning minus one.
                return -1
        # NB: factory functions get overridden too!
        x = torch.randn(1)
        with A():
            self.assertEqual(torch.randn(3), -1)
            self.assertEqual(torch.add(x, x), -1)
            self.assertEqual(torch.split(None, [2]), -1)  # python side
            self.assertEqual(bar(x), -1)

    def test_factory_override(self):
        # [FUNCTION DOC] Checks that a mode intercepts tensor factory functions,
        # including sparse constructors and as_tensor.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode whose hook returns minus one.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Mode hook returning minus one.
                return -1

        with A():
            self.assertEqual(torch.tensor([1]), -1)
            self.assertEqual(torch.sparse_coo_tensor(1, 1, 1), -1)
            self.assertEqual(torch.sparse_csr_tensor(1, 1, 1), -1)
            self.assertEqual(torch.sparse_coo_tensor(1, 1, (1, 1), check_invariants=False), -1)
            self.assertEqual(torch.sparse_csr_tensor(1, 1, 1, (1, 1), check_invariants=False), -1)
            self.assertEqual(torch.as_tensor([1]), -1)

    def test_modes_handle_first(self):
        # [FUNCTION DOC] Checks that an active mode is consulted before a subclass's own
        # override.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode whose hook returns minus forty.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Mode hook returning minus forty.
                return -40

        x = SubTensor()
        with A():
            self.assertEqual(torch.neg(x), -40)
            self.assertEqual(torch.mean(x), -40)
            self.assertEqual(torch.mm(x, x), -40)
            self.assertEqual(bar(x), -40)

    def test_modes_return_notimplemented(self):
        # [FUNCTION DOC] Checks that when a mode returns NotImplemented, dispatch falls
        # back to the subclass's override, and a function with no override raises
        # TypeError naming the subclass.
        class MyMode(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode whose hook always returns NotImplemented.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Mode hook returning NotImplemented.
                return NotImplemented

        x = SubTensor()
        with MyMode():
            self.assertEqual(torch.mean(x), 0)
            self.assertEqual(torch.mm(x, x), -1)
            self.assertEqual(bar(x), 1)
            self.assertRaisesRegex(
                TypeError, r'SubTensor',
                lambda: self.assertEqual(torch.max(x, x)))

    def test_with_mode(self):
        # [FUNCTION DOC] Checks that an error raised from a mode hook propagates out of
        # the with block.
        class ErrorA(RuntimeError):
            # [CLASS DOC] HELPER: Error type raised by the mode.
            pass

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode whose hook raises the local error.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Mode hook that raises the local error.
                raise ErrorA

        with self.assertRaises(ErrorA):
            with A():
                torch.empty([])

    def test_with_mode_created_separately(self):
        # [FUNCTION DOC] Same as the previous check but the mode is created before
        # entering the block.
        class ErrorA(RuntimeError):
            # [CLASS DOC] HELPER: Error type raised by the mode.
            pass

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode whose hook raises the local error.
            def __torch_function__(self, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Mode hook that raises the local error.
                raise ErrorA

        x = A()
        with self.assertRaises(ErrorA):
            with x:
                torch.empty([])

    def test_with_nested_modes(self):
        # [FUNCTION DOC] Checks nested modes run innermost first, each forwarding to the
        # next, recording their messages in order.
        out = []

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that records a message in a shared list and
            # forwards the call.
            def __init__(self, msg):
                # [FUNCTION DOC] HELPER: Stores the message string.
                self.msg = msg

            def __torch_function__(self, func, _, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook appending the message to the
                # shared list and calling the function.
                if kwargs is None:
                    kwargs = {}
                out.append(self.msg)
                return func(*args, **kwargs)

        with A("layer1"):
            with A("layer2"):
                torch.empty([])

        self.assertEqual(out, ["layer2", "layer1"])

    def test_nested_same_mode(self):
        # [FUNCTION DOC] Checks that entering the same mode object twice in nested
        # fashion runs it twice per call.
        out = []

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that records a message in a shared list and
            # forwards the call.
            def __init__(self, msg):
                # [FUNCTION DOC] HELPER: Stores the message string.
                self.msg = msg

            def __torch_function__(self, func, _, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook appending the message to the
                # shared list and calling the function.
                if kwargs is None:
                    kwargs = {}
                out.append(self.msg)
                return func(*args, **kwargs)

        with A("layer1") as a:
            with a:
                torch.empty([])

        self.assertEqual(out, ["layer1", "layer1"])

    def test_error_using_class_method_on_mode(self):
        # [FUNCTION DOC] Checks that defining the mode hook as a class method raises
        # RuntimeError saying class methods are unsupported when the mode is used.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode with an invalid class-method hook.
            @classmethod
            def __torch_function__(cls, func, _, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Invalid class-method hook that forwards
                # the call.
                return func(args, kwargs)

        x = torch.tensor(5.)
        with self.assertRaisesRegex(RuntimeError, "classmethod is not supported, please make it a plain method"):
            with A():
                x + x

    def test_restacking_with_ancestor(self):
        # [FUNCTION DOC] Checks a mode that was entered nested inside another instance
        # of its class can be re-entered afterward.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode with no overrides.
            pass

        with A():
            with A() as x:
                pass

        with x:
            pass

    def test_get_cur_mode(self):
        # [FUNCTION DOC] Checks the current-mode query returns the entered mode,
        # including when re-entering a previously used mode and nesting a new one.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode defining only a dispatch hook that does
            # nothing.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that does nothing.
                # KNOWN ISSUE: defines the torch dispatch hook, not the torch
                # function hook, so it is never used by function modes.
                pass

        with A() as mode1:
            self.assertEqual(_get_current_function_mode(), mode1)

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_function_mode(), mode2)


    def test_get_mode_stack(self):
        # [FUNCTION DOC] Checks the mode stack query is empty initially and lists
        # entered modes from outermost to innermost.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode defining only a dispatch hook that does
            # nothing.
            def __torch_dispatch__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Dispatch hook that does nothing.
                # KNOWN ISSUE: defines the torch dispatch hook, not the torch
                # function hook, so it is never used by function modes.
                pass

        self.assertEqual(_get_current_function_mode_stack(), [])

        with A() as mode1:
            self.assertEqual(_get_current_function_mode_stack(), [mode1])

        with mode1:
            with A() as mode2:
                self.assertEqual(_get_current_function_mode_stack(), [mode1, mode2])

    def test_all_same_mode(self):
        # [FUNCTION DOC] Checks the all-same-mode helper is true for repeated identical
        # mode, false with a missing entry or different mode instances.
        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode with no overrides.
            pass

        x = A()
        y = A()
        self.assertTrue(all_same_mode([x, x, x]))
        self.assertFalse(all_same_mode([x, None]))
        self.assertFalse(all_same_mode([x, y]))

    def test_nested_modes_with_python_has_torch_function(self):
        # [FUNCTION DOC] Checks nested modes both run, innermost first, for a python
        # function that checks for overrides, and that the result is the unchanged
        # input.
        called = []

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that appends A to the shared list and
            # forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook recording the marker and
                # calling the function.
                called.append("A")
                kwargs = {} if kwargs is None else kwargs
                return func(*args, **kwargs)

        class B(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that appends B to the shared list and
            # forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook recording the marker and
                # calling the function.
                called.append("B")
                kwargs = {} if kwargs is None else kwargs
                return func(*args, **kwargs)

        x = torch.randn(3, 4)
        with A():
            with B():
                y = bar(x)

        self.assertEqual(y, x)
        self.assertEqual(called, ["B", "A"])


    def test_reentrant_mode_idiom(self):
        # [FUNCTION DOC] Checks a mode that re-enters itself while handling a
        # subtraction can see the add it issues itself, so the log holds sub then add.
        log = []

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode logging every function; for sub it re-enters
            # itself and issues add with a negative alpha instead.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook: logs the function; for sub,
                # re-enters the mode, asserts no keyword arguments (raising
                # AssertionError otherwise) and returns an add with alpha minus
                # one; otherwise forwards the call.
                if kwargs is None:
                    kwargs = {}
                log.append(func)
                if func is torch.sub:
                    with self:
                        input, other = args
                        if kwargs:
                            raise AssertionError(f"expected kwargs to be empty, got {kwargs}")
                        return torch.add(input, other, alpha=-1)
                return func(*args, **kwargs)

        x = torch.randn(1)
        y = torch.randn(1)
        with A():
            torch.sub(x, y)
        # add hits the torch function again!
        self.assertEqual(log, [torch.sub, torch.add])

    def test_nn_parse_to(self):
        # [FUNCTION DOC] Checks the mode intercepts the internal argument parser for to,
        # regardless of its name.
        # This failed because the parser thinks the function is called to()
        # but it's actually called _parse_to()

        called = False

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that sets a flag and forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook setting the enclosing called
                # flag and forwarding the call.
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        with A():
            torch._C._nn._parse_to('cpu')

        self.assertTrue(called)

    @skipIfTorchDynamo(msg="https://github.com/pytorch/pytorch/issues/162586")
    def test_getitem_call(self):
        # [FUNCTION DOC] Checks tensor indexing by a tensor goes through an active
        # mode; skipped under Dynamo for a known issue.
        # KNOWN ISSUE: the explanatory comment was copied from the parse-to test and
        # does not describe indexing.
        # This failed because the parser thinks the function is called to()
        # but it's actually called _parse_to()

        called = False

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that sets a flag and forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook setting the enclosing called
                # flag and forwarding the call.
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        a = torch.zeros(5)
        b = torch.tensor(0)
        with A():
            a[b]

        self.assertTrue(called)


    def test_distributions_bernoulli(self):
        # [FUNCTION DOC] Checks constructing a Bernoulli distribution under a mode
        # invokes the mode (regression for wrong override checks in broadcasting).
        # This failed because improper use of has_torch_function when
        # is_tensor_like should have been used instead, inside the
        # broadcasting logic called by distributions (Bernoulli doesn't
        # matter per se)

        called = False

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that sets a flag and forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook setting the enclosing called
                # flag and forwarding the call.
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        with A():
            torch.distributions.Bernoulli(0.3)

        self.assertTrue(called)

    def test_mode_notimplemented_loop(self):
        # [FUNCTION DOC] Checks that a mode returning NotImplemented for subclass types
        # is called twice (once with the subclass, once as plain tensor) for both a
        # torch op and bar, and the result keeps the subclass.
        # Default tensor subclass implementation disables torch function;
        # when we redispatch to mode we must not treat the objects as
        # eligible

        called = 0

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that counts calls; returns NotImplemented if
            # any type is not the plain Tensor, else forwards.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook incrementing the enclosing
                # counter, returning NotImplemented when any type differs from
                # Tensor, otherwise forwarding the call.
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called += 1
                # The first time we call, the mode sees an active type that
                # it doesn't know how to deal with.  The second time, we're
                # instructed to treat it "as if it were a tensor", and so
                # we keep going.  I'm not entirely clear if the subclasses
                # disappearing from types is the correct way to do it.
                if any(t is not torch.Tensor for t in types):
                    return NotImplemented
                else:
                    return func(*args, **kwargs)

        class B(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        b = B()

        with A():
            r = torch.neg(b)

        self.assertIs(type(r), B)
        self.assertEqual(called, 2)

        called = 0

        with A():
            r = bar(b)

        self.assertIs(type(r), B)
        self.assertEqual(called, 2)

    def test_disable_subclass_not_mode(self):
        # [FUNCTION DOC] Checks that disabling only subclass overrides still lets an
        # active mode run, and the result is not the subclass.
        called = False

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that sets a flag and forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook setting the enclosing called
                # flag and forwarding the call.
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        class B(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        x = B(torch.randn(5))
        with A():
            with torch._C.DisableTorchFunctionSubclass():
                self.assertNotIsInstance(torch.sum(x), B)

        self.assertTrue(called)

    def test_disable_subclass_mode(self):
        # [FUNCTION DOC] Checks that fully disabling torch function bypasses both an
        # active mode and subclass overrides.
        called = False

        class A(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that sets a flag and forwards the call.
            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook setting the enclosing called
                # flag and forwarding the call.
                nonlocal called
                if kwargs is None:
                    kwargs = {}
                called = True
                return func(*args, **kwargs)

        class B(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        x = B(torch.randn(5))
        with A():
            with torch._C.DisableTorchFunction():
                self.assertNotIsInstance(torch.sum(x), B)

        self.assertFalse(called)

    @unittest.skipIf(TEST_WITH_TORCHDYNAMO, "https://github.com/pytorch/pytorch/issues/182318")
    def test_disable_enable_subclass(self):
        # [FUNCTION DOC] Checks that re-enabling torch function inside a subclass-only
        # disable region restores subclass results; skipped under Dynamo for a known
        # issue.
        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        x = A(torch.randn(5))
        with torch._C.DisableTorchFunctionSubclass():
            g = torch._C._EnableTorchFunction()
            try:
                self.assertIsInstance(torch.sum(x), A)
            finally:
                del g

    def test_disable_enable_torch_function_ctx(self):
        # [FUNCTION DOC] Checks that the enabling context manager inside full disable
        # restores subclass results.
        class A(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        x = A(torch.randn(5))
        with torch._C.DisableTorchFunction():
            with torch.overrides._enable_torch_function():
                self.assertIsInstance(torch.sum(x), A)

    def test_torch_function_all_disabled_api(self):
        # [FUNCTION DOC] Checks the all-disabled query is false normally, true inside
        # full disable, and false inside subclass-only disable.
        from torch._C import _is_torch_function_all_disabled

        state = _is_torch_function_all_disabled()
        self.assertFalse(state)

        with torch._C.DisableTorchFunction():
            state = _is_torch_function_all_disabled()
            self.assertTrue(state)

        state = _is_torch_function_all_disabled()
        self.assertFalse(state)

        with torch._C.DisableTorchFunctionSubclass():
            state = _is_torch_function_all_disabled()
            self.assertFalse(state)


    def test_subclass_hash(self):
        # [FUNCTION DOC] Checks a subclass defining an override hook that densifies
        # diagonal operands still hashes by identity (two instances can be added to a
        # set) and supports arithmetic.
        class DiagTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Tensor subclass storing a diagonal vector and
            # expanding it to a full matrix inside its override hook.
            def __init__(self, diag):
                # [FUNCTION DOC] HELPER: Stores the diagonal vector on the
                # instance.
                self._diag = diag

            @classmethod
            def __torch_function__(cls, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Override hook converting any such
                # argument in positional and keyword args into a full diagonal
                # matrix before calling the function.
                kwargs = kwargs or {}

                def get_full_matrices(t):
                    # [FUNCTION DOC] HELPER: Returns a full diagonal matrix
                    # for instances of the subclass, else the input
                    # unchanged.
                    if isinstance(t, DiagTensor):
                        return torch.diag_embed(t._diag)
                    else:
                        return t

                return func(*tree_map(get_full_matrices, args), **tree_map(get_full_matrices, kwargs))

        d = torch.rand(2)
        a = DiagTensor(d)

        self.assertEqual((a + 1), torch.diag_embed(d) + 1)

        # If the hash function was returning the same value, this would
        # fail inside `Tensor.__eq__`.
        # If __hash__ was going through torch_function, the implementation above would
        # be wrong as it would compute the hash on a temporary Tensor thus not ensuring
        # the uniqueness of the hash that we rely on for Tensors.
        s = set()
        s.add(a)
        s.add(DiagTensor(d))

    def test_custom_device_type(self):
        # [FUNCTION DOC] Checks a mode that rewrites integer device specifications into
        # an xla device type, for both positional and keyword forms.
        class CustomDeviceContext(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode that rewrites device construction with an
            # integer index to an xla device.

            def __torch_function__(self, func, types, args=(), kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook: for device construction with
                # an integer positional or keyword device, replaces it with an
                # xla device, then calls the function.
                kwargs = kwargs or {}
                if func == torch.device:
                    if args and isinstance(args[0], int):
                        args = ("xla", args[0])
                    elif isinstance(kwargs.get('device'), int):
                        kwargs['device'] = f"xla:{kwargs.get('device')}"
                return func(*args, **kwargs)

        with CustomDeviceContext():
            d_args = torch.device(0)
            self.assertEqual(d_args.type, "xla")
            self.assertEqual(d_args.index, 0)
            d_kwargs = torch.device(device=0)
            self.assertEqual(d_kwargs.type, "xla")
            self.assertEqual(d_kwargs.index, 0)

    def test_device_context_semantics(self):
        # [FUNCTION DOC] Checks that setting the default device installs a device
        # context at the bottom of the function mode stack and that it persists after a
        # base mode exits. Sets the default device to cuda, then restores it to none in
        # a finally block.
        from torch._C import _len_torch_function_stack
        from torch.utils._device import DeviceContext
        try:
            torch.set_default_device("cuda")

            def get_stack():
                # [FUNCTION DOC] HELPER: Returns the list of function modes
                # currently on the stack, from bottom to top.
                return [torch._C._get_function_stack_at(i) for i in range(_len_torch_function_stack())]

            base_mode = BaseTorchFunctionMode()
            with base_mode:
                torch.set_default_device("cpu")
                stack = get_stack()
                self.assertIsInstance(stack[0], DeviceContext)
                self.assertEqual(stack[0].device, torch.device("cpu"))

            stack = get_stack()
            self.assertIsInstance(stack[0], DeviceContext)
            self.assertEqual(stack[0].device, torch.device("cpu"))
        finally:
            torch.set_default_device(None)



class TestTorchFunctionRedispatch(TestCase):
    # [CLASS DOC] Tests for redispatch of torch function: skipping the next hook once, with
    # tensor subclasses and with modes. Setup and teardown verify the skip-next thread-local
    # flag is clear.
    def setUp(self):
        # [FUNCTION DOC] Calls the base setup then asserts the skip-next flag is not set
        # at test start.
        super().setUp()
        self.assertFalse(
            torch._C._peek_should_skip_torch_function(),
            "skip_next TLS was set at test start",
        )

    def tearDown(self):
        # [FUNCTION DOC] Reads the skip-next flag; if leaked, clears it, calls the base
        # teardown, then fails if it had leaked.
        leaked = torch._C._peek_should_skip_torch_function()
        if leaked:
            torch._C._set_skip_next_torch_function(False)
        super().tearDown()
        self.assertFalse(leaked, "skip_next TLS leaked from test")

    @staticmethod
    def _filter_log(call_log, allowed_qualnames):
        # [FUNCTION DOC] Static helper returning the log entries whose first element is
        # in the allowed set of qualified names.
        return [e for e in call_log if e[0] in allowed_qualnames]

    def test_simple(self):
        # [FUNCTION DOC] Checks calling bar with a logging redispatch tensor returns the
        # same object and logs exactly one bar entry.
        call_log = []
        x = RedispatchTensor(torch.ones(1), call_log=call_log)
        ret = bar(x)
        self.assertIs(ret, x)
        filtered = self._filter_log(call_log, {"bar"})
        call_log_str = '\n'.join(f"{entry[0]}" for entry in filtered)
        self.assertExpectedInline(call_log_str, """bar""")

    def test_skip_to_inner(self):
        # [FUNCTION DOC] Checks a redispatching tensor skips the hook once for foo, then
        # the adds inside foo dispatch again; compares the filtered log with an expected
        # snapshot and the result with the sum.
        call_log = []
        x = RedispatchTensor(torch.full((1,), 1), call_log=call_log)
        y = RedispatchTensor(torch.full((1,), 2), call_log=call_log)
        z = RedispatchTensor(torch.full((1,), 3), call_log=call_log)
        ret = foo(x, y, z)

        # Key behavior: redispatch skips dispatch for foo once,
        # but then the + operations inside foo DO dispatch to __torch_function__
        # So we should see: foo, then add (from a+b), then add (from temp+c)
        # Snapshot the log before assertEqual triggers more __torch_function__ calls.
        filtered = self._filter_log(call_log, {"foo", "TensorBase.add"})
        call_log_str = '\n'.join(f"{entry[0]}: {entry[1]}" for entry in filtered)
        self.assertEqual(ret, torch.full((1,), 6))
        self.assertExpectedInline(call_log_str, """\
foo: (<class 'torch.testing._internal.common_subclass.RedispatchTensor'>,)
TensorBase.add: (<class 'torch.testing._internal.common_subclass.RedispatchTensor'>,)
TensorBase.add: (<class 'torch.testing._internal.common_subclass.RedispatchTensor'>,)""")

    def test_mode_with_redispatch(self):
        # [FUNCTION DOC] Checks that a mode redispatching without re-entering itself
        # only sees the outer foo call, not the inner adds.
        call_log = []

        class LoggingMode(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode logging function names and redispatching.
            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook appending the function name
                # to the shared log and redispatching the call.
                call_log.append(func.__name__)
                return redispatch_function(func, types, args, kwargs)

        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        z = torch.tensor([3.0])

        with LoggingMode():
            ret = foo(x, y, z)

        self.assertEqual(ret, torch.tensor([6.0]))
        # Without 'with self:', mode only sees the outer call
        filtered = [n for n in call_log if n in ("foo", "add")]
        self.assertEqual(filtered, ['foo'])

    def test_mode_with_redispatch_reentrant(self):
        # [FUNCTION DOC] Checks that a mode that re-enters itself around redispatch also
        # sees the inner add calls.
        call_log = []

        class LoggingMode(TorchFunctionMode):
            # [CLASS DOC] HELPER: Mode logging function names and redispatching
            # inside a re-entered context.
            def __torch_function__(self, func, types, args, kwargs=None):
                # [FUNCTION DOC] HELPER: Mode hook appending the function name,
                # re-entering the mode and redispatching.
                call_log.append(func.__name__)
                # Re-enable mode for inner calls
                with self:
                    return redispatch_function(func, types, args, kwargs)

        x = torch.tensor([1.0])
        y = torch.tensor([2.0])
        z = torch.tensor([3.0])

        with LoggingMode():
            ret = foo(x, y, z)

        self.assertEqual(ret, torch.tensor([6.0]))
        # With 'with self:', mode sees outer call and inner add operations
        filtered = [n for n in call_log if n in ("foo", "add")]
        self.assertEqual(filtered, ['foo', 'add', 'add'])


class TestTorchFunctionRedispatchOps(TestCase):
    # [CLASS DOC] Device-generic tests running each registered operator through the
    # redispatch tensor wrapper.
    @ops(op_db)
    def test_redispatch(self, device, dtype, op):
        # [FUNCTION DOC] For each sample input of the operator, runs the op on plain
        # clones and on a clone with the first input wrapped in a redispatching tensor,
        # then compares results with torch function disabled. Skips one conv transpose
        # case for low precision dtypes and ops with nondeterministic output.
        if op.name == "nn.functional.conv_transpose3d" and dtype in (
            torch.float16,
            torch.bfloat16,
            torch.complex32,
        ):
            # Disabled due to CI failures; see #190241
            self.skipTest("disabled due to CI failures; see #190241")
        if op.has_nondeterministic_output:
            self.skipTest("output is nondeterministic; not comparable across calls")

        def clone_and_wrap(x):
            # [FUNCTION DOC] HELPER: Detaches a tensor, clones it preserving strides
            # when strided, and wraps it in the redispatching tensor; other values
            # unchanged.
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.layout == torch.strided:
                    x = clone_preserve_strides_offset(x)
                else:
                    x = clone_input(x)
                return RedispatchTensor(x)
            return x

        def clone_only(x):
            # [FUNCTION DOC] HELPER: Detaches a tensor and clones it preserving
            # strides when strided, without wrapping; other values unchanged.
            if isinstance(x, torch.Tensor):
                x = x.detach()
                if x.layout == torch.strided:
                    return clone_preserve_strides_offset(x)
                return clone_input(x)
            return x

        for sample in op.sample_inputs(device=device, dtype=dtype):
            # Wrap only sample.input in RedispatchTensor so
            # __torch_function__ fires. Clone args/kwargs without wrapping
            # because some ops pass auxiliary tensors (e.g. spacing, bins)
            # through C++ arg parsing that rejects subclasses.
            wrapped = SampleInput(
                clone_and_wrap(sample.input),
                args=tree_map(clone_only, sample.args),
                kwargs=tree_map(clone_only, sample.kwargs),
            )

            expect = op(sample.input, *sample.args, **sample.kwargs)
            actual = op(wrapped.input, *wrapped.args, **wrapped.kwargs)

            with torch._C.DisableTorchFunction():
                self.assertEqual(expect, actual)


instantiate_device_type_tests(TestTorchFunctionRedispatchOps, globals())


if __name__ == '__main__':
    run_tests()
