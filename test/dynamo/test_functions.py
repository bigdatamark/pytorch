# Owner(s): ["module: dynamo"]
# flake8: noqa: E731, C405, F811, C418, C417
# [MODULE DOC] Dynamo tests for how torch.compile traces plain Python functions, builtins,
# itertools/functools/operator helpers, containers, closures and tensor operations.
# Most tests are produced by the make_test factory, which wraps a bare function into a
# test method comparing eager and compiled results with a frame-count check. Module level
# helper functions, classes and globals (d, e, flag, clip01, ...) are the inputs those tests
# trace. Importing the module only defines these objects and builds the device_type string
# from the current accelerator (cpu if none); it also star-imports the triton test kernels.
import collections
import collections.abc
import contextlib
import enum
import functools
import inspect
import itertools
import keyword
import math
import operator
import random
import sys
import types
import typing
import unittest
from dataclasses import dataclass, field
from typing import Any, Generic, TypeVar
from typing_extensions import NamedTuple
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo.test_case
import torch._dynamo.testing
from torch import sub
from torch._dynamo.exc import Unsupported
from torch._dynamo.testing import (
    CompileCounterWithBackend,
    EagerAndRecordGraphs,
    normalize_gm,
)
from torch._dynamo.utils import counters, ifdynstaticdefault, range_iterator, same
from torch._dynamo.variables import ConstantVariable, SkipFunctionVariable
from torch._dynamo.variables.lists import RangeVariable
from torch.nn import functional as F
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)
from torch.testing._internal.inductor_utils import HAS_GPU

# Defines all the kernels for tests
from torch.testing._internal.triton_utils import *  # noqa: F403


device_type = (
    acc.type if (acc := torch.accelerator.current_accelerator(True)) else "cpu"
)

T = TypeVar("T")

d = torch.ones(10, 10)
e = torch.nn.Linear(10, 10)
flag = True


class CustomDictSubclass(collections.OrderedDict):
    # [CLASS DOC] HELPER: Empty OrderedDict subclass used to test tracing of dict subclasses.
    pass


clip01 = functools.partial(torch.clip, min=0.0, max=1.0)


def constant3(a, b):
    # [FUNCTION DOC] HELPER: Return a minus b plus the constant 3.0 (folded sum 1.0 plus 2).
    return a - b + (1.0 + 2)


def call(f, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Generic trampoline that calls f with the given positional and keyword
    # arguments and returns its result.
    return f(*args, **kwargs)


_variable = 0


def update_global(x):
    # [FUNCTION DOC] HELPER: Increment the module global _variable by one (side effect) and
    # return x multiplied by the updated value, so tests can check Dynamo sees global updates.
    global _variable
    _variable += 1
    # Check that updated global variable value is picked up
    return x * _variable


def pos_only_fn(*args, **kwargs):
    # [FUNCTION DOC] HELPER: Forward all positional and keyword arguments to _pos_only_fn and
    # return its result.
    return _pos_only_fn(*args, **kwargs)


def _pos_only_fn(a, b=3, /, **kwargs):
    # [FUNCTION DOC] HELPER: Function with positional-only a and b (b defaults to 3) plus extra
    # keyword arguments. Returns a tuple: a times b plus the product of keyword "a" (default
    # -13) and keyword "b" (default 42), then two booleans saying whether "a" and "b" were
    # passed as keywords.
    return (
        a * b + kwargs.get("a", -13) * kwargs.get("b", 42),
        "a" in kwargs,
        "b" in kwargs,
    )


@contextlib.contextmanager
def update_global_ctx(x):
    # [FUNCTION DOC] HELPER: Context manager (via contextmanager) that yields the result of
    # update_global applied to x, so entering it bumps the global counter; the finally clause
    # does nothing.
    try:
        yield update_global(x)
    finally:
        pass


def func_with_default(a, b, some_default_arg=True):
    # [FUNCTION DOC] HELPER: Return a minus b when some_default_arg is truthy; otherwise falls
    # through and returns None.
    if some_default_arg:
        return a - b


def make_test(fn=None, expected_frame_count=1):
    # [FUNCTION DOC] HELPER: Test factory usable bare or as a decorator with arguments. With no fn
    # it returns a decorator that remembers expected_frame_count. Otherwise it counts the
    # parameters of fn and returns a test method that runs the standard Dynamo test helper on fn,
    # checking compiled vs eager results and the expected frame count.
    if fn is None:
        return lambda fn: make_test(fn, expected_frame_count=expected_frame_count)

    nargs = len(inspect.signature(fn).parameters)

    def test_fn(self):
        # [FUNCTION DOC] HELPER: The generated test method; delegates to the standard_test helper
        # with the captured fn, parameter count and expected frame count, and returns its result.
        return torch._dynamo.testing.standard_test(
            self,
            fn=fn,
            nargs=nargs,
            expected_frame_count=expected_frame_count,
        )

    return test_fn


class MyCls:
    # [CLASS DOC] HELPER: Trivial class with one class attribute a equal to 1; used for object and
    # class identity/equality tests.
    a = 1


@torch.jit.script_if_tracing
def inline_script_if_tracing(x):
    # [FUNCTION DOC] HELPER: Function marked script_if_tracing; returns x plus 1.2.
    return x + 1.2


@torch.jit.ignore
def inline_ignore(x):
    # [FUNCTION DOC] HELPER: Function marked jit.ignore; returns x plus 3.4.
    return x + 3.4


@torch.jit.unused
def inline_unused(x):
    # [FUNCTION DOC] HELPER: Function marked jit.unused; returns x plus 5.6.
    return x + 5.6


@functools.lru_cache
def inline_lru_cache_fn_with_default_args(x, y, _=None):
    # [FUNCTION DOC] HELPER: lru_cache-wrapped function returning the sine of x times y; the third
    # argument is accepted but ignored.
    return torch.sin(x * y)


@torch.jit.script_if_tracing
def inline_script_if_tracing_fn_with_default_args(x, y, c=1.2):
    # [FUNCTION DOC] HELPER: script_if_tracing function returning the cosine of x times y plus c
    # (default 1.2).
    return torch.cos(x * y) + c


class FunctionTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Main Dynamo test case for function tracing. Most methods are built by make_test
    # from a bare function, so each one traces a small snippet and compares eager and compiled
    # output; other methods compile explicitly and assert on results, errors or counters.
    @make_test
    def test_inline_jit_annotations(x):
        # [FUNCTION DOC] Calls three jit-annotated helpers (script_if_tracing, ignore, unused) in
        # sequence and returns None; checks Dynamo can inline them.
        x = inline_script_if_tracing(x)
        x = inline_ignore(x)
        x = inline_unused(x)
        return

    @make_test
    def test_inline_script_if_tracing_fn_with_default_args(a, b):
        # [FUNCTION DOC] Calls a script_if_tracing function that has a default argument, passing
        # two tensors.
        return inline_script_if_tracing_fn_with_default_args(a, b)

    @make_test
    def test_inline_lru_cache_fn_with_default_args(a, b):
        # [FUNCTION DOC] Calls an lru_cache function with a tensor, the int 2 and a tensor for the
        # defaulted parameter.
        return inline_lru_cache_fn_with_default_args(a, 2, b)

    def test_str_split_returns_mutable_list(self):
        # [FUNCTION DOC] Compiles with fullgraph a function using string split, rsplit and
        # splitlines then mutating the resulting lists (sort, append, reverse, pop). Asserts
        # compiled outputs equal eager outputs, so the lists are tracked as fresh mutable lists.
        # str.split/rsplit/splitlines return a fresh caller-owned mutable list;
        # in-place mutations on the result must be tracked (no graph break).
        def fn(t, s):
            # [FUNCTION DOC] HELPER: Splits the string three ways, mutates each list in place and
            # returns the tensor plus 1 and the three lists.
            words = s.split()
            words.sort(key=str.lower)
            words.append("zzz")
            rparts = s.rsplit(" ")
            rparts.reverse()
            lines = s.splitlines()
            lines.pop()
            return t + 1, words, rparts, lines

        s = "the Quick brown Fox"
        opt = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(torch.ones(3), s)
        res = opt(torch.ones(3), s)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1:], res[1:])

    def test_lru_cache_warning_issued_during_tracing(self):
        # [FUNCTION DOC] Compiles an lru_cache-wrapped function with the eager backend while
        # recording warnings, then asserts one of them mentions Dynamo detecting an
        # lru_cache-wrapped call. Fails with an assertion if no such warning was issued.
        import warnings
        from functools import lru_cache

        @lru_cache
        def foo(x):
            # [FUNCTION DOC] HELPER: lru_cache function returning x plus 1.
            return x + 1

        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            torch.compile(foo, backend="eager")(torch.randn(4))

        for warning in w:
            warning_message = str(warning.message)
            if (
                "Dynamo detected a call to a `functools.lru_cache`-wrapped"
                in warning_message
            ):
                break
        else:
            self.assertTrue(False, "Expected warning about lru_cache not found")

    @make_test
    def test_add(a, b):
        # [FUNCTION DOC] Returns the sum of two inputs.
        return a + b

    @make_test
    def test_add_(a, b):
        # [FUNCTION DOC] Copies a with the tensor constructor and applies in-place add of b with
        # alpha 5.
        a_copy = torch.tensor(a)
        return a_copy.add_(b, alpha=5.0)

    @make_test
    def test_addcdiv(a, b, c):
        # [FUNCTION DOC] Calls addcdiv with value 5.0; the comment says Dynamo decomposes this to
        # avoid a graph break when value is given.
        # dynamo decomposes this to avoid a graph break when
        # the value kwarg is populated
        return torch.addcdiv(a, b, c, value=5.0)

    @make_test
    def test_addcdiv_(a, b, c):
        # [FUNCTION DOC] Copies a and applies in-place addcdiv with b, c and value 5.0.
        a_copy = torch.tensor(a)
        return a_copy.addcdiv_(b, c, value=5.0)

    @make_test
    def test_is_not_null(a, b):
        # [FUNCTION DOC] Returns a plus b only if both are not None, else implicitly None.
        if a is not None and b is not None:
            return a + b

    def test_foreach_lerp_(self):
        # [FUNCTION DOC] Compiles a function calling the in-place foreach lerp on two tensor
        # lists with a tensor weight (fullgraph, counting backend); asserts compiled and eager
        # results match.
        def fn(x, y, s):
            # [FUNCTION DOC] HELPER: Return the result of the in-place foreach lerp of x towards y
            # by s.
            return torch._foreach_lerp_(x, y, s)

        cnt = torch._dynamo.testing.CompileCounter()

        fn_opt = torch.compile(backend=cnt, fullgraph=True)(fn)
        expected = fn(
            [torch.ones(2, 2) * 4.26, torch.ones(2, 2) * 3.14],
            [torch.ones(2, 2), torch.ones(2, 2)],
            torch.tensor(0.5),
        )

        actual = fn_opt(
            [torch.ones(2, 2) * 4.26, torch.ones(2, 2) * 3.14],
            [torch.ones(2, 2), torch.ones(2, 2)],
            torch.tensor(0.5),
        )
        self.assertTrue(same(expected, actual))

    def test_broadcast_foreach_pow(self):
        # [FUNCTION DOC] Compiles foreach pow with a tensor base and a list of tensor exponents
        # (broadcast case) and asserts results match eager; the frame count check uses a truthiness
        # assertion, so it only verifies the count is non-zero.
        # KNOWN ISSUE: the final check passes the expected value as the message of a truth
        # assertion, so it does not compare the frame count to 1.
        from torch._dynamo.utils import same

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return foreach pow of x (tensor or list) with exponents y.
            return torch._foreach_pow(x, y)

        cnt = torch._dynamo.testing.CompileCounter()

        fn_opt = torch.compile(backend=cnt, fullgraph=True)(fn)
        inps = (torch.tensor(0.80), [torch.tensor(3.4), torch.tensor(7.8)])

        actual = fn_opt(*inps)
        expected = fn(*inps)
        self.assertTrue(same(actual, expected))
        self.assertTrue(cnt.frame_count, 1)

    def test_foreach_norm_inf_dynamic(self):
        # [FUNCTION DOC] Compiles with dynamic shapes a foreach norm with infinity order over two
        # tensors of different shapes and asserts the per-tensor results equal eager.
        def fn(tensors):
            # [FUNCTION DOC] HELPER: Return foreach norm of the tensor list with ord set
            # to infinity.
            return torch._foreach_norm(tensors, float("inf"))

        fn_opt = torch.compile(fn, fullgraph=True, dynamic=True, backend="eager")

        tensors = [torch.randn(10), torch.randn(20, 30)]

        expected = fn(tensors)
        actual = fn_opt(tensors)

        self.assertEqual(len(actual), len(expected))
        for a, e in zip(actual, expected):
            self.assertEqual(a, e)

    def test_addcmul_(self):
        # [FUNCTION DOC] Compiles a function calling in-place addcmul with a tensor value on deep
        # copies of inputs; asserts outputs match eager and exactly one frame was compiled.
        from copy import deepcopy

        from torch._dynamo.utils import same

        def fn(x, y, z, s):
            # [FUNCTION DOC] HELPER: Apply in-place addcmul of y and z scaled by s on x and
            # return it.
            return x.addcmul_(y, z, value=s)

        cnt = torch._dynamo.testing.CompileCounter()
        fn_opt = torch.compile(backend=cnt, fullgraph=True)(fn)
        inps = (
            torch.ones(2, 2),
            torch.ones(2, 2) + 1,
            torch.rand(2, 2),
            torch.tensor(0.3),
        )
        inps_2 = deepcopy(inps)
        actual = fn_opt(*inps)
        expected = fn(*inps_2)
        self.assertTrue(same(actual, expected))
        self.assertEqual(cnt.frame_count, 1)

    @make_test
    def test_functools_partial(a, b):
        # [FUNCTION DOC] Applies the module level partial of clip with min 0 and max 1 to a plus b.
        return clip01(a + b)

    def test_functools_partial_from_unregistered_module(self):
        # [FUNCTION DOC] Builds a fresh module object not registered in sys.modules, executes
        # source defining a helper, a function and a functools partial in its namespace, then
        # compiles the partial with fullgraph and asserts it matches eager on a random tensor.
        module_name = "test_dynamo_unregistered_module"
        self.assertNotIn(module_name, sys.modules)
        module = types.ModuleType(module_name)
        exec(
            """
import functools
import torch

def helper(x, scale):
    return torch.nn.functional.normalize(x, dim=-1) * scale

def fn(x, scale):
    return helper(x, scale)

partial_fn = functools.partial(fn, scale=2)
""",
            module.__dict__,
        )

        x = torch.randn(4, 4)
        compiled_fn = torch.compile(module.partial_fn, backend="eager", fullgraph=True)
        self.assertEqual(compiled_fn(x), module.partial_fn(x))

    @make_test
    def test_itertools_product(a, b):
        # [FUNCTION DOC] Accumulates a plus x times i over the product of the list of a and b with
        # [1, 2].
        v = a
        for x, i in itertools.product([a, b], [1, 2]):
            v = v + x * i
        return v

    def test_itertools_product_args(self):
        # [FUNCTION DOC] Checks that product with an unsupported keyword argument raises
        # Unsupported when compiled with fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Return a tensor built from the list of itertools.product of
            # the args.
            return torch.tensor(list(itertools.product(*args, **kwargs)))

        self.assertRaises(Unsupported, fn, [1, 2, 3], fake_arg=1)

    @make_test
    def test_itertools_product_various_iterators(a, b):
        # [FUNCTION DOC] Builds an itertools product over lists, zip, map and filter iterators and
        # discards it; returns a.
        itertools.product(
            [a, b],
            zip([1, 2], [3, 4]),
            map(lambda x: x, [1, 2]),
            filter(lambda x: True, [1, 2]),
        )
        return a

    def test_itertools_permutations_basic(self):
        # [FUNCTION DOC] Compiles a function building a tensor from permutations of length 2 of a
        # list and asserts it equals the eager result.
        def fn():
            # [FUNCTION DOC] HELPER: Return a tensor of all 2-length permutations of [1, 2, 3].
            return torch.tensor(list(itertools.permutations([1, 2, 3], 2)))

        actual = torch.compile(fn, backend="eager", fullgraph=True)()
        expected = fn()
        self.assertEqual(actual, expected)

    def test_itertools_permutations_args(self):
        # [FUNCTION DOC] Checks permutations with bad arguments (none, too many, unknown keyword,
        # negative length) raises Unsupported under fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Return a tensor from permutations called with the
            # given arguments.
            return torch.tensor(list(itertools.permutations(*args, **kwargs)))

        self.assertRaises(Unsupported, fn)
        self.assertRaises(Unsupported, fn, [1, 2, 3], 1, 2)
        self.assertRaises(Unsupported, fn, [1, 2, 3], fake_arg=1)
        self.assertRaises(Unsupported, fn, [1, 2, 3], -1)

    @make_test
    def test_itertools_permutations_various_iterators(a, b):
        # [FUNCTION DOC] Builds permutations over a list, zip, map and filter iterators, discards
        # them and returns a.
        itertools.permutations([a, b])
        itertools.permutations(zip([1, 2], [3, 4]))
        itertools.permutations(map(lambda x: x, [1, 2]))
        itertools.permutations(filter(lambda x: True, [1, 2]))
        return a

    @make_test
    def test_itertools_filterfalse_basic(a, b):
        # [FUNCTION DOC] Adds to a every element of [-0.5, 0, 0.5] for which the predicate x > 0 is
        # false (via filterfalse) and returns a.
        for x in itertools.filterfalse(lambda x: x > 0, [-0.5, 0, 0.5]):
            a += x
        return a

    @make_test
    def test_itertools_chain(a, b):
        # [FUNCTION DOC] Sums a with every element of chaining [a, b] and [1, 2].
        v = a
        for x in itertools.chain([a, b], [1, 2]):
            v = v + x
        return v

    @make_test
    def test_itertools_chain_from_iterable(a, b):
        # [FUNCTION DOC] Sums a with every element of chain.from_iterable over [[a, b], [1, 2]].
        v = a
        for x in itertools.chain.from_iterable([[a, b], [1, 2]]):
            v = v + x
        return v

    def test_itertools_chain_fullgraph(self):
        # [FUNCTION DOC] Compiles with fullgraph a loop over a three-way chain and asserts it
        # equals eager.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Add each element of a chain over three lists to a and
            # return it.
            result = a
            for x in itertools.chain([a, b], [1, 2], [3]):
                result = result + x
            return result

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        a, b = torch.tensor(1.0), torch.tensor(2.0)
        self.assertEqual(fn(a, b), opt_fn(a, b))

    def test_itertools_chain_from_iterable_fullgraph(self):
        # [FUNCTION DOC] Same as the chain fullgraph test but using chain.from_iterable.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Add each element of the flattened chain to a and return
            # the result.
            result = a
            for x in itertools.chain.from_iterable([[a, b], [1, 2]]):
                result = result + x
            return result

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        a, b = torch.tensor(1.0), torch.tensor(2.0)
        self.assertEqual(fn(a, b), opt_fn(a, b))

    def test_itertools_chain_reconstruct(self):
        # [FUNCTION DOC] Checks that a partially consumed chain iterator returned from a compiled
        # function is reconstructed: first element and remaining list match eager.
        def fn(a):
            # [FUNCTION DOC] HELPER: Build a chain, advance it once, and return the iterator
            # and element.
            it = itertools.chain([a + 1, a + 2], [a + 3])
            result = next(it)
            return it, result

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        a = torch.tensor(0.0)
        it_ref, r_ref = fn(a)
        it_opt, r_opt = opt_fn(a)
        self.assertEqual(r_ref, r_opt)
        self.assertEqual(list(it_ref), list(it_opt))

    def test_itertools_reconstruct(self):
        # [FUNCTION DOC] Checks reconstruction of repeat and count iterators returned from compiled
        # code: next values and the tensor result match eager.
        def fn(a):
            # [FUNCTION DOC] HELPER: Advance a repeat and a count iterator three times each, adding
            # to a in place; return both iterators and a.
            it1 = itertools.repeat(1)
            it2 = itertools.count(2)
            for _ in range(3):
                a += next(it1)
                a += next(it2)
            return it1, it2, a

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        i1, i2, a = fn(torch.ones(3, 3))
        it1, it2, b = opt_fn(torch.ones(3, 3))
        self.assertEqual(next(i1), next(it1))
        self.assertEqual(next(i2), next(it2))
        self.assertEqual(a, b)

    def test_itertools_islice_basic_ops(self):
        # [FUNCTION DOC] Runs, under fullgraph compile, islice cases taken from CPython tests
        # (start/stop/step combos, stop None, items consumed from the iterator) asserting equality
        # with range slicing inside the compiled function.
        # Test cases taken from the CPython test TestBasicOps.test_islice. That test has a lot of
        # cases that we can't realistically support, whence we copy the sensible cases here.
        def fn():
            # [FUNCTION DOC] HELPER: Body with many islice assertions using the test
            # case's assertEqual.
            for args in [  # islice(args) should agree with range(args)
                (10, 20, 3),
                (10, 3, 20),
                (10, 20),
                (10, 10),
                (10, 3),
                (20,),
            ]:
                self.assertEqual(
                    list(itertools.islice(range(100), *args)), list(range(*args))
                )

            for args, tgtargs in [  # Stop when seqn is exhausted
                ((10, 110, 3), ((10, 100, 3))),
                ((10, 110), ((10, 100))),
                ((110,), (100,)),
            ]:
                self.assertEqual(
                    list(itertools.islice(range(100), *args)), list(range(*tgtargs))
                )

            # Test stop=None
            self.assertEqual(list(itertools.islice(range(10), None)), list(range(10)))
            self.assertEqual(
                list(itertools.islice(range(10), None, None)), list(range(10))
            )
            self.assertEqual(
                list(itertools.islice(range(10), None, None, None)), list(range(10))
            )
            self.assertEqual(
                list(itertools.islice(range(10), 2, None)), list(range(2, 10))
            )
            self.assertEqual(
                list(itertools.islice(range(10), 1, None, 2)), list(range(1, 10, 2))
            )

            # Test number of items consumed     SF #1171417
            it = iter(range(10))
            self.assertEqual(list(itertools.islice(it, 3)), list(range(3)))
            self.assertEqual(list(it), list(range(3, 10)))

            it = iter(range(10))
            self.assertEqual(list(itertools.islice(it, 3, 3)), [])
            self.assertEqual(list(it), list(range(3, 10)))

            # Issue #10323:  Less islice in a predictable state
            c = itertools.count()
            self.assertEqual(list(itertools.islice(c, 1, 3, 50)), [1])
            self.assertEqual(next(c), 3)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        opt_fn()

    @unittest.expectedFailure
    def test_itertools_islice_intlike(self):
        # [FUNCTION DOC] Expected failure: islice should accept integer-like objects (index method)
        # for its arguments, but Dynamo does not support this yet.
        # CPython issue #30537: islice can accept integer-like objects as arguments.
        class IntLike:
            # [CLASS DOC] HELPER: Minimal object exposing an integer through its index method.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Store val on the instance.
                self.val = val

            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the stored integer.
                return self.val

        def fn():
            # [FUNCTION DOC] HELPER: Asserts islice with IntLike start, stop and step
            # matches range.
            self.assertEqual(
                list(itertools.islice(range(100), IntLike(10))), list(range(10))
            )
            self.assertEqual(
                list(itertools.islice(range(100), IntLike(10), IntLike(50))),
                list(range(10, 50)),
            )
            self.assertEqual(
                list(
                    itertools.islice(range(100), IntLike(10), IntLike(50), IntLike(5))
                ),
                list(range(10, 50, 5)),
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        opt_fn()

    def test_bin_oct_hex_index(self):
        # [FUNCTION DOC] Checks bin, oct and hex on an object with an index method inside a
        # fullgraph compile; asserts the strings 0b11111111, 0o377, 0xff and equality with eager.
        # bin/oct/hex dispatch through __index__ (CPython PyNumber_ToBase).
        class Indexable:
            # [CLASS DOC] HELPER: Object whose index method returns the stored integer.
            def __init__(self, val):
                # [FUNCTION DOC] HELPER: Store val.
                self.val = val

            def __index__(self):
                # [FUNCTION DOC] HELPER: Return the stored integer.
                return self.val

        def fn(t):
            # [FUNCTION DOC] HELPER: Return the tensor plus 1 and bin, oct, hex of an Indexable
            # object holding 255.
            obj = Indexable(255)
            return t + 1, bin(obj), oct(obj), hex(obj)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(torch.zeros(1))
        res = opt_fn(torch.zeros(1))
        self.assertEqual(res[1:], ("0b11111111", "0o377", "0xff"))
        self.assertEqual(ref[1:], res[1:])

    def test_hex_index_type_error(self):
        # [FUNCTION DOC] Checks hex of an object lacking an index method raises TypeError inside
        # compiled code, handled by the function returning t minus 1; compares to eager.
        # hex on an object without __index__ raises TypeError, like CPython.
        class NoIndex:
            # [CLASS DOC] HELPER: Empty class without an index method.
            pass

        def fn(t):
            # [FUNCTION DOC] HELPER: Try hex of a NoIndex instance; return t plus 1 on success, t
            # minus 1 on TypeError.
            try:
                hex(NoIndex())
                return t + 1
            except TypeError:
                return t - 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(torch.zeros(1)), fn(torch.zeros(1)))

    def test_bin_oct_hex_require_index_not_int(self):
        # [FUNCTION DOC] Checks bin, oct and hex reject an object defining only an int conversion,
        # by raising TypeError; the compiled result is the tuple (0, 0, 0) and matches eager.
        # oct/hex/bin dispatch through __index__, not __int__: an object with
        # only __int__ must raise TypeError (CPython PyNumber_ToBase).
        class IntNoIndex:
            # [CLASS DOC] HELPER: Object defining only the int conversion method (returns 5),
            # no index.
            def __int__(self):
                # [FUNCTION DOC] HELPER: Return 5.
                return 5

        def fn(t):
            # [FUNCTION DOC] HELPER: For each of bin, oct, hex record 1 if no error else 0 on
            # TypeError; return t plus 1 and the tuple of flags.
            results = []
            for f in (bin, oct, hex):
                try:
                    f(IntNoIndex())
                    results.append(1)
                except TypeError:
                    results.append(0)
            return t + 1, tuple(results)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        ref = fn(torch.zeros(1))
        res = opt_fn(torch.zeros(1))
        self.assertEqual(res[1], (0, 0, 0))
        self.assertEqual(ref[1], res[1])

    def test_bin_oct_hex_index_int_subclass(self):
        # [FUNCTION DOC] Regression test (PR 191408) for int subclasses, IntEnum and IntFlag
        # members: bin, oct, hex and operator.index must constant fold to the underlying int
        # rather than recursing. Resets Dynamo per object and compares compiled to eager and to
        # plain int results.
        # For an int subclass (incl. IntEnum/IntFlag members) __index__ is the
        # inherited int slot. hex/oct/bin/operator.index must const-fold to the
        # underlying int rather than re-dispatching into the slot, which would
        # recurse forever (regression for PR #191408).
        class MyInt(int):
            # [CLASS DOC] HELPER: int subclass with a custom new method.
            def __new__(cls, v):
                # [FUNCTION DOC] HELPER: Create the int instance by calling the parent new with v.
                return super().__new__(cls, v)

        class Color(enum.IntEnum):
            # [CLASS DOC] HELPER: IntEnum with members RED equal to 1 and GREEN equal to 2.
            RED = 1
            GREEN = 2

        class Perm(enum.IntFlag):
            # [CLASS DOC] HELPER: IntFlag with members R equal to 4 and W equal to 2.
            R = 4
            W = 2

        for obj in (MyInt(10), Color.RED, Color.GREEN, Perm.R | Perm.W):

            def fn(t, obj=obj):
                # [FUNCTION DOC] HELPER: Return t plus 1 plus bin, oct, hex and operator.index of
                # the captured obj.
                return t + 1, bin(obj), oct(obj), hex(obj), operator.index(obj)

            torch._dynamo.reset()
            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            ref = fn(torch.zeros(1))
            res = opt_fn(torch.zeros(1))
            self.assertEqual(res[1:], ref[1:])
            self.assertEqual(
                res[1:],
                (bin(int(obj)), oct(int(obj)), hex(int(obj)), int(obj)),
            )

    @make_test
    def test_obj_eq(a, b):
        # [FUNCTION DOC] Checks equality and inequality of fresh MyCls instances and with None; the
        # expected path returns v plus 1 where v is the sine of a plus b.
        v = a + b
        if MyCls() == None:  # noqa: E711
            return -1
        if MyCls() != None:  # noqa: E711
            v = v.sin()
        if MyCls() == MyCls():
            return -2
        if MyCls() != MyCls():
            return v + 1
        return -3

    @make_test
    def test_cls_eq(a, b):
        # [FUNCTION DOC] Same as the object equality test but comparing the class MyCls itself.
        v = a + b
        if MyCls == None:  # noqa: E711
            return -1
        if MyCls != None:  # noqa: E711
            v = v.sin()
        if MyCls != MyCls:
            return -2
        if MyCls == MyCls:
            return v + 1
        return -3

    @make_test
    def test_obj_is(a, b):
        # [FUNCTION DOC] Checks identity operators on fresh MyCls instances and None.
        v = a + b
        if MyCls() is None:
            return -1
        if MyCls() is not None:
            v = v.sin()
        if MyCls() is MyCls():
            return -2
        if MyCls() is not MyCls():
            return v + 1
        return -3

    @make_test
    def test_cls_is(a, b):
        # [FUNCTION DOC] Checks identity operators comparing the class MyCls to None and itself.
        v = a + b
        if MyCls is None:
            return -1
        if MyCls is not None:
            v = v.sin()
        if MyCls is not MyCls:
            return -2
        if MyCls is MyCls:
            return v + 1
        return -3

    @make_test
    def test_itertools_combinations(a, b):
        # [FUNCTION DOC] Builds a list of ones tensors sized from combinations of two out
        # of (1,2,3,4).
        combs = []
        for size in itertools.combinations((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    def test_itertools_combinations_args(self):
        # [FUNCTION DOC] Checks itertools combinations argument validation under fullgraph:
        # calls with no args, too many args, an unknown keyword and a negative size, all
        # expecting Unsupported.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Return a list of combinations called with the given arguments.
            return list(itertools.combinations(*args, **kwargs))

        self.assertRaises(Unsupported, fn)
        self.assertRaises(Unsupported, fn, [1, 2, 3], 1, 2)
        self.assertRaises(Unsupported, fn, [1, 2, 3], fake_arg=1)
        self.assertRaises(Unsupported, fn, [1, 2, 3], -1)

    @make_test
    def test_itertools_combinations_various_iterators(a, b):
        # [FUNCTION DOC] Builds combinations of size 1 over a list, zip, map and filter iterators,
        # discards them and returns a.
        itertools.combinations([a, b], 1)
        itertools.combinations(zip([1, 2], [3, 4]), 1)
        itertools.combinations(map(lambda x: x, [1, 2]), 1)
        itertools.combinations(filter(lambda x: True, [1, 2]), 1)
        return a

    @make_test
    def test_itertools_combinations_with_replacement(a, b):
        # [FUNCTION DOC] Accumulates ones tensors sized by each pair from
        # combinations_with_replacement of (1,2,3,4) taken 2 at a time; returns the list.
        combs = []
        for size in itertools.combinations_with_replacement((1, 2, 3, 4), 2):
            combs.append(torch.ones(size))
        return combs

    if sys.version_info >= (3, 12):

        @make_test
        def test_itertools_batched(a):
            # [FUNCTION DOC] Only defined on Python 3.12 or newer. Builds ones tensors sized by
            # each batch of two from itertools.batched over five ints; returns the list of tensors.
            batches = []
            for size in itertools.batched((1, 2, 3, 4, 5), 2):
                batches.append(torch.ones(size))
            return batches

    @make_test
    def test_itertools_pairwise(a):
        # [FUNCTION DOC] Builds ones tensors sized by consecutive pairs from pairwise
        # over (1,2,3,4).
        pairs = []
        for size in itertools.pairwise((1, 2, 3, 4)):
            pairs.append(torch.ones(size))
        return pairs

    def test_itertools_compress(self):
        # [FUNCTION DOC] Compiles with fullgraph a function returning itertools.compress over a
        # string and selectors; asserts the resulting lists equal eager.
        def fn():
            # [FUNCTION DOC] HELPER: Return compress of "ABCDEF" with selectors 1,0,1,0,1,1.
            return itertools.compress("ABCDEF", [1, 0, 1, 0, 1, 1])

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertListEqual(list(opt_fn()), list(fn()))

    def test_itertools_compress_tensors(self):
        # [FUNCTION DOC] Same as the compress test but selecting from a list of three tensors.
        def fn():
            # [FUNCTION DOC] HELPER: Return compress of three single-element tensors with
            # selectors 1,0,1.
            return itertools.compress(
                [torch.tensor([0]), torch.tensor([1]), torch.tensor([2])], [1, 0, 1]
            )

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertListEqual(list(opt_fn()), list(fn()))

    @make_test
    def test_np_iinfo(a):
        # [FUNCTION DOC] Adds numpy int16 maximum (from numpy iinfo) to a.
        max_dim = np.iinfo(np.int16).max
        return a + max_dim

    @make_test
    def test_np_finfo(a):
        # [FUNCTION DOC] Adds numpy float32 minimum (from numpy finfo) to a.
        min_dim = np.finfo(np.float32).min
        return a + min_dim

    @make_test
    def test_constant1(a, b, c):
        # [FUNCTION DOC] Returns a minus b times c plus 1.0.
        return a - b * c + 1.0

    @make_test
    def test_constant2(a, b, c):
        # [FUNCTION DOC] Returns a minus b times c plus 1 (integer constant).
        return a - b * c + 1

    @make_test
    def test_constant3(a):
        # [FUNCTION DOC] Local constant arithmetic 1 plus 2 minus 3 added to a.
        b = 1
        c = 2
        d = 3
        return b + c - d + a

    @make_test
    def test_constant4(a, b):
        # [FUNCTION DOC] Branch on a constant comparison (2 > 3 is false): returns b minus a.
        c = 2
        d = 3
        if c > d:
            return a - b
        return b - a

    @make_test
    def test_cls_hasattr(self, x):
        # [FUNCTION DOC] Uses hasattr on class MyCls: adds 1 since attribute a exists, adds
        # nothing for the missing attribute b. The first parameter is named self but is just the
        # tensor input.
        if hasattr(MyCls, "a"):
            x = x + 1
        if hasattr(MyCls, "b"):
            x = x + 2
        return x

    @make_test
    def test_finfo(a, b):
        # [FUNCTION DOC] If the int32 bit width is 32 (true) returns the float finfo minimum of a's
        # dtype times b; otherwise None.
        if torch.iinfo(torch.int32).bits == 32:
            return torch.finfo(a.dtype).min * b

    @make_test
    def test_globalfn(a, b):
        # [FUNCTION DOC] Calls the sub function imported at module level.
        return sub(a, b)

    @make_test
    def test_viatorch(a, b):
        # [FUNCTION DOC] Calls torch.sub on the inputs.
        return torch.sub(a, b)

    @make_test
    def test_viamethod(a, b):
        # [FUNCTION DOC] Calls the tensor method sub.
        return a.sub(b)

    @make_test
    def test_indirect1(a, b):
        # [FUNCTION DOC] Binds the sub method of a to a local and calls it.
        t = a.sub
        return t(b)

    @make_test
    def test_indirect2(a, b):
        # [FUNCTION DOC] Calls a bound method through star-args unpacking of a tuple.
        t = a.sub
        args = (b,)
        return t(*args)

    @make_test
    def test_indirect3(a, b):
        # [FUNCTION DOC] Calls a bound method with star-args and an empty star-kwargs mapping.
        t = a.sub
        args = (b,)
        kwargs = {}
        return t(*args, **kwargs)

    @make_test
    def test_methodcall1(a, b, c):
        # [FUNCTION DOC] Calls module function constant3 positionally and multiplies by c.
        return constant3(a, b) * c

    @make_test
    def test_methodcall2(a, b):
        # [FUNCTION DOC] Calls constant3 with swapped keyword arguments and adds 1.
        return constant3(a=b, b=a) + 1

    @make_test
    def test_methodcall3(a, b):
        # [FUNCTION DOC] Calls constant3 with a keyword for b and adds b.
        return constant3(a, b=1.0) + b

    def test_is_integer(self):
        # [FUNCTION DOC] Compiles a function using float is_integer in a conditional; asserts the
        # result is 2 for 1.0 and 1 for 1.5.
        @torch.compile(backend="eager", fullgraph=True)
        def forward(t, m):
            # [FUNCTION DOC] HELPER: Return twice t if m reports it is an integer, else t.
            return 2 * t if m.is_integer() else t

        t = torch.tensor([1])
        self.assertEqual(forward(t, 1.0).item(), 2)
        self.assertEqual(forward(t, 1.5).item(), 1)

    @parametrize(
        "method, num_type",
        (
            ("as_integer_ratio", int),
            ("bit_length", int),
            ("conjugate", int),
            ("as_integer_ratio", float),
            ("conjugate", float),
            ("hex", float),
            ("is_integer", float),
        ),
    )
    def test_number_method(self, method, num_type):
        # [FUNCTION DOC] Parametrized over number methods and types (int and float). Compiles a
        # function that calls the named zero-argument method on a number and doubles t if the
        # result is truthy; checks results match eager for 0, 1 and 2.5 converted to the type.
        def forward(t, m):
            # [FUNCTION DOC] HELPER: Call the named method on m via attribute lookup; return twice
            # t if truthy.
            return 2 * t if getattr(m, method)() else t

        wrapped = torch.compile(backend="eager", fullgraph=True)(forward)

        for i in (0, 1, 2.5):
            m = num_type(i)
            t = torch.tensor([1])
            actual = wrapped(t, m)
            expected = forward(t, m)
            self.assertEqual(actual, expected)

    @make_test
    def test_device_constant(a):
        # [FUNCTION DOC] Adds a ones tensor of shape one to a.
        return a + torch.ones(1)

    @make_test
    def test_tuple1(a, b):
        # [FUNCTION DOC] Calls sub with star-args from a tuple.
        args = (a, b)
        return sub(*args)

    @make_test
    def test_tuple2(a, b):
        # [FUNCTION DOC] Calls sub with star-args from a list.
        args = [a, b]
        return sub(*args)

    @make_test
    def test_tuple_map(a, b):
        # [FUNCTION DOC] Maps torch.sin over [a, b] into a tuple and sums its two elements.
        t = tuple(map(torch.sin, [a, b]))
        return t[0] + t[1]

    def test_size_tuple_add(self):
        # [FUNCTION DOC] Checks that adding two torch.Size objects yields a Size, and that Size
        # plus an empty tuple is both a tuple and a Size instance. Runs the function eagerly, then
        # compiled with fullgraph and the eager backend.
        def fn():
            # [FUNCTION DOC] HELPER: Build assertions about Size addition results (plain asserts).
            size = torch.Size([])
            assert isinstance(size + size, torch.Size)  # noqa: S101
            assert isinstance(size + (), tuple)  # noqa: S101
            assert isinstance(size + (), torch.Size)  # noqa: S101

        fn()
        compiled_fn = torch.compile(fn, backend="eager", fullgraph=True)
        compiled_fn()

    @make_test
    def test_is_in_onnx_export(x, y):
        # [FUNCTION DOC] Branches on whether ONNX export is in progress: x minus 1 if so, else y
        # plus 1.
        if torch.onnx.is_in_onnx_export():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_is_fx_tracing(x, y):
        # [FUNCTION DOC] Branches on whether torch.fx symbolic tracing is active: x minus 1 if so,
        # else y plus 1.
        if torch.fx._symbolic_trace.is_fx_tracing():
            return x - 1
        else:
            return y + 1

    @make_test
    def test_listarg1(a, b):
        # [FUNCTION DOC] Concatenates a and b passed as a list.
        return torch.cat([a, b])

    @make_test
    def test_listarg2(a, b):
        # [FUNCTION DOC] Concatenates a and b passed as a tuple with dim 0 keyword.
        return torch.cat((a, b), dim=0)

    @make_test
    def test_listarg3(a, b):
        # [FUNCTION DOC] Calls cat with a keyword mapping holding the tensors tuple and dim.
        kwargs = {"tensors": (a, b), "dim": 0}
        return torch.cat(**kwargs)

    @make_test
    def test_listarg4(a, b):
        # [FUNCTION DOC] Calls cat with keywords tensors and dim.
        return torch.cat(tensors=[a, b], dim=0)

    @make_test
    def test_listarg5(a, b):
        # [FUNCTION DOC] Calls cat with star-args of a one-item list and a star-kwargs dim.
        args = [(a, b)]
        kwargs = {"dim": 0}
        return torch.cat(*args, **kwargs)

    def test_list_extend_set_and_dict_iterables(self):
        # [FUNCTION DOC] Compiles with fullgraph a function extending a list with a set, dict, dict
        # keys, values and items; asserts the first item and last entries equal eager and the set
        # part contains 1, 2, 3 (set order is not compared).
        def fn(x):
            # [FUNCTION DOC] HELPER: Build a list by extending with various iterables and
            # return it.
            out = [x]
            out.extend({1, 2, 3})

            mapping = {"a": x + 1, "b": x + 2}
            out.extend(mapping)
            out.extend(mapping.keys())
            out.extend(mapping.values())
            out.extend(mapping.items())
            return out

        x = torch.ones(2, 2)
        expected = fn(x)
        actual = torch.compile(fn, backend="eager", fullgraph=True)(x)

        self.assertEqual(actual[0], expected[0])
        self.assertEqual(set(actual[1:4]), {1, 2, 3})
        self.assertEqual(actual[4:], expected[4:])

    def test_list_slice(self):
        # [FUNCTION DOC] Compiles a method that slices an instance list attribute to the last three
        # items, appends the input and returns sin; asserts the list holds one item after one call.
        class Mock:
            # [CLASS DOC] HELPER: Holds an empty list ets and a counter; its compiled run method
            # updates ets.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize ets to an empty list and counter to 0.
                self.ets = []
                self.counter = 0

            @torch.compile(backend="eager")
            def run(self, x):
                # [FUNCTION DOC] HELPER: Compiled method that truncates the ets list, appends x and
                # returns sin.
                self.ets = self.ets[-3:]
                self.ets.append(x)
                return torch.sin(x)

        mock = Mock()
        mock.run(torch.randn(4))
        self.assertEqual(len(mock.ets), 1)

    @make_test
    def test_deque(a, b):
        # [FUNCTION DOC] Exercises a deque: append, extend, insert, pop, extendleft from another
        # deque, clear, item assignment, copy, rotate one element with popleft, and extending by an
        # empty deque. Returns the final deque.
        d = collections.deque([a, b])
        d.append(a + 1)
        d.extend([a, b])
        d.insert(0, "foo")
        tmp = d.pop()

        another_deque = collections.deque([tmp])
        d.extendleft(another_deque)
        another_deque.clear()
        d.extend(another_deque)

        d[2] = "setitem"
        d = d.copy()
        d.append(d.popleft())

        empty = collections.deque()
        d.extend(empty)

        return d

    @make_test
    def test_deque_reinit_resets_maxlen(a, b):
        # [FUNCTION DOC] Checks that re-initializing a deque with a larger iterable resets maxlen
        # and is not clamped to the old max length; returns the deque and its maxlen.
        # deque.__init__ resets maxlen; re-init with more items than the old
        # maxlen must not be clamped to the old maxlen.
        d = collections.deque([a, b], maxlen=2)
        d.__init__([a, b, a + 1, b + 1])
        return d, d.maxlen

    @make_test
    def test_slice1(a):
        # [FUNCTION DOC] Indexes a at position 5.
        return a[5]

    @make_test
    def test_slice2(a):
        # [FUNCTION DOC] Slices a up to index 5.
        return a[:5]

    @make_test
    def test_slice3(a):
        # [FUNCTION DOC] Slices a from index 5.
        return a[5:]

    @make_test
    def test_slice4(a):
        # [FUNCTION DOC] Slices a from index 2 to 5.
        return a[2:5]

    @make_test
    def test_slice5(a):
        # [FUNCTION DOC] Slices a with step 2.
        return a[::2]

    @make_test
    def test_slice6(a):
        # [FUNCTION DOC] Unsqueezes a at dim 0 and slices the last dimension from index 2.
        return torch.unsqueeze(a, 0)[:, 2:]

    @make_test
    def test_range1(a):
        # [FUNCTION DOC] Builds a tensor from a range of a's first-dimension size.
        return torch.tensor(range(a.size(0)))

    @make_test
    def test_range2(x, y):
        # [FUNCTION DOC] Divides x plus y by y repeatedly in a loop of range of the first dimension
        # size of x plus 2.
        r = x + y
        for _ in range(x.size(0) + 2):
            r = r / y
        return r

    @make_test
    def test_unpack1(a):
        # [FUNCTION DOC] Splits a into two slices by unpacking and returns their difference.
        a, b = a[:5], a[5:]
        return a - b

    @make_test
    def test_unpack2(a):
        # [FUNCTION DOC] Unpacks a two-element list of slices and returns their difference.
        packed = [a[:5], a[5:]]
        a, b = packed
        return a - b

    @make_test
    def test_unpack3(a):
        # [FUNCTION DOC] Unpacks a two-element tuple of slices and returns their difference.
        packed = (a[:5], a[5:])
        a, b = packed
        return a - b

    @make_test
    def test_fn_with_self_set(a, b):
        # [FUNCTION DOC] Calls avg_pool2d (a builtin with __self__ set) on the product of
        # unsqueezed inputs with kernel size 2 and padding 1.
        # avg_pool2d is an odd one with __self__ set
        return F.avg_pool2d(
            torch.unsqueeze(a, 0) * torch.unsqueeze(b, 1), kernel_size=2, padding=1
        )

    @make_test
    def test_return_tuple1(a, b):
        # [FUNCTION DOC] Returns a tuple of a-b, b-a, a and b.
        return (a - b, b - a, a, b)

    @make_test
    def test_globalvar(a, b):
        # [FUNCTION DOC] Reads global tensor d and returns a minus b plus d.
        return a - b + d

    @make_test
    def test_globalmodule(x):
        # [FUNCTION DOC] Calls global Linear module e on x.
        return e(x)

    @make_test
    def test_inline_with_default(a, b, c):
        # [FUNCTION DOC] Calls func_with_default with its default argument and multiplies by c.
        return func_with_default(a, b) * c

    @make_test
    def test_inner_function(x):
        # [FUNCTION DOC] Defines and calls an inner function that returns x plus x.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return torch.add of x with itself.
            return torch.add(x, x)

        return fn(x)

    @make_test
    def test_transpose_for_scores(x):
        # [FUNCTION DOC] Reshapes x into its leading sizes plus a (2, 5) tail and permutes dims 0,
        # 2, 1.
        new_x_shape = x.size()[:-1] + (2, 5)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1)

    @make_test
    def test_return_tuple2(x):
        # [FUNCTION DOC] Returns a tuple of x plus x and x.
        return (torch.add(x, x), x)

    @make_test
    def test_load_global_bool(x):
        # [FUNCTION DOC] Branches on the module global flag (True): returns x plus x else x.
        if flag:
            return torch.add(x, x)
        else:
            return x

    @make_test
    def test_len_tensor(x):
        # [FUNCTION DOC] Adds the length of the tensor (first dim) to x.
        z = len(x)
        return torch.add(x, z)

    @make_test
    def test_len_constant_list(x):
        # [FUNCTION DOC] Adds the length of a constant list (3) to x.
        z = len([1, 2, 3])
        return torch.add(x, z)

    @make_test
    def test_len_constant_dict(x):
        # [FUNCTION DOC] Adds the length of a constant dict (1) to x.
        z = len({"foo": "bar"})
        return torch.add(x, z)

    @make_test
    def test_dict_copy(x):
        # [FUNCTION DOC] Copies a dict via the dict constructor and returns it.
        z = dict({"foo": x + 1})
        return z

    @make_test
    def test_dict_keys(x):
        # [FUNCTION DOC] Takes a live keys view of a dict, mutates the dict afterwards and
        # checks key membership and view equality against another dict; returns three booleans
        # plus comparison.
        d = {3: x}
        keys = d.keys()
        d[4] = x + 1
        d2 = {3: 2, 4: "aa"}
        return 3 in keys, 4 in keys, 5 in keys, d2.keys() == keys

    @make_test
    def test_dict_values(x):
        # [FUNCTION DOC] Takes a values view, mutates the dict afterwards and returns the
        # view length.
        d = {3: x}
        values = d.values()
        d[3] = x + 1
        d[4] = x + 2
        return len(values)

    @make_test
    def test_dict_setdefault1(x):
        # [FUNCTION DOC] setdefault on an existing key leaves it unchanged, so returns x plus 1.
        d = {"a": 1, "b": 2}
        d.setdefault("a", 10)
        if d["a"] == 1:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_dict_setdefault2(x):
        # [FUNCTION DOC] setdefault on a new key stores the given default (10), so returns x
        # plus 1.
        d = {"a": 1, "b": 2}
        d.setdefault("c", 10)
        if d["c"] == 10:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_dict_setdefault3(x):
        # [FUNCTION DOC] setdefault with no default on a new key stores None, so returns x plus 1.
        d = {"a": 1, "b": 2}
        d.setdefault("c")
        if d["c"] is None:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_dict_update_kwargs(x):
        # [FUNCTION DOC] Updates a dict with a keyword argument and multiplies x by two values.
        d = {"a": 2}
        d.update(b=4)
        return x * d["a"] * d["b"]

    @make_test
    def test_defaultdict_setdefault1(x):
        # [FUNCTION DOC] Same as the dict setdefault existing-key test but on a defaultdict.
        d = collections.defaultdict.fromkeys("a", "b")
        d["a"] = 1
        d["b"] = 2
        d.setdefault("a", 10)
        if d["a"] == 1:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_defaultdict_setdefault2(x):
        # [FUNCTION DOC] Same as the dict setdefault new-key test but on a defaultdict.
        d = collections.defaultdict.fromkeys("a", "b")
        d["a"] = 1
        d["b"] = 2
        d.setdefault("c", 10)
        if d["c"] == 10:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_defaultdict_setdefault3(x):
        # [FUNCTION DOC] Same as the dict setdefault None-default test but on a defaultdict.
        d = collections.defaultdict.fromkeys("a", "b")
        d["a"] = 1
        d["b"] = 2
        d.setdefault("c")
        if d["c"] is None:
            return x + 1
        else:
            return x - 1

    def test_dict_id_guard(self):
        # [FUNCTION DOC] Compiles a function that iterates one OrderedDict and indexes it through
        # two aliases of the same dict, forcing a dict guard keyed by id; compares to eager.
        d1 = collections.OrderedDict({"a": 2})
        d2 = d1

        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply x by each dict value twice (via both aliases).
            # Iteration forces DictGuardManager
            for k in d1:
                x = x * d1[k] * d2[k]
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    @make_test
    def test_callable_lambda(x):
        # [FUNCTION DOC] Checks callable on a lambda is true (x plus 1 branch).
        if callable(lambda x: True):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_torch(x):
        # [FUNCTION DOC] Checks callable on torch.abs is true.
        if callable(torch.abs):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_callable_builtin(x):
        # [FUNCTION DOC] Checks callable on the builtin sum is true.
        if callable(sum):
            return x + 1
        else:
            return x - 1

    def test_callable_class(self):
        # [FUNCTION DOC] Compiles two functions branching on callable of an argument and asserts
        # correct results for an instance of a class defining a call method, one that does not, and
        # for ints, floats, bools and tensors (not callable).
        class CallableClass:
            # [CLASS DOC] HELPER: Class that defines a call method, so its instances are callable.
            def __call__():
                # [FUNCTION DOC] HELPER: Call method with no parameters (never invoked);
                # does nothing.
                pass

        class NotCallableClass:
            # [CLASS DOC] HELPER: Empty class whose instances are not callable.
            pass

        @torch.compile(backend="eager", fullgraph=True)
        def fn1(x, arg):
            # [FUNCTION DOC] HELPER: Return x if arg is callable, else x plus 1.
            if callable(arg):
                return x
            return x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn2(x, arg):
            # [FUNCTION DOC] HELPER: Return twice x if arg is callable, else x plus 1.
            if callable(arg):
                return x * 2
            return x + 1

        input = torch.randn(4)

        for f in [fn1, fn2]:
            self.assertEqual(f(input, NotCallableClass()), input + 1)
            self.assertEqual(
                f(input, CallableClass()), input if f is fn1 else input * 2
            )

            # passing tensor and scalars
            self.assertEqual(f(input, 1), input + 1)
            self.assertEqual(f(input, 1.1), input + 1)
            self.assertEqual(f(input, True), input + 1)
            self.assertEqual(f(input, input), input + 1)

    def test_callable_list(self):
        # [FUNCTION DOC] Checks callable returns false for a list and a tuple under fullgraph.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(x, arg):
            # [FUNCTION DOC] HELPER: Return x if arg is callable, else x plus 1.
            if callable(arg):
                return x
            return x + 1

        input = torch.randn(4)
        self.assertEqual(fn(input, [1, 2, 3]), input + 1)
        self.assertEqual(fn(input, (1, 2, 3)), input + 1)

    def test_pos_only_args_with_same_name_in_star_kwargs(self):
        # [FUNCTION DOC] Compares compiled and eager pos_only_fn for calls where keyword arguments
        # have the same names as the positional-only parameters, in several combinations.
        opt_fn = torch.compile(pos_only_fn, backend="eager", fullgraph=True)
        a = torch.randn(4)
        b = torch.randn(4)
        x = torch.randn(4)
        y = torch.randn(4)
        self.assertEqual(pos_only_fn(a), opt_fn(a))
        self.assertEqual(pos_only_fn(a, a=x), opt_fn(a, a=x))
        self.assertEqual(pos_only_fn(a, b=y), opt_fn(a, b=y))
        self.assertEqual(pos_only_fn(a, b=b, a=x), opt_fn(a, b=b, a=x))
        self.assertEqual(pos_only_fn(a, a=x, b=y), opt_fn(a, a=x, b=y))
        self.assertEqual(pos_only_fn(a, b, a=x, b=y), opt_fn(a, b, a=x, b=y))

    @make_test
    def test_len_constant_misc_iterables(x):
        # [FUNCTION DOC] Adds len of a tuple (3) plus len of a string (8) to x.
        a = len((1, 2, 3))
        b = len("test str")
        c = a + b
        return torch.add(x, c)

    @make_test
    def test_dict_kwargs(x):
        # [FUNCTION DOC] Builds a dict through keyword arguments of the dict constructor and
        # returns it.
        z = dict(text_embed=x + 1, other=x + 2)
        return z

    @make_test
    def test_ordered_dict_kwargs(x):
        # [FUNCTION DOC] Builds an OrderedDict through keyword arguments and returns it.
        z = collections.OrderedDict(sample=torch.ones(10))
        return z

    @make_test
    def test_custom_dict_kwargs(x):
        # [FUNCTION DOC] Builds CustomDictSubclass through keyword arguments and returns it.
        z = CustomDictSubclass(sample=torch.ones(10))
        return z

    @make_test
    def test_float(x):
        # [FUNCTION DOC] Builds floats from a float and a string and adds the sum to x.
        y = float(1.2)  # noqa: UP018
        y += float("1.2")
        return torch.add(x, y)

    def test_float_or_complex_from_number(self):
        # [FUNCTION DOC] Builds two make_test tests using float.from_number and
        # complex.from_number. Before Python 3.14 calling them must raise AttributeError; from 3.14
        # they must run.
        @make_test
        def _float_from_number_impl(x):
            # [FUNCTION DOC] HELPER: Adds the result of float from_number applied to 10 to x.
            y = float.from_number(10)
            return torch.add(x, y)

        @make_test
        def _complex_from_number_impl(x):
            # [FUNCTION DOC] HELPER: Adds the result of complex from_number applied to 10 to x.
            y = complex.from_number(10)
            return torch.add(x, y)

        if sys.version_info < (3, 14):
            with self.assertRaises(AttributeError):
                _float_from_number_impl(self)
            with self.assertRaises(AttributeError):
                _complex_from_number_impl(self)
        else:
            _float_from_number_impl(self)
            _complex_from_number_impl(self)

    @make_test
    def test_is_floating_point(x):
        # [FUNCTION DOC] Returns the pair of is_floating_point results (positional and keyword) for
        # x plus 1.
        y = x + 1
        return torch.is_floating_point(y), torch.is_floating_point(input=y)

    @make_test
    def test_dtype(x):
        # [FUNCTION DOC] Returns x plus 1 only if x has float32 dtype; otherwise None.
        if x.dtype == torch.float32:
            return x + 1

    @make_test
    def test_get_default_dtype(x):
        # [FUNCTION DOC] Compares dtype with the default dtype to choose between x plus 1 and x
        # minus 1.
        if x.dtype == torch.get_default_dtype():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_get_autocast_gpu_dtype(x):
        # [FUNCTION DOC] Casts x to the autocast GPU dtype.
        dtype = torch.get_autocast_gpu_dtype()
        return x.type(dtype)

    @make_test
    def test_is_any_autocast_enabled(x):
        # [FUNCTION DOC] Branches on whether any autocast is enabled.
        if torch._C._is_any_autocast_enabled():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_is_checkpoint_valid(x):
        # [FUNCTION DOC] Branches on the autograd checkpoint validity flag.
        if torch.autograd._is_checkpoint_valid():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_list_compare_polyfill(x):
        # [FUNCTION DOC] Compares constant tuples with all six comparison operators in a loop,
        # adding a weighted constant for each true comparison (tests the list compare polyfill);
        # returns x.
        for a, b, c in [
            [(1, 2, 3), (1, 2, 3), 7.77],
            [(1, 4, 3), (1, 2, 3), 3.33],
            [(1, 2), (1, 2, 3), 5.55],
            [(1, 2, 3), (1, 2), 11.11],
            [(1, -1, 3), (1, 2, 3), 13.33],
        ]:
            if a != b:
                x = x + 1 * c
            if a == b:
                x = x + 2 * c
            if a < b:
                x = x + 4 * c
            if a > b:
                x = x + 8 * c
            if a <= b:
                x = x + 16 * c
            if a >= b:
                x = x + 32 * c
        return x

    @make_test
    def test_list_compare_polyfill_non_lists(x):
        # [FUNCTION DOC] Compares a tuple with a string and an int with a tuple using equality and
        # inequality only; returns x and the list of condition results.
        conds = []

        # Non-list instances only work for eq and ne
        for a, b, c in [
            [(1, 2, 3), "(1, 2, 3)", 7.77],
            [143, (143,), 3.33],
        ]:
            conds.append(a != b)
            if conds[-1]:
                x = x + 1 * c

            conds.append(a == b)
            if conds[-1]:
                x = x + 2 * c

        return x, conds

    @make_test
    def test_promote_types(x):
        # [FUNCTION DOC] Branches on whether x's dtype equals the promotion of int32 and float32.
        if x.dtype == torch.promote_types(torch.int32, torch.float32):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_cublas_allow_tf32(x):
        # [FUNCTION DOC] Branches on the cuda matmul allow_tf32 setting: sin plus 1, else cos
        # minus 1.
        if torch.backends.cuda.matmul.allow_tf32:
            return x.sin() + 1

        return x.cos() - 1

    @make_test
    def test_get_calculate_correct_fan(x):
        # [FUNCTION DOC] Computes the fan-in via the init helper and adds it to x.
        fan_in = torch.nn.init._calculate_correct_fan(x, "fan_in")
        return x + fan_in

    @make_test
    def test_is_complex(x):
        # [FUNCTION DOC] Branches on torch.is_complex.
        if torch.is_complex(x):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_tensor_is_complex(x):
        # [FUNCTION DOC] Branches on the tensor method is_complex.
        if x.is_complex():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_tensor_size(x):
        # [FUNCTION DOC] Calls the unbound Tensor.size function on x plus 1.
        fn = torch.Tensor.size
        return fn(x + 1)

    @make_test
    def test_tensor_dim(x):
        # [FUNCTION DOC] Calls the unbound Tensor.dim function on x plus 1.
        fn = torch.Tensor.dim
        return fn(x + 1)

    def test_is_inference_recompilation(self):
        # [FUNCTION DOC] Compiles a function branching on is_inference with a counting backend; the
        # normal tensor compiles once and an inference-mode tensor triggers a recompile (count 2).
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 for an inference tensor, else x minus 1.
            if x.is_inference():
                return x + 1
            else:
                return x - 1

        with torch.inference_mode():
            x_inference = torch.randn(2, 2)

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)
        x = torch.randn(2, 2)

        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnts.frame_count, 1)

        self.assertEqual(fn(x_inference), opt_fn(x_inference))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_is_inference_mode_global_recompilation(self):
        # [FUNCTION DOC] Compiles with a counting backend a function that branches on whether
        # global inference mode is enabled; asserts it matches eager and compiles one frame.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 if inference mode is enabled, else x minus 1.
            if torch.is_inference_mode_enabled():
                return x + 1
            else:
                return x - 1

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=False)

        x = torch.randn(2, 2)

        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(cnts.frame_count, 1)

    @make_test
    def test_get_privateuse1_name(x):
        # [FUNCTION DOC] Branches on whether the privateuse1 backend name equals the default value.
        if torch._C._get_privateuse1_backend_name() == "privateuseone":
            return x + 1
        else:
            return x - 1

    @make_test
    def test_device(x):
        # [FUNCTION DOC] Returns x plus 1 when the tensor device type is cpu; otherwise None.
        if x.device.type == "cpu":
            return x + 1

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    @make_test
    def test_get_device_properties_tensor_device(a):
        # [FUNCTION DOC] Skipped without CUDA. Moves a to cuda, reads device properties for that
        # device and adds multi_processor_count if the major version is 8, else the max threads
        # per multiprocessor.
        x = a.to("cuda")
        prop = torch.cuda.get_device_properties(x.device)
        if prop.major == 8:
            return x + prop.multi_processor_count
        return x + prop.max_threads_per_multi_processor

    @make_test
    def test_tensor_type(a, b):
        # [FUNCTION DOC] Casts a to float16 and casts b to the type string of that tensor.
        m = a.to(torch.float16)
        return b.type(m.type())

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    @make_test
    def test_tensor_type2(a, b):
        # [FUNCTION DOC] Skipped without a GPU. Moves a to the accelerator device and adds b cast
        # to the resulting tensor type string.
        m = a.to(device_type)
        return m + b.type(m.type())

    @make_test
    def test_tensor_type3(a, b):
        # [FUNCTION DOC] Casts a to the HalfTensor type object and b to its type string.
        m = a.type(torch.HalfTensor)
        return b.type(m.type())

    @make_test
    def test_tensor_type4(a, b):
        # [FUNCTION DOC] Casts a using the string torch.HalfTensor and b to the result's
        # type string.
        m = a.type("torch.HalfTensor")
        return b.type(m.type())

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    @make_test
    def test_tensor_type5(a, b):
        # [FUNCTION DOC] Skipped without a GPU. Moves a to the device as half and casts b to
        # that type.
        m = a.to(device_type).half()
        return b.type(m.type())

    @make_test
    def test_tensor_element_size(a):
        # [FUNCTION DOC] Branches on element_size of a; returns a pair of a +/- element size in an
        # order depending on whether it exceeds 1.
        if a.element_size() > 1:
            return (a + a.element_size(), a - a.element_size())
        return (a - a.element_size(), a + a.element_size())

    @make_test
    def test_ndim(x):
        # [FUNCTION DOC] Returns x plus 1 if ndim, ndimension and dim all equal 2; otherwise None.
        if x.ndim == 2 and x.ndimension() == 2 and x.dim() == 2:
            return x + 1

    @make_test
    def test_T(x):
        # [FUNCTION DOC] Returns ones with the shape of x.T.
        return torch.ones_like(x.T)

    @make_test
    def test_mT(x):
        # [FUNCTION DOC] Returns ones with the shape of x.mT.
        return torch.ones_like(x.mT)

    @make_test
    def test_is_sparse(x):
        # [FUNCTION DOC] Returns x plus 1 for a non-sparse tensor; otherwise None.
        if not x.is_sparse:
            return x + 1

    @make_test
    def test_shape1(x):
        # [FUNCTION DOC] Returns x plus 1 if the first dimension is 10; otherwise None.
        if x.shape[0] == 10:
            return x + 1

    @make_test
    def test_shape2(x):
        # [FUNCTION DOC] Returns x plus 1 if size of dim 1 is 10; otherwise None.
        if x.size(1) == 10:
            return x + 1

    @make_test
    def test_del(a, b):
        # [FUNCTION DOC] Deletes two local variables after use and returns b plus d.
        c = a + 1
        d = c + 2
        del c, a
        return b + d

    @make_test
    def test_chunks1(x):
        # [FUNCTION DOC] Asserts the first dim is divisible by 5 and equals 2 chunks; returns the
        # difference of the two chunks.
        chunk_size = 5
        assert x.shape[0] % chunk_size == 0  # noqa: S101
        assert x.shape[0] // chunk_size == 2  # noqa: S101
        return x[:chunk_size] - x[chunk_size:]

    @make_test
    def test_import1(x, y):
        # [FUNCTION DOC] Imports torch and sub inside the function body and returns sub applied to
        # the sum of x and y, and y.
        import torch
        from torch import sub

        return sub(torch.add(x, y), y)

    @make_test
    def test_isinstance(x):
        # [FUNCTION DOC] Runs isinstance of a list against list, tuple, Sequence from
        # collections.abc, typing Sequence, a tuple of types and a union type; appends sin or cos
        # of x per answer.
        results = []
        if isinstance([x], list):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], tuple):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], collections.abc.Sequence):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], typing.Sequence):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], (tuple, list, typing.Sequence)):
            results.append(x.sin())
        else:
            results.append(x.cos())
        if isinstance([x], list | tuple):
            results.append(x.sin())
        else:
            results.append(x.cos())
        return results

    @make_test
    def test_return_dict(x, y):
        # [FUNCTION DOC] Returns a dict holding x, a shared list z (aliased under two keys) and
        # x again.
        z = [x + y, y, False]
        return {"x": x, "z": z, "a": x, "b": z, "c": x}

    @make_test
    def test_return_dict2(x, y):
        # [FUNCTION DOC] Builds a dict incrementally, appends to a list stored inside it, and
        # returns it.
        tmp = {"x": x}
        tmp["z"] = [x + y, y]
        tmp["y"] = y
        tmp["z"].append(False)
        return tmp

    @make_test
    def test_funcdef_closure(x, y):
        # [FUNCTION DOC] Defines a closure with nonlocal x and y and calls it twice with floats,
        # updating both; returns (x, y).
        x = x + y + 1.0

        def inner(z):
            # [FUNCTION DOC] HELPER: Rebinds enclosing y to x plus z plus 20 and then x to y plus z
            # plus 10.
            nonlocal x, y
            y = x + z + 20.0
            x = y + z + 10.0

        inner(2.0)
        inner(3.0)

        return x, y

    @make_test
    def test_module_constant(x, y):
        # [FUNCTION DOC] Divides x plus y by y repeatedly, looping a number of times from the
        # testing module constant three.
        r = x + y
        for _ in range(torch._dynamo.testing.three):
            r = r / y
        return r

    @make_test
    def test_inline_softmax(x, y):
        # [FUNCTION DOC] Applies a Softmax module over the last dim to x plus twice y (common in
        # HF models).
        # This is common in some huggingface models
        return torch.nn.Softmax(dim=-1)(x + y * 2)

    @make_test
    def test_dtype_compare(a, b):
        # [FUNCTION DOC] Compares dtype with float16 and float32; returns a plus 10 or a minus 32
        # times b; None otherwise.
        if a.dtype == torch.float16:
            return a + 10
        if a.dtype == torch.float32:
            return a - b * 32

    @make_test
    def test_build_list_unpack(a, b):
        # [FUNCTION DOC] Builds two generator expressions and concatenates their unpacked contents
        # along the last dim.
        it1 = (x + 1 for x in (a, b))
        it2 = (x - 1 for x in (a, b))
        return torch.cat([*it1, *it2], dim=-1)

    @make_test
    def test_tensor_len(a, b):
        # [FUNCTION DOC] Adds a, b, the length of a and the dunder len of b.
        return a + b + len(a) + b.__len__()

    @make_test
    def test_pop(a, b):
        # [FUNCTION DOC] Exercises list append, extend, pop with index -1, 0 and default; unpacks
        # the remaining two items and returns their difference.
        ll = [a, b]
        ll.append(a + 1)
        ll.extend(
            [
                b + 2,
                a + b,
            ]
        )
        ll.pop(-1)
        ll.pop(0)
        ll.pop()
        v1, v2 = ll
        return v1 - v2

    @make_test
    def test_list_convert(a, b):
        # [FUNCTION DOC] Round trips a list through tuple and list, unpacks and returns difference
        # plus tmp.
        ll = [a + 2, b]
        ll = tuple(ll)
        tmp = b + 3
        ll = list(ll)
        v1, v2 = ll
        return v1 - v2 + tmp

    @make_test
    def test_list_add(a, b):
        # [FUNCTION DOC] Adds a tuple and an empty constant tuple, then sums the two elements.
        l1 = (a, b)
        l2 = ()  # being a LOAD_CONST in the bytecode
        l3 = l1 + l2
        return l3[0] + l3[1]

    @make_test
    def test_list_index_with_constant_tensor(a, b):
        # [FUNCTION DOC] Indexes a list with a constant one-element tensor created from 2.
        l1 = [a, b, a + 1, b + 1]
        return l1[torch.as_tensor(2)]

    @make_test
    def test_startswith(a, b):
        # [FUNCTION DOC] Checks string startswith and that the module name of constant3
        # contains test.
        x = a + b
        if "foobar".startswith("foo") and "test" in constant3.__module__:
            x = x + 1
        return x

    @make_test
    def test_dict_ops(a, b):
        # [FUNCTION DOC] Exercises dict get, pop with and without default, update, setitem,
        # membership and len; returns a combined tensor result if the final condition holds.
        tmp = {"a": a + 1, "b": b + 2}
        assert tmp.get("zzz") is None  # noqa: S101
        v = tmp.pop("b") + tmp.get("a") + tmp.get("missing", 3) + tmp.pop("missing", 4)
        tmp.update({"d": 3})
        tmp["c"] = v + tmp["d"]
        if "c" in tmp and "missing" not in tmp:
            return tmp["c"] - tmp["a"] + len(tmp)

    @make_test
    def test_inline_jit__unwrap_optional(x):
        # [FUNCTION DOC] Calls the jit unwrap_optional helper on x; returns ones if None else sin
        # of x.
        if torch.jit._unwrap_optional(x) is None:
            return torch.ones(2, 2)
        return x.sin()

    @make_test
    def test_zip_longest(x):
        # [FUNCTION DOC] Returns sin of x plus 1 and a list from zip_longest of three lists with
        # fill None.
        list1 = [1, 2, 3]
        list2 = ["a", "b"]
        list3 = [True, False, True, False]
        return torch.sin(x + 1), list(
            itertools.zip_longest(list1, list2, list3, fillvalue=None)
        )

    def test_torch_size_as_dict_key(self):
        # [FUNCTION DOC] Compiles a function that uses a tensor shape as a dict key in a passed-in
        # cache; asserts results equal eager with fresh caches for both runs.
        def fn(x, cached):
            # [FUNCTION DOC] HELPER: Store x in the cache under its shape if missing; return x plus
            # the entry.
            if x.shape not in cached:
                cached[x.shape] = x
            return x + cached[x.shape]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x1 = torch.randn(2, 3)
        x2 = torch.randn(2, 3)
        cached = {}
        ref1 = fn(x1, cached)
        ref2 = fn(x2, cached)
        cached = {}
        res1 = opt_fn(x1, cached)
        res2 = opt_fn(x2, cached)
        self.assertEqual(ref1, res1)
        self.assertEqual(ref2, res2)

    def test_dict_param_keys(self):
        # [FUNCTION DOC] Builds a make_test test from a function using a Parameter as a dict key
        # alongside a string key, and runs it as this test case.
        a_param = torch.nn.Parameter(torch.ones([4, 4]))

        def fn(a):
            # [FUNCTION DOC] HELPER: Return the string-keyed tensor plus the value stored under the
            # Parameter key.
            tmp = {"a": a, a_param: 3}
            return tmp["a"] + tmp[a_param]

        test = make_test(fn)
        test(self)

    def test_dict_mutable_map(self):
        # [FUNCTION DOC] Compiles a function copying a custom MutableMapping into a dict and
        # indexing it; asserts equality with eager.
        from collections.abc import MutableMapping

        class TensorDict(MutableMapping):
            # [CLASS DOC] HELPER: MutableMapping backed by an internal dict, with an add method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the empty backing dict.
                self._dict = {}

            def add(self, key, value):
                # [FUNCTION DOC] HELPER: Store value under key in the backing dict.
                self._dict[key] = value

            def items(self):
                # [FUNCTION DOC] HELPER: Return the items view of the backing dict.
                return self._dict.items()

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Delete the key from the backing dict.
                del self._dict[key]

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the value for key from the backing dict.
                return self._dict[key]

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return an iterator over the keys.
                return iter(self._dict)

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of entries.
                return len(self._dict)

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store value under key in the backing dict.
                self._dict[key] = value

        tensor_dict = TensorDict()
        tensor_dict.add("a", torch.ones(4) * 2)

        def fn(x):
            # [FUNCTION DOC] HELPER: Copy the mapping into a dict and multiply x by its entry a.
            copy_tensordict = dict(tensor_dict)
            return x * copy_tensordict["a"]

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_unpack_mutable_map(self):
        # [FUNCTION DOC] Compiles a function that passes a custom MutableMapping as keyword
        # arguments via double-star unpacking; asserts equality with eager.
        from collections.abc import MutableMapping

        class TensorDict(MutableMapping):
            # [CLASS DOC] HELPER: MutableMapping backed by an internal dict, with an add method.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create the empty backing dict.
                self._dict = {}

            def add(self, key, value):
                # [FUNCTION DOC] HELPER: Store value under key in the backing dict.
                self._dict[key] = value

            def items(self):
                # [FUNCTION DOC] HELPER: Return the items view of the backing dict.
                return self._dict.items()

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Delete the key from the backing dict.
                del self._dict[key]

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the value for key from the backing dict.
                return self._dict[key]

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return an iterator over the keys.
                return iter(self._dict)

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of entries.
                return len(self._dict)

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store value under key in the backing dict.
                self._dict[key] = value

        tensor_dict = TensorDict()
        tensor_dict.add("a", torch.ones(4) * 2)

        def gn(x, a=1):
            # [FUNCTION DOC] HELPER: Return x times a (default 1).
            return x * a

        def fn(x):
            # [FUNCTION DOC] HELPER: Call gn with the mapping unpacked as keyword arguments.
            return gn(x, **tensor_dict)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.randn(4)

        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_mutable_mapping_property_setter_nested(self):
        # [FUNCTION DOC] Regression test for issue 172000: property setters on MutableMapping
        # objects created inside compiled code. Compares a direct method call and the names
        # property setter on a nested container, asserting both set the names on root and
        # nested objects.
        """Test that property setters work correctly on nested MutableMapping objects.

        This tests the fix for a bug where property setters on newly created
        MutableMapping objects would fail when accessing nested objects.
        The issue was that property.__set__ is a slot wrapper (not a Python function),
        so Dynamo wasn't tracing the property setter (fset), causing the setter code
        to run on uninitialized example objects.

        See: https://github.com/pytorch/pytorch/issues/172000
        """
        from collections.abc import MutableMapping

        class Container(MutableMapping):
            # [CLASS DOC] HELPER: MutableMapping holding data, batch size and optional names;
            # setting names propagates a padded names list to nested Container values.
            def __init__(self, data=None, batch_size=None):
                # [FUNCTION DOC] HELPER: Store data (default empty dict), batch size (default empty
                # tuple), names None.
                self._data = data if data is not None else {}
                self._batch_size = batch_size if batch_size is not None else ()
                self._names = None

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the stored value for key.
                return self._data[key]

            def __setitem__(self, key, value):
                # [FUNCTION DOC] HELPER: Store value under key.
                self._data[key] = value

            def __delitem__(self, key):
                # [FUNCTION DOC] HELPER: Delete the key from the data.
                del self._data[key]

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Return an iterator over the keys.
                return iter(self._data)

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of entries.
                return len(self._data)

            @property
            def batch_size(self):
                # [FUNCTION DOC] HELPER: Property returning the stored batch size.
                return self._batch_size

            @property
            def names(self):
                # [FUNCTION DOC] HELPER: Property returning the stored names.
                return self._names

            @names.setter
            def names(self, value):
                # [FUNCTION DOC] HELPER: Property setter that delegates to the private
                # set-names method.
                self._set_names(value)

            def _set_names(self, value):
                # [FUNCTION DOC] HELPER: Set names; for each nested Container value set its names
                # to the list of value padded with None up to the child batch size length.
                self._names = value
                for item in self._data.values():
                    if isinstance(item, Container):
                        child_size = len(item.batch_size)
                        item._names = list(value) + [None] * (child_size - len(value))

        # Test that both direct method call and property setter work identically
        @torch.compile(backend="eager", fullgraph=True)
        def using_method(x):
            # [FUNCTION DOC] HELPER: Compiled; build nested and root containers, call the
            # set-names method.
            nested = Container({"b": x}, (3,))
            root = Container({"nested": nested}, (3,))
            root._set_names(["time"])
            return root, x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def using_property_setter(x):
            # [FUNCTION DOC] HELPER: Compiled; build nested and root containers, assign names
            # via property.
            nested = Container({"b": x}, (3,))
            root = Container({"nested": nested}, (3,))
            root.names = ["time"]
            return root, x + 1

        x = torch.tensor(2.0)

        # Both should work without error
        result_method, _ = using_method(x)
        self.assertEqual(result_method._names, ["time"])
        self.assertEqual(result_method["nested"]._names, ["time"])

        result_property, _ = using_property_setter(x)
        self.assertEqual(result_property._names, ["time"])
        self.assertEqual(result_property["nested"]._names, ["time"])

    def _test_default_dict_helper(self, factory):
        # [FUNCTION DOC] HELPER: Shared body for the default dict tests. Builds a defaultdict with
        # the given factory, compiles a function that stores keys including a Parameter key, reads
        # a missing key (creating a default) and returns it with the dict; asserts compiled and
        # eager values agree.
        dd = collections.defaultdict(factory)
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            # [FUNCTION DOC] HELPER: Mutate the captured defaultdict and return the default of
            # missing key b with the dict.
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_default_dict_dict(self):
        # [FUNCTION DOC] Runs the default dict helper with dict as factory.
        self._test_default_dict_helper(dict)

    def test_default_dict_list(self):
        # [FUNCTION DOC] Runs the default dict helper with list as factory.
        self._test_default_dict_helper(list)

    def test_default_dict_tuple(self):
        # [FUNCTION DOC] Runs the default dict helper with tuple as factory.
        self._test_default_dict_helper(tuple)

    def test_default_dict_set(self):
        # [FUNCTION DOC] Runs the default dict helper with set as factory.
        self._test_default_dict_helper(set)

    def test_default_dict_lambda(self):
        # [FUNCTION DOC] Runs the default dict helper with a lambda factory.
        self._test_default_dict_helper(lambda: dict())

    def test_default_dict_closure(self):
        # [FUNCTION DOC] Runs the default dict helper with a local closure factory.
        def factory():
            # [FUNCTION DOC] HELPER: Factory returning a new empty dict.
            return dict()

        self._test_default_dict_helper(factory)

    def test_class_dict(self):
        # [FUNCTION DOC] Checks the class dict of an instance's type via a membership test inside
        # compiled code.
        class A:
            # [CLASS DOC] HELPER: Class with class attributes x and y and instance attribute a set
            # in init.
            x = 4
            y = 5

            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Set instance attribute a to 6.
                self.a = 6

        a = A()

        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 if x is a key in the dict of A, else x plus 2.
            if "x" in type(a).__dict__:
                return x + 1
            return x + 2

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_default_dict_constr(self):
        # [FUNCTION DOC] Compiles a function that builds a defaultdict, uses setitem, a Parameter
        # key and updates from a mapping, list of pairs and zip; asserts all entries equal eager.
        param = torch.nn.Parameter(torch.ones([2, 2]))

        def fn(x):
            # [FUNCTION DOC] HELPER: Fill a defaultdict with several update styles; return key b
            # and the dict.
            dd = collections.defaultdict(lambda: dict())
            dd["a"] = x + 1
            dd[param] = 123
            dd["c"] = x * 2
            dd.update({"b": x * 3})
            dd.update([["d", x - 2], ("e", x + 2)])
            dd.update(zip("ab", [x + 3, x + 4]))
            return dd["b"], dd

        x = torch.randn(10, 10)
        ref = fn(x)
        opt_fn = torch._dynamo.optimize_assert("eager")(fn)
        res = opt_fn(x)

        self.assertTrue(same(ref[0], res[0]))
        self.assertTrue(same(ref[1]["a"], res[1]["a"]))
        self.assertTrue(same(ref[1]["b"], res[1]["b"]))
        self.assertTrue(same(ref[1]["c"], res[1]["c"]))
        self.assertTrue(same(ref[1]["d"], res[1]["d"]))
        self.assertTrue(same(ref[1]["e"], res[1]["e"]))
        self.assertTrue(same(ref[1][param], res[1][param]))

    def test_dict_tuple_lazy_guard(self):
        # [FUNCTION DOC] Checks guard laziness: after compiling with a dict argument, changing the
        # value of an unused key causes no recompile (error_on_recompile is patched on); the same
        # for a tuple.
        @torch.compile(backend="eager")
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return sin of x times the element at index 1 of y.
            return torch.sin(x) * y[1]

        fn(torch.randn(3), {1: 1, 2: 2})
        # Changing the value of other key should not causing recompilation
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(3), {1: 1, 2: 3})

        fn(torch.randn(3), (1, 2, 3))
        # Changing the value of index 0, 2 (not 1) should not cause recompilation
        with unittest.mock.patch("torch._dynamo.config.error_on_recompile", True):
            fn(torch.randn(3), (11, 2, 13))

    @make_test
    def test_call_dict1(x):
        # [FUNCTION DOC] Builds dict and OrderedDict by setitem and sums their entries plus 1.
        d1 = dict()
        d1["x"] = x + 1
        d2 = collections.OrderedDict()
        d2["x"] = x + 2
        return d1["x"] + d2["x"] + 1

    @make_test
    def test_call_dict2(x):
        # [FUNCTION DOC] Builds an OrderedDict from a dict and branches on its isinstance result.
        d1 = dict()
        d1["x"] = x
        d2 = collections.OrderedDict(d1)
        if isinstance(d2, collections.OrderedDict):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_call_dict3(x):
        # [FUNCTION DOC] Builds dict and OrderedDict from a list of pairs, mutates each,
        # sums entries.
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict4(x):
        # [FUNCTION DOC] Same as the list-of-pairs test but from a tuple of pairs.
        my_list = (("a", x), ("b", x + 1), ("c", x + 2))
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_call_dict5(x):
        # [FUNCTION DOC] Same but from an iterator of pairs; the second construction sees the
        # exhausted iterator so d2 is empty before its assignment.
        my_list = iter([("a", x), ("b", x + 1), ("c", x + 2)])
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = collections.OrderedDict(my_list)
        d2["c"] = x + 20
        return d1["a"] + d2["c"] + 1

    @make_test
    def test_dict_fromkeys(x, y):
        # [FUNCTION DOC] Uses fromkeys on dict, defaultdict and OrderedDict with different keys and
        # values (including a keyword value) and combines entries.
        lst = ["a", "b"]
        d = dict.fromkeys(lst)
        d1 = dict.fromkeys(d, x + 1)
        d2 = collections.defaultdict.fromkeys(iter(d1), x - 2)
        d3 = collections.OrderedDict.fromkeys(tuple(lst), value=y)
        return d1["a"] * d2["b"] + d2["a"] + d1["b"] + d3["a"] + d3["b"] + 1

    @make_test
    def test_dict_copy(x):
        # [FUNCTION DOC] Copies a dict and an OrderedDict, mutates the copies and combines entries.
        my_list = [("a", x), ("b", x + 1), ("c", x + 2)]
        d1 = dict(my_list)
        d1["a"] = x + 10
        d2 = d1.copy()
        d2["a"] = x - 5
        d2["b"] = x + 3
        d3 = collections.OrderedDict(my_list)
        d3["c"] = x + 20
        d4 = d3.copy()
        d4["c"] = x - 10
        return d1["a"] * d2["a"] + d2["b"] + d3["c"] * d4["c"] + 1

    @make_test
    def test_dict_update(x, y, z):
        # [FUNCTION DOC] Exercises update on dict and OrderedDict with a mapping, list/tuple pairs
        # and zip; returns a combined tensor expression.
        d = {"a": x, "b": y}
        d.update({"a": y - 1})
        d.update([("b", z + 1), ["c", z]])
        d.update(zip("ab", [z + 3, y + 2]))

        od = collections.OrderedDict(a=x * 3, b=y + 2)
        od.update({"a": y + 5})
        od.update([["b", z + 6], ("c", z - 7)])
        od.update(zip("ab", [z - 3, x + 2]))
        return d["a"] * od["a"] + od["c"] + d["b"] + od["b"] * d["c"]

    @make_test
    def test_min_max(a, b):
        # [FUNCTION DOC] Reduces a and b to sums, clamps via nested min and max and combines
        # with c.
        c = a + b
        a = a.sum()
        b = b.sum()
        a = min(max(a, 0), 1)
        b = max(0, min(1, b))
        return max(a, b) - min(a, b) + c

    @make_test
    def test_symbool_to_int(x):
        # [FUNCTION DOC] Sums a generator comparing each size to -1 (einops pattern) and branches
        # on zero.
        # this is roughly the pattern found in einops.unpack()
        if sum(s == -1 for s in x.size()) == 0:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_map_sum(a, b, c, d):
        # [FUNCTION DOC] Sums the results of mapping a lambda that adds 1 over four tensors.
        return sum(map(lambda x: x + 1, [a, b, c, d]))

    @make_test
    def test_sum(a, b, c, d):
        # [FUNCTION DOC] Sums a list of four tensors.
        return sum([a, b, c, d])

    @make_test
    def test_sum_with_start_arg(a, b, c, d):
        # [FUNCTION DOC] Sums three tensors with a as the positional start value.
        return sum([b, c, d], a)

    @make_test
    def test_sum_with_start_kwarg(a, b, c, d):
        # [FUNCTION DOC] Sums three tensors with a as the keyword start value.
        return sum([b, c, d], start=a)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut():
        # [FUNCTION DOC] Takes no tensor input and sums constants; expects zero compiled frames.
        return sum([0, 1.0, 2, 3.0])

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_arg():
        # [FUNCTION DOC] Same with a positional start of -10; expects zero compiled frames.
        return sum([0, 1.0, 2, 3.0], -10)

    @make_test(expected_frame_count=0)
    def test_sum_shortcut_with_start_kwarg():
        # [FUNCTION DOC] Same with a keyword start of -10; expects zero compiled frames.
        return sum([0, 1.0, 2, 3.0], start=-10)

    @make_test
    def test_reduce(a, b, c, d):
        # [FUNCTION DOC] Folds add over four tensors with functools reduce.
        return functools.reduce(operator.add, [a, b, c, d])

    @make_test
    def test_reduce_with_initial(a, b, c, d):
        # [FUNCTION DOC] Folds add over three tensors with a as the initial value.
        return functools.reduce(operator.add, [b, c, d], a)

    @make_test
    def test_reduce_with_single(x):
        # [FUNCTION DOC] Reduce over a single-element list without initial value (returns
        # the element).
        return functools.reduce(lambda a, b: (a, b), [x])

    @make_test(expected_frame_count=0)
    def test_reduce_with_single_with_initial(x, y):
        # [FUNCTION DOC] Reduce with initial value x over a one-element list; expects zero
        # compiled frames.
        return functools.reduce(lambda a, b: (a, b), [y], x)

    @make_test(expected_frame_count=0)
    def test_reduce_with_none_initial(x):
        # [FUNCTION DOC] Reduce with initial None over a one-element list; expects zero
        # compiled frames.
        return functools.reduce(lambda a, b: (a, b), [x], None)

    @make_test
    def test_tuple_contains(a, b):
        # [FUNCTION DOC] Checks membership and non-membership in two string tuples; returns a plus
        # b or a minus b.
        v1 = "a"
        v2 = "b"
        v3 = "c"
        vals1 = (v1, v2, v3)
        vals2 = ("d", "e", "f")
        if "a" in vals1 and "b" not in vals2:
            return a + b
        return a - b

    @make_test
    def test_set_in_frozenset(x):
        # [FUNCTION DOC] Checks a set of characters is found inside a set containing a frozenset.
        var = set("abc")
        other = set([frozenset("abc")])
        if var in other:
            return x + 1
        else:
            return x - 1

    @make_test
    def test_set_update_bytecode(x):
        # [FUNCTION DOC] Builds a set literal (set-update bytecode) and branches on isinstance
        # of set.
        # This produces bytecode SET_UPDATE since python 3.9
        var = {"apple", "banana", "cherry"}
        if isinstance(var, set):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_set_update_list_with_duplicated_items(x):
        # [FUNCTION DOC] Unions two lists with duplicates via unpacking into a set and checks size
        # is 3.
        list1 = ["apple", "banana", "apple"]
        list2 = ["orange", "banana"]
        if len({*list1, *list2}) == 3:
            return x + 1
        else:
            return x - 1

    def test_set_keys_view(self):
        # [FUNCTION DOC] Compiles a function converting a custom KeysView subclass to a set and
        # multiplying x by its length; asserts equality with eager.
        from collections.abc import KeysView

        class StringKeys(KeysView):
            # [CLASS DOC] HELPER: KeysView subclass wrapping a list of keys with indexing,
            # iteration, repr, length and membership delegated to that list.
            def __init__(self, keys):
                # [FUNCTION DOC] HELPER: Store the keys list in an instance attribute.
                self.keys = keys

            def __getitem__(self, key):
                # [FUNCTION DOC] HELPER: Return the item at the given index or key of the
                # wrapped keys.
                return self.keys.__getitem__(key)

            def __iter__(self):
                # [FUNCTION DOC] HELPER: Generator yielding each wrapped key.
                yield from self.keys

            def __repr__(self):
                # [FUNCTION DOC] HELPER: Return a string with the class name and the keys.
                return f"{type(self).__name__}({self.keys})"

            def __len__(self):
                # [FUNCTION DOC] HELPER: Return the number of wrapped keys.
                return len(self.keys)

            def __contains__(self, item):
                # [FUNCTION DOC] HELPER: Return whether the item is among the wrapped keys.
                return self.keys.__contains__(item)

        a = StringKeys([1, 2, 3, 3])

        def fn(x):
            # [FUNCTION DOC] HELPER: Convert the captured keys view to a set and scale x by
            # its length.
            set_a = set(a)
            return len(set_a) * x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.rand(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_constant_set(self):
        # [FUNCTION DOC] Compiles a function reading the length of a captured set; mutates the set
        # afterward and expects the compiled function to give the updated result (recompilation).
        s = set([1, 2])

        def fn(x):
            # [FUNCTION DOC] HELPER: Return cos of x times the length of the captured set.
            return torch.cos(x) * len(s)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.rand(4)
        self.assertEqual(fn(x), opt_fn(x))

        # This should cause recompilation
        s.add(3)
        self.assertEqual(fn(x), opt_fn(x))

    def test_set_add(self):
        # [FUNCTION DOC] Compiles a function that adds an element to a captured set; asserts
        # compiled output equals eager and the set still has three entries (the add is idempotent).
        s = set([1, 2])

        def fn(x):
            # [FUNCTION DOC] HELPER: Add 3 to the captured set and return cos of x times len of x.
            s.add(3)
            return torch.cos(x) * len(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.rand(4)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertEqual(len(s), 3)

    @make_test
    def test_tuple_iadd(a, b):
        # [FUNCTION DOC] Extends a tuple with in-place add of another tuple and returns it.
        output = (a, b)
        output += (a + b, a - b)
        return output

    @make_test
    def test_unpack_ex1(x):
        # [FUNCTION DOC] Star-unpacks a 4-tuple as first, second and rest; returns a minus b
        # divided by the first item of rest.
        output = (x, x + 1, x + 2, x + 3)
        a, b, *cd = output
        return a - b / cd[0]

    @make_test
    def test_unpack_ex2(x):
        # [FUNCTION DOC] Star-unpacks a 4-tuple as leading rest and last two; returns c minus d
        # divided by rest[0].
        output = (x, x + 1, x + 2, x + 3)
        *ab, c, d = output
        return c - d / ab[0]

    @make_test
    def test_unpack_ex3(x):
        # [FUNCTION DOC] Star-unpacks a 4-tuple as first, middle rest, last; returns a minus d
        # divided by rest[0].
        output = (x, x + 1, x + 2, x + 3)
        a, *bc, d = output
        return a - d / bc[0]

    @make_test
    def test_const_tuple_add1(x):
        # [FUNCTION DOC] Adds empty constant tuples on both sides of a tensor tuple; sums elements
        # 2 and 3.
        output = (x, x + 1, x + 2, x + 3)
        output = () + output + ()
        return output[2] + output[3]

    @make_test
    def test_const_tuple_add2(x):
        # [FUNCTION DOC] Pads a tensor tuple with one-element None tuples; sums elements 2 and 3.
        output = (x, x + 1, x + 2, x + 3)
        output = (None,) + output + (None,)
        return output[2] + output[3]

    @make_test
    def test_list_truth(a, b):
        # [FUNCTION DOC] Branches on the truthiness of a non-empty constant list.
        tmp = [1, 2, 3]
        if tmp:
            return a + b
        else:
            return a - b

    @make_test
    def test_list_reversed(a, b):
        # [FUNCTION DOC] Takes the first item of the reversed iterator of a list and adds it to a
        # plus b.
        tmp = [a + 1, a + 2, a + 3]
        return a + b + next(iter(reversed(tmp)))

    @make_test
    def test_list_sorted1(x):
        # [FUNCTION DOC] Returns x plus 1 and four sorted views of a constant int list (default,
        # key None, reverse, key None with reverse).
        tmp = [1, 10, 3, 0]
        return (
            x + 1,
            sorted(tmp),
            sorted(tmp, key=None),
            sorted(tmp, reverse=True),
            sorted(tmp, key=None, reverse=True),
        )

    @make_test
    def test_list_sorted2(x):
        # [FUNCTION DOC] Sorts a list of tuples with default order, key None, reverse and key
        # functions selecting the third field, returning x plus 1 and all results.
        y = [
            ("john", "A", 8),
            ("jane", "B", 5),
            ("dave", "B", 10),
        ]
        return (
            x + 1,
            sorted(y),
            sorted(y, key=None),
            sorted(y, reverse=True),
            sorted(y, key=None, reverse=True),
            sorted(y, key=lambda student: student[2]),
            sorted(y, key=lambda student: student[2], reverse=True),
        )

    @make_test
    def test_tuple_sorted(x):
        # [FUNCTION DOC] Same four sorted variants applied to a constant tuple of ints.
        tmp = (1, 10, 3, 0)
        return (
            x + 1,
            sorted(tmp),
            sorted(tmp, key=None),
            sorted(tmp, reverse=True),
            sorted(tmp, key=None, reverse=True),
        )

    @make_test
    def test_dict_sorted(x):
        # [FUNCTION DOC] Same four sorted variants applied to a dict, which sorts its keys.
        tmp = {1: "D", 10: "B", 3: "E", 0: "F"}
        return (
            x + 1,
            sorted(tmp),
            sorted(tmp, key=None),
            sorted(tmp, reverse=True),
            sorted(tmp, key=None, reverse=True),
        )

    @make_test
    def test_dict_items_sorted(x):
        # [FUNCTION DOC] Sorts dict items by pair, by key and by value using itemgetter keys, each
        # also in reverse; returns x plus 1 and all results.
        tmp = {1: "D", 10: "B", 3: "E", 0: "F"}
        return (
            x + 1,
            # ordered by (key, value) pair
            sorted(tmp.items()),
            sorted(tmp.items(), key=None),
            sorted(tmp.items(), reverse=True),
            sorted(tmp.items(), key=None, reverse=True),
            # ordered by key
            sorted(tmp.items(), key=operator.itemgetter(0)),
            sorted(tmp.items(), key=operator.itemgetter(0), reverse=True),
            # ordered by value
            sorted(tmp.items(), key=operator.itemgetter(1)),
            sorted(tmp.items(), key=operator.itemgetter(1), reverse=True),
        )

    @make_test
    def test_sorted_const_key_non_const_items(x, y):
        # [FUNCTION DOC] Sorts a dict with constant keys and tensor values by key (keys only and
        # items with itemgetter, forward and reverse).
        tmp = {1: x, 10: x - 1, 3: 2 * x, -1: y + 2, 0: torch.ones(3, 4)}
        return (
            sorted(tmp),
            sorted(tmp.items(), key=operator.itemgetter(0)),
            sorted(tmp.items(), key=operator.itemgetter(0), reverse=True),
        )

    def test_dict_hasattr(self):
        # [FUNCTION DOC] Compiles a function that uses hasattr to dispatch on a dict (items) or
        # tensor (to); asserts compiled equals eager for both kinds of input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x moved to cpu if it has to, cos of key a if it has
            # items, else x.
            if hasattr(x, "to"):
                return x.to("cpu")
            if hasattr(x, "items"):
                return torch.cos(x["a"])
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = dict(a=torch.randn(3))
        self.assertEqual(fn(x), opt_fn(x))

        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    @make_test
    def test_list_clear(a, b):
        # [FUNCTION DOC] Clears a list and appends a plus b; returns the list.
        tmp = [a + 1, a + 2]
        tmp.clear()
        tmp.append(a + b)
        return tmp

    @make_test
    def test_not_list(a):
        # [FUNCTION DOC] Returns the logical not of a one-element list.
        return not [a + 1]

    @make_test
    def test_islice_chain(a, b):
        # [FUNCTION DOC] Takes items 1..2 of chain of two lists and the second item of one list via
        # islice; returns a minus b divided by c.
        tmp1 = [a + 1, a + 2]
        tmp2 = [a + 3, a + 4]
        a, b = list(itertools.islice(itertools.chain(tmp1, tmp2), 1, 3))
        c = next(itertools.islice(tmp1, 1, None))
        return a - b / c

    @make_test
    def test_slice_eq(a, b):
        # [FUNCTION DOC] Checks slice isinstance and equality with a full slice; indexes a list
        # with it.
        tmp1 = [a + 1, b + 2]
        s = slice(1, 2)
        if isinstance(s, slice) and s == slice(None):
            return tmp1[s] * 2
        return tmp1[s] * 3

    @make_test
    def test_namedtuple(a, b):
        # [FUNCTION DOC] Defines a namedtuple in the function, builds it, reads fields by attribute
        # and index, and returns a new instance.
        mytuple = collections.namedtuple("mytuple", ["x", "y", "xy"])
        tmp = mytuple(a, b, a + b)
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    @make_test
    def test_namedtuple_defaults(a, b):
        # [FUNCTION DOC] Defines a namedtuple with defaults, constructs with a keyword and returns
        # a rebuilt instance.
        mytuple = collections.namedtuple(
            "mytuple", ["x", "y", "xy"], defaults=(None, 1, None)
        )
        tmp = mytuple(a, xy=b)
        return mytuple(tmp.x, tmp[1], tmp.xy + b)

    @make_test
    def test_namedtuple_replace_1(a, b):
        # [FUNCTION DOC] Calls replace on a namedtuple, discards the copy and returns t.x plus t.y
        # of the original.
        mytuple = collections.namedtuple("mytuple", ["x", "y"])
        t = mytuple(a, b)
        t._replace(x=b)
        return t.x + t.y

    @make_test
    def test_namedtuple_fields(a, b):
        # [FUNCTION DOC] Compares the _fields of a locally defined namedtuple with the tuple
        # of names.
        mytuple = collections.namedtuple("mytuple", ["x", "y"])
        if mytuple._fields == ("x", "y"):
            return a + b
        else:
            return a - b

    class MyNamedTuple(NamedTuple):
        # [CLASS DOC] HELPER: NamedTuple with tensor fields first and second plus an add method, a
        # static method returning 1 and a class method returning the class name.
        first: torch.Tensor
        second: torch.Tensor

        def add(self) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Return first plus second.
            return self.first + self.second

        @staticmethod
        def static_method() -> int:
            # [FUNCTION DOC] HELPER: Return the integer 1.
            return 1

        @classmethod
        def class_method(cls) -> str:
            # [FUNCTION DOC] HELPER: Return the name of the class.
            return cls.__name__

    class MyGenericNamedTuple(NamedTuple, Generic[T]):
        # [CLASS DOC] HELPER: Generic NamedTuple over type variable T with fields first and second
        # and the same add, static and class methods.
        first: T
        second: T

        def add(self) -> T:
            # [FUNCTION DOC] HELPER: Return first plus second.
            return self.first + self.second

        @staticmethod
        def static_method() -> int:
            # [FUNCTION DOC] HELPER: Return the integer 1.
            return 1

        @classmethod
        def class_method(cls) -> str:
            # [FUNCTION DOC] HELPER: Return the name of the class.
            return cls.__name__

    class MyNamedTupleSubclass(MyNamedTuple):
        # [CLASS DOC] HELPER: Empty subclass of MyNamedTuple.
        pass

    class MyGenericNamedTupleSubclass(MyGenericNamedTuple[T]):
        # [CLASS DOC] HELPER: Empty subclass of the generic named tuple.
        pass

    @make_test
    def test_namedtuple_user_methods(a, b):
        # [FUNCTION DOC] Builds the MyNamedTuple fixture and calls its add, static and
        # class methods.
        mytuple = FunctionTests.MyNamedTuple(a, b)
        return mytuple.add(), mytuple.static_method(), mytuple.class_method()

    @make_test
    def test_namedtuple_replace_2(a, b):
        # [FUNCTION DOC] Replaces the first field of a MyNamedTuple and sums fields of both tuples.
        mytuple = FunctionTests.MyNamedTuple(a, b)
        replaced = mytuple._replace(first=b)
        return mytuple.first + mytuple.second + replaced.first + replaced.second

    @make_test
    def test_generic_namedtuple_user_methods(a, b):
        # [FUNCTION DOC] Builds the generic namedtuple fixture and calls its add, static and
        # class methods.
        mytuple = FunctionTests.MyGenericNamedTuple(a, b)
        return mytuple.add(), mytuple.static_method(), mytuple.class_method()

    @make_test
    def test_namedtuple_hasattr(a, b):
        # [FUNCTION DOC] Checks a namedtuple fixture satisfies tuple isinstance and has asdict
        # and fields.
        mytuple = FunctionTests.MyNamedTuple(a, b)

        def isinstance_namedtuple(obj) -> bool:
            # [FUNCTION DOC] HELPER: Return whether obj is a tuple with _asdict and
            # _fields attributes.
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        if isinstance_namedtuple(mytuple):
            return a + b
        else:
            return a - b

    @make_test
    def test_namedtuple_equality(a, b):
        # [FUNCTION DOC] Compares two equal MyNamedTuple instances; adds 1.0 to a plus b when they
        # are equal.
        mytuple_1 = FunctionTests.MyNamedTuple(a, b)
        mytuple_2 = FunctionTests.MyNamedTuple(a, b)

        if mytuple_1 == mytuple_2:
            return 1.0 + a + b
        else:
            return a + b

    @make_test
    def test_generic_namedtuple_hasattr(a, b):
        # [FUNCTION DOC] Same as the namedtuple hasattr test but with the generic namedtuple.
        mytuple = FunctionTests.MyGenericNamedTuple(a, b)

        def isinstance_namedtuple(obj) -> bool:
            # [FUNCTION DOC] HELPER: Return whether obj is a tuple with _asdict and
            # _fields attributes.
            return (
                isinstance(obj, tuple)
                and hasattr(obj, "_asdict")
                and hasattr(obj, "_fields")
            )

        if isinstance_namedtuple(mytuple):
            return a + b
        else:
            return a - b

    @make_test
    def test_namedtuple_subclass(a, b):
        # [FUNCTION DOC] Sets extra attributes x, y, z (z twice) on a MyNamedTuple subclass
        # instance and returns hasattr for x, x plus y and z.
        mytuple = FunctionTests.MyNamedTupleSubclass(a, b)
        mytuple.x = a
        mytuple.y = b
        mytuple.z = b
        mytuple.z = a
        return hasattr(mytuple, "x"), mytuple.x + mytuple.y, mytuple.z

    @make_test
    def test_generic_namedtuple_subclass(a, b):
        # [FUNCTION DOC] Same attribute-setting test on the generic namedtuple subclass.
        mytuple = FunctionTests.MyGenericNamedTupleSubclass(a, b)
        mytuple.x = a
        mytuple.y = b
        mytuple.z = b
        mytuple.z = a
        return hasattr(mytuple, "x"), mytuple.x + mytuple.y, mytuple.z

    @make_test
    def test_sourceless_build_method_type(a, b):
        # [FUNCTION DOC] Creates a namedtuple class without a source and checks its _make attribute
        # is callable.
        cls = collections.namedtuple("Foo", ["x", "y"])  # sourceless variable

        # The type of `cls._make` is method type
        if callable(getattr(cls, "_make", None)):
            return a + b
        else:
            return a - b

    @make_test
    def test_torch_size_hasattr(x):
        # [FUNCTION DOC] Checks that a torch.Size has no _fields attribute (not a namedtuple).
        if hasattr(x.shape, "_fields"):
            return x + 1
        else:
            return x - 1

    @make_test
    def test_is_quantized(a, b):
        # [FUNCTION DOC] Returns a plus b if a is not quantized; otherwise None.
        if not a.is_quantized:
            return a + b

    @make_test
    def test_fstrings1(a, b):
        # [FUNCTION DOC] Formats a float constant with a format spec in an f-string and checks
        # the prefix.
        x = 1.229
        tmp = f"{x:.2f} bar"
        if tmp.startswith("1.23"):
            return a + b

    @make_test
    def test_fstrings2(x):
        # [FUNCTION DOC] Builds an f-string from the first dimension size and checks it starts
        # with 10.
        tmp = f"{x.shape[0]} bar"
        if tmp.startswith("10"):
            return x + 1

    @make_test
    def test_fstrings3(x):
        # [FUNCTION DOC] Builds an f-string from the class name of x and checks the prefix Tensor.
        tmp = f"{x.__class__.__name__} foo"
        if tmp.startswith("Tensor"):
            return x + 1

    @make_test
    def test_fstrings4(x):
        # [FUNCTION DOC] Builds an f-string from a size and checks substring membership of 10.
        tmp = f"{x.shape[0]} bar"
        if "10" in tmp:
            return x + 1

    @make_test
    def test_fstrings5(x):
        # [FUNCTION DOC] Concatenates a string to an f-string and checks substring membership.
        tmp = f"{x.shape[0]} bar"
        if "10" in (tmp + "haha"):
            return x + 1

    @make_test
    def test_fstrings6(x):
        # [FUNCTION DOC] Formats the sum of two sizes in an f-string and checks it contains 20.
        tmp = f"{x.shape[0] + x.shape[1]}"
        if "20" in tmp:
            return x + 1

    @make_test
    def test_tensor_new_with_size(x):
        # [FUNCTION DOC] Creates a tensor via x.new from a Size and asserts the same size;
        # returns None.
        y = torch.rand(5, 8)
        z = x.new(y.size())
        assert z.size() == y.size()  # noqa: S101

    @make_test
    def test_tensor_new_with_shape(x):
        # [FUNCTION DOC] Creates a tensor via x.new from a shape and asserts the same size;
        # returns None.
        y = torch.rand(5, 8)
        z = x.new(y.shape)
        assert z.size() == y.size()  # noqa: S101

    @make_test
    def test_jit_annotate(x):
        # [FUNCTION DOC] Uses the jit annotate helper on x plus 1 and adds 2.
        y = torch.jit.annotate(Any, x + 1)
        return y + 2

    @make_test
    def test_is_contiguous_memory_format(tensor):
        # [FUNCTION DOC] Returns None if scripting, else x plus 1 when the tensor is contiguous.
        if torch.jit.is_scripting():
            return None
        elif tensor.is_contiguous(memory_format=torch.contiguous_format):
            return tensor + 1

    def test_is_contiguous_frame_counts(self):
        # [FUNCTION DOC] Runs a function with contiguity branches over eight tensors of
        # differing strides/layouts, with the counting backend (static or dynamic by default
        # setting), and checks the cumulative frame count after each input against the expected
        # list for the mode.
        data = [
            torch.rand(10),
            torch.rand(2, 3, 32, 32),
            torch.rand(2, 3, 32, 32).contiguous(memory_format=torch.channels_last),
            torch.rand(10)[::2],
            torch.rand(12),
            torch.rand(2, 3, 24, 24).contiguous(memory_format=torch.channels_last),
            torch.rand(50)[::2],
            torch.rand(2, 3, 32, 32)[:, :, 2:-2, 3:-3],
        ]
        # dynamo should recompile for all inputs in static shapes mode
        expected_frame_counts_static = [1, 2, 3, 4, 5, 6, 7, 8]
        # dynamo should recompile for items 0, 1, 2, 6 in dynamic shapes mode
        expected_frame_counts_dynamic = [1, 2, 3, 4, 4, 4, 4, 5]
        expected_frame_counts = ifdynstaticdefault(
            expected_frame_counts_static, expected_frame_counts_dynamic
        )
        dynamic = ifdynstaticdefault(False, True)

        def func(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 if contiguous, x plus 2 if channels_last
            # contiguous, else x plus 3.
            if x.is_contiguous():
                return x + 1
            elif x.is_contiguous(memory_format=torch.channels_last):
                return x + 2
            else:
                return x + 3

        cnt = torch._dynamo.testing.CompileCounter()
        cfunc = torch._dynamo.optimize_assert(cnt, dynamic=dynamic)(func)

        if cnt.frame_count != 0:
            raise AssertionError(f"Expected frame_count 0, got {cnt.frame_count}")
        for i, x in enumerate(data):
            expected = func(x)
            output = cfunc(x)
            self.assertTrue(same(output, expected))
            if cnt.frame_count != expected_frame_counts[i]:
                raise AssertionError(
                    f"Expected frame_count {expected_frame_counts[i]}, got {cnt.frame_count}"
                )

    @make_test
    def test_list_slice_assignment(x):
        # [FUNCTION DOC] Assigns to a slice of a local constant list; returns x plus 1.
        m = [1, 2, 3, 4]
        m[1:] = [6] * (len(m) - 1)
        return x + 1

    @make_test
    def test_distributed_is_available(x):
        # [FUNCTION DOC] Branches on torch.distributed.is_available.
        if torch.distributed.is_available():
            return x + 1
        else:
            return x - 1

    def test_backend_availability_predicates(self):
        # [FUNCTION DOC] Compiles functions that query backend availability predicates (mps,
        # mtia, cuda, xpu, accelerator) and asserts compiled output equals eager, including top
        # level predicates.
        def is_mps_available():
            # [FUNCTION DOC] HELPER: Return whether the mps backend exists and is available
            # and built.
            return (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and torch.backends.mps.is_built()
            )

        def is_mtia_available():
            # [FUNCTION DOC] HELPER: Return whether torch has an mtia attribute and it
            # reports available.
            return hasattr(torch, "mtia") and torch.mtia.is_available()

        def clear_device_cache_like(x):
            # [FUNCTION DOC] HELPER: Return x plus a constant chosen by the first available
            # backend, else x minus 1.
            if is_mps_available():
                return x + 1
            elif torch.mps.is_available():
                return x + 2
            elif is_mtia_available():
                return x + 3
            elif torch.cuda.is_available():
                return x + 4
            elif torch.xpu.is_available():
                return x + 5
            elif torch.accelerator.is_available():
                return x + 6
            return x - 1

        x = torch.randn(3)
        opt_fn = torch.compile(clear_device_cache_like, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), clear_device_cache_like(x))

        def top_level_predicates(x):
            # [FUNCTION DOC] HELPER: Adjust x by +/-7 on mps availability and +/-11 on
            # mtia availability.
            if torch.mps.is_available():
                x = x + 7
            else:
                x = x - 7
            if torch.mtia.is_available():
                x = x + 11
            else:
                x = x - 11
            return x

        opt_fn = torch.compile(top_level_predicates, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), top_level_predicates(x))

    def test_backend_mps_is_built(self):
        # [FUNCTION DOC] Compiles a function branching on the mps is_built flag; asserts equality
        # with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 if mps is built, else x minus 1.
            if torch.backends.mps.is_built():
                return x + 1
            return x - 1

        x = torch.randn(3)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(opt_fn(x), fn(x))

    @unittest.skipIf(
        not torch.distributed.is_available(), "requires distributed package"
    )
    @make_test
    def test_distributed_is_initialized(x):
        # [FUNCTION DOC] Skipped without the distributed package. Branches on is_initialized.
        if torch.distributed.is_initialized():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_torch_distributions_functions(x):
        # [FUNCTION DOC] Builds Normal and Independent distributions from x and returns log_prob
        # of x.
        normal = torch.distributions.Normal(x, torch.tensor(1))
        independent = torch.distributions.Independent(normal, 1)
        return independent.log_prob(x)

    @make_test
    def test_context_wrapping_nested_functions_no_closure(x):
        # [FUNCTION DOC] Defines a nested function wrapped with the no_grad decorator and calls it.
        @torch.no_grad()
        def augment(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Return twice (x plus 1).
            return (x + 1) * 2

        return augment(x)

    @make_test
    def test_match_sequence(a):
        # [FUNCTION DOC] Matches a constant pair (5, 8) against sequence patterns; returns a plus x
        # minus y.
        point = (5, 8)
        match point:
            case (0, 0):
                return a
            case (0, y):
                return a - y
            case (x, 0):
                return a + x
            case (x, y):
                return a + x - y

    @make_test
    def test_match_mapping_and_match_keys(x):
        # [FUNCTION DOC] Uses match with mapping patterns on a dict with key a; returns x
        # times param.
        param = {"a": 0.5}
        match param:
            case {"a": param}:
                return x * param
            case {"b": param}:
                return x / param

    def test_math_radians(self):
        # [FUNCTION DOC] Compiles a function adding math.radians of a constant to a tensor; checks
        # zero frames before the call, matching results and one compiled frame after.
        def func(x, a):
            # [FUNCTION DOC] HELPER: Return x plus math radians of a.
            return x + math.radians(a)

        cnt = torch._dynamo.testing.CompileCounter()
        cfunc = torch._dynamo.optimize_assert(cnt)(func)

        if cnt.frame_count != 0:
            raise AssertionError(f"Expected frame_count 0, got {cnt.frame_count}")
        x = torch.rand(10)
        expected = func(x, 12)
        output = cfunc(x, 12)
        self.assertTrue(same(output, expected))
        if cnt.frame_count != 1:
            raise AssertionError(f"Expected frame_count 1, got {cnt.frame_count}")

    @unittest.skipIf(sys.version_info < (3, 13), "math.fma introduced in python 3.13")
    def test_math_fma(self):
        # [FUNCTION DOC] Skipped before Python 3.13. Checks math.fma: with float constants the call
        # is folded (zero frames); with tensors it compiles one frame and equals x times y plus z.
        def fma_func(a, b, c):
            # [FUNCTION DOC] HELPER: Return the fused multiply add of a, b and c.
            return math.fma(a, b, c)

        # Test with scalar constants (constant folding path)
        cnt = torch._dynamo.testing.CompileCounter()
        cfma_scalars = torch._dynamo.optimize_assert(cnt)(fma_func)

        if cnt.frame_count != 0:
            raise AssertionError(f"Expected frame_count 0, got {cnt.frame_count}")
        expected = fma_func(2.0, 3.0, 4.0)
        output = cfma_scalars(2.0, 3.0, 4.0)
        self.assertEqual(output, expected)
        if cnt.frame_count != 0:
            raise AssertionError(f"Expected frame_count 0, got {cnt.frame_count}")

        # Test with tensors (Inductor path)
        cnt2 = torch._dynamo.testing.CompileCounter()
        cfma_tensors = torch._dynamo.optimize_assert(cnt2)(fma_func)

        if cnt2.frame_count != 0:
            raise AssertionError(f"Expected frame_count 0, got {cnt2.frame_count}")
        x = torch.tensor(2.0)
        y = torch.tensor(3.0)
        z = torch.tensor(4.0)
        expected_tensors = x * y + z
        output_tensors = cfma_tensors(x, y, z)
        torch.testing.assert_close(output_tensors, expected_tensors)
        if cnt2.frame_count != 1:
            raise AssertionError(f"Expected frame_count 1, got {cnt2.frame_count}")

    @make_test
    def test_numpy_meshgrid(x, y):
        # [FUNCTION DOC] Builds a numpy meshgrid from two tensors and returns tensors from
        # the results.
        r1, r2 = np.meshgrid(x.numpy(), y.numpy())
        return torch.from_numpy(r1), torch.from_numpy(r2)

    @make_test
    def test_torch_from_numpy(x):
        # [FUNCTION DOC] Converts to numpy and back; returns a bool tensor for whether size 0
        # equals 1.
        a = x.numpy()
        b = torch.from_numpy(a)
        if b.size(0) == 1:
            return torch.tensor(True)
        else:
            return torch.tensor(False)

    @make_test
    def test_numpy_size(x):
        # [FUNCTION DOC] Returns the size attribute of a numpy view of x.
        a = x.numpy()
        return a.size

    @make_test
    def test_numpy_attributes(x):
        # [FUNCTION DOC] Returns numpy itemsize, strides, shape, ndim, size, and tensors of the
        # transposed, real and imaginary parts of the numpy view.
        a = x.numpy()
        return (
            a.itemsize,
            a.strides,
            a.shape,
            a.ndim,
            a.size,
            torch.from_numpy(a.T),
            torch.from_numpy(a.real),
            torch.from_numpy(a.imag),
        )

    @make_test
    def test_mean_sum_np(x: torch.Tensor):
        # [FUNCTION DOC] Takes the numpy mean along axis 1, sums it, and returns the result as
        # a tensor.
        x_mean = np.mean(x.numpy(), 1)
        x_sum = np.sum(x_mean)
        x_sum_array = np.asarray(x_sum)
        return torch.from_numpy(x_sum_array)

    @make_test
    def test_return_numpy_ndarray(x):
        # [FUNCTION DOC] Returns the transposed numpy array.
        a = x.numpy()
        return a.T

    @make_test
    def test_return_multiple_numpy_ndarray(x):
        # [FUNCTION DOC] Returns the transpose, imaginary and real parts of a numpy view as
        # a tuple.
        a = x.numpy()
        return a.T, a.imag, a.real

    @make_test
    def test_ndarray_method(x):
        # [FUNCTION DOC] Returns a numpy copy.
        a = x.numpy()
        return a.copy()

    @make_test
    def test_ndarray_transpose(x):
        # [FUNCTION DOC] Returns a numpy array transposed over dims 0 and 1.
        a = x.numpy()
        return a.transpose(0, 1)

    @make_test
    def test_ndarray_reshape(x):
        # [FUNCTION DOC] Reshapes a numpy array into a row vector using its size.
        a = x.numpy()
        return a.reshape([1, a.size])

    @make_test
    def test_ndarray_methods_returning_scalar(x):
        # [FUNCTION DOC] Returns the numpy max and all along axis 0 (scalar results).
        a = x.numpy()
        return a.max(axis=0), a.all(axis=0)

    @make_test
    def test_ndarray_builtin_functions(x):
        # [FUNCTION DOC] Returns the sum and difference of a numpy array with itself.
        a = x.numpy()
        return a + a, a - a

    @make_test
    def test_numpy_dtype_argument_to_function(x):
        # [FUNCTION DOC] Calls numpy ones_like on a tensor with a numpy float64 dtype argument.
        return np.ones_like(x, dtype=np.float64)

    @make_test
    def test_numpy_dtype_call_in_function(x):
        # [FUNCTION DOC] Builds a numpy dtype in the function and uses it with full_like.
        dt = np.dtype("float")
        return np.full_like(x, 2.4, dtype=dt)

    @make_test
    def test_numpy_linalg(x):
        # [FUNCTION DOC] Returns the numpy linalg norm along axis 0.
        return np.linalg.norm(x.numpy(), axis=0)

    @make_test
    def test_numpy_fft(x):
        # [FUNCTION DOC] Returns the numpy fftshift of the array.
        return np.fft.fftshift(x.numpy())

    @make_test
    def test_numpy_random():
        # [FUNCTION DOC] Takes no tensor inputs; draws a random numpy array and returns x minus x.
        x = np.random.randn(2, 2)
        return x - x

    @make_test
    def test_partials_torch_op_kwarg(x):
        # [FUNCTION DOC] Applies a partial of torch.mul with a keyword other of ones.
        par_mul = functools.partial(torch.mul, other=torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_torch_op_arg(x):
        # [FUNCTION DOC] Applies a partial of torch.mul with a positional ones tensor.
        par_mul = functools.partial(torch.mul, torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_arg(x):
        # [FUNCTION DOC] Applies a partial of the udf_mul function with a positional ones tensor.
        par_mul = functools.partial(udf_mul, torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_list_add_then_mutate(x):
        # [FUNCTION DOC] Concatenates a list with another, appends, and sums the list.
        my_list = [1, x]
        y = x / 4.0
        my_list = my_list + [x / 2.0, 4]
        my_list.append(y)
        return sum(my_list)

    @make_test
    def test_list_expand_lhs(x):
        # [FUNCTION DOC] Sums a list repeated four times (constant on the left of
        # the multiplication).
        return sum(4 * [x])

    @make_test
    def test_in_not_in(x):
        # [FUNCTION DOC] Asserts membership and non-membership on lists, then sums the list.
        mylist = [1, 2, 3, 4, 5, x]
        myotherlist = [1, 2, 3, 4, 5]
        assert 3 in mylist  # noqa: S101
        assert 6 not in myotherlist  # noqa: S101
        return sum(mylist)

    @make_test
    def test_is(x, y):
        # [FUNCTION DOC] Raises and catches an exception, asserts identity with the original and
        # returns x plus y.
        exc = ValueError("abcd")
        try:
            raise exc
        except Exception as e:
            assert e is exc  # noqa: S101
            return x + y

    @make_test
    def test_is_not(x, y):
        # [FUNCTION DOC] Raises and catches an exception, asserts it is not another
        # exception object.
        exc = ValueError("abcd")
        exc1 = TypeError("abc")
        try:
            raise exc
        except Exception as e:
            assert e is not exc1  # noqa: S101
            return x + y

    @make_test
    def test_are_functorch_transforms_active(x):
        # [FUNCTION DOC] Branches on whether functorch transforms are active.
        if torch._C._are_functorch_transforms_active():
            return x + 1
        else:
            return x - 1

    @make_test
    def test_partials_udf_kwarg(x):
        # [FUNCTION DOC] Applies a partial of udf_mul with a keyword y of ones.
        par_mul = functools.partial(udf_mul, y=torch.ones(10, 10))
        return par_mul(x)

    @make_test
    def test_partials_udf_kwarg_module(x, y):
        # [FUNCTION DOC] Applies a partial of udf_module with a SmallNN module as keyword
        # argument mod.
        par_mod = functools.partial(udf_module, mod=SmallNN())
        return par_mod(x=x, y=y)

    @make_test
    def test_partials_udf_kwarg_method(x, y):
        # [FUNCTION DOC] Applies a partial of udf_module with a bound forward method as mod.
        par_mod = functools.partial(udf_module, mod=SmallNN().forward)
        return par_mod(x=x, y=y)

    @make_test
    def test_partials_lambda(x):
        # [FUNCTION DOC] Applies a partial of a lambda with keyword y equal to 3.
        multiply = lambda x, y: x * y
        triple = functools.partial(multiply, y=3)
        return triple(x)

    @unittest.skipUnless(torch.distributed.is_available(), "requires torch.distributed")
    @make_test
    def test_flat_param_same_storage_size(x, y):
        # [FUNCTION DOC] Skipped without distributed. Uses the FSDP flat_param helper to test if a
        # tensor storage size equals 100 and 123, adjusting x and y by +/- 1.
        import torch.distributed.fsdp._flat_param as flat_param

        if flat_param._same_storage_size(x, 100):
            x = x + 1
        else:
            x = x - 1
        if flat_param._same_storage_size(y, 123):
            y = y + 1
        else:
            y = y - 1
        return x, y

    @parametrize(
        "attr",
        (
            # True
            "__subclasshook__",
            "__lt__",
            "__hash__",
            "__ge__",
            "__le__",
            "__gt__",
            "__dict__",
            "__getattribute__",
            "__setattr__",
            "__doc__",
            "__repr__",
            "__dir__",
            "__init__",
            "__new__",
            "__class__",
            "__eq__",
            "__delattr__",
            "__reduce__",
            "__module__",
            "__format__",
            "__str__",
            "__sizeof__",
            "__ne__",
            "__call__",
            "__reduce_ex__",
            "__init_subclass__",
            "args",
            "keywords",
            "func",
            # False
            "__code__",
            "__kwdefaults__",
            "__defaults__",
            "__name__",
            "__annotations__",
            "__get__",
            "__builtins__",
            "__qualname__",
            "__globals__",
            "__closure__",
        ),
    )
    def test_partials_hasattr(self, attr):
        # [FUNCTION DOC] Parametrized over many attribute names (some present on a partial object,
        # some not). Compiles a function that builds a partial of a lambda, and calls it if hasattr
        # says the attribute exists, else returns zeros; the results are compared to eager.
        def fn(t):
            # [FUNCTION DOC] HELPER: Return the partial applied to t when it has the attribute,
            # else zeros.
            f = lambda x, y: torch.sin(x) + torch.cos(y)
            p = functools.partial(f, y=t)
            if hasattr(p, attr):
                return p(t)
            else:
                return torch.zeros_like(t)

        t = torch.randn(3, 4)
        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend=counter)(fn)
        self.assertEqual(opt_fn(t), fn(t))
        self.assertGreater(counter.frame_count, 0)

    @unittest.expectedFailure
    def test_partials_hasattr_set_attr(self):
        # [FUNCTION DOC] Expected failure: assigning a name attribute on a functools partial inside
        # compiled code is not supported yet, so the compile-and-compare fails.
        def fn(t):
            # [FUNCTION DOC] HELPER: Build a partial of a lambda, set its name attribute, call it
            # if the attribute exists, else return zeros.
            f = lambda x, y: torch.sin(x) + torch.cos(y)
            p = functools.partial(f, y=t)
            p.__name__ = "test"
            if hasattr(p, "__name__"):
                return p(t)
            else:
                return torch.zeros_like(t)

        t = torch.randn(3, 4)
        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fullgraph=True, backend=counter)(fn)
        self.assertEqual(opt_fn(t), fn(t))

    def test_filter(self):
        # [FUNCTION DOC] Compiles with fullgraph a filter over tensors using requires_grad and
        # compares with eager.
        def fn(inputs):
            # [FUNCTION DOC] HELPER: Multiply the first input by every later input whose
            # requires_grad is set.
            out = inputs[0]
            for inp in filter(lambda x: (x.requires_grad), inputs):
                out = out * inp
            return out

        input1 = torch.arange(2, dtype=torch.bfloat16)
        input2 = torch.arange(2, dtype=torch.bfloat16).requires_grad_(True)
        inputs = [input1, input2]

        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(inputs), fn(inputs))

    def test_filter_fallback(self):
        # [FUNCTION DOC] Filter with a predicate on tensor values: compiles and matches eager
        # without fullgraph, then after a Dynamo reset asserts that fullgraph raises Unsupported.
        def fn(inputs):
            # [FUNCTION DOC] HELPER: Multiply the first input by every input whose first element
            # equals 1.
            out = inputs[0]
            for inp in filter(lambda x: x[0] == 1, inputs):
                out = out * inp
            return out

        input1 = torch.ones(2, dtype=torch.bfloat16)
        input2 = torch.arange(2, dtype=torch.bfloat16)
        inputs = [input1, input2]

        opt_fn = torch.compile(backend="eager")(fn)
        self.assertEqual(opt_fn(inputs), fn(inputs))

        torch._dynamo.reset()

        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
            opt_fn(inputs)

    def test_filter_infinite_iterator(self):
        # [FUNCTION DOC] Zips range with filters over an infinite count iterator and compares the
        # lists to eager under fullgraph.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 and two lists from zip of range with
            # filtered count.
            x = x + 1
            return (
                x,
                list(zip(range(3), filter(lambda y: y < 10, itertools.count()))),
                list(zip(range(10, 12), filter(lambda y: y > 10, itertools.count()))),
            )

        inputs = torch.ones(1)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertTupleEqual(opt_fn(inputs), fn(inputs))

    def test_filter_reconstruct(self):
        # [FUNCTION DOC] Returns a filter object from compiled code and checks it is reconstructed
        # as a filter with the same contents as in eager.
        def fn(a):
            # [FUNCTION DOC] HELPER: Return a filter over zipped pairs with sum below 10 and a
            # plus 1.
            return filter(lambda x: x[0] + x[1] < 10, zip([1, 2, 3], [1, 2, 3])), a + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        m = opt_fn(torch.ones(3, 3))[0]
        n = fn(torch.ones(3, 3))[0]
        self.assertIsInstance(m, filter)
        self.assertEqual(list(m), list(n))

    def test_filter_graph_break_reconstruct(self):
        # [FUNCTION DOC] Compiles a function with a data-dependent branch (a graph break) using a
        # recording backend; asserts equality and, in the static or dynamic mode, the printed
        # first graph.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return x plus y if sum of x is positive, else x times y.
            if x.sum() > 0:
                return x + y
            return x * y

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnts)
        a = torch.zeros(3)
        b = torch.ones(3)
        self.assertEqual(opt_fn(a, b), fn(a, b))

        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[3]"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum();  l_x_ = None
        gt: "b8[]" = sum_1 > 0;  sum_1 = None
        return (gt,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77]", s77: "Sym(s77)"):
        l_x_ = L_x_

        sum_1: "f32[]" = l_x_.sum();  l_x_ = None
        gt: "b8[]" = sum_1 > 0;  sum_1 = None
        return (gt,)
""",
            )

    def test_filter_with_graph_break(self):
        # [FUNCTION DOC] Compiles a function that uses a filter with a nonlocal-mutating predicate
        # across an explicit graph break; asserts the result equals eager and three frames compile.
        def f(a):
            # [FUNCTION DOC] HELPER: Increment a, create a filter, take next before and after a
            # graph break.
            a += 1

            def g(x):
                # [FUNCTION DOC] HELPER: Increment the enclosing a and return whether x
                # is positive.
                nonlocal a
                a += 1
                return x > 0

            m = filter(g, [1, 2, 3, 4, 5])
            a += next(m)  # won't graph break
            torch._dynamo.graph_break()
            a += next(m)  # will graph break
            return a

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        self.assertEqual(f(torch.ones(3, 3)), opt_f(torch.ones(3, 3)))
        self.assertEqual(cnts.frame_count, 3)

    @make_test
    def test_getattr(x):
        # [FUNCTION DOC] Looks up an attribute on a function type: sin of x when AttributeError is
        # raised, else cos of x.
        def fn(y):
            # [FUNCTION DOC] HELPER: Return y plus 1.
            return y + 1

        try:
            _exit = type(fn).__exit__
        except AttributeError:
            return x.sin()
        else:
            return x.cos()

    def test_getattr_metaclass(self):
        # [FUNCTION DOC] Checks getattr on a class with a metaclass attribute hook: class attribute
        # lookups for missing names return the name length. Expects t plus 123 plus 12.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose missing-attribute hook returns the length of
            # the name.
            def __getattr__(cls, name):
                # [FUNCTION DOC] HELPER: Return the length of the requested attribute name.
                return len(name)

        class C(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using the Meta metaclass with an attribute attr set to 123.
            attr = 123

        @torch.compile(backend="eager", fullgraph=True)
        def fn(t):
            # [FUNCTION DOC] HELPER: Return t plus C.attr plus C.dynamic_attr (the latter via
            # the metaclass).
            return t + C.attr + C.dynamic_attr

        t = torch.randn(2)
        y = fn(t)
        self.assertEqual(y, t + 123 + 12)

    def test_two_point_iter(self):
        # [FUNCTION DOC] Consumes a map iterator in a loop while also calling next on it inside the
        # loop body; compiled output must equal eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add each map element to x and the following element to y.
            it = map(lambda n: n + 1, range(6))
            for i in it:
                x = x + i
                y = y + next(it)
            return x, y

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.ones(3)
        y = torch.ones(3)
        self.assertEqual(fn(x, y), opt_fn(x, y))

    # Test dict_keys passed along with the corresponding dict object
    def test_dict_key_set1(self):
        # [FUNCTION DOC] Passes a dict and its live keys view into a compiled function; adds a key,
        # then recompiles and checks results again. Compares to eager both times.
        d = {"a": 1, "b": 2}

        def fn(x, d, keys):
            # [FUNCTION DOC] HELPER: Return x plus d["c"] if "c" is in keys, else x plus 1.
            if "c" in keys:
                return x + d["c"]
            else:
                return x + 1

        x = torch.zeros(2, 3)
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(x, d, d.keys()), fn(x, d, d.keys()))

        d.update({"c": 3})
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(x, d, d.keys()), fn(x, d, d.keys()))

    # Test only dict_keys passed into the compiled region
    def test_dict_key_set2(self):
        # [FUNCTION DOC] Same, but only the dict keys view is passed into the compiled region.
        d = {"a": 1, "b": 2}

        def fn(x, keys):
            # [FUNCTION DOC] HELPER: Return x minus 1 if "c" is in keys, else x plus 1.
            if "c" in keys:
                return x - 1
            else:
                return x + 1

        x = torch.zeros(2, 3)
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(x, d.keys()), fn(x, d.keys()))

        d.update({"c": 3})
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(x, d.keys()), fn(x, d.keys()))

    def test_dict_key_set3(self):
        # [FUNCTION DOC] Passes a nested dict and keys view of an inner dict into compiled code;
        # adds an entry, recompiles, and compares with eager.
        a = {
            "domains": {
                "d1": {"attr": 1},
                "d2": {"attr": 2},
            }
        }
        b = a["domains"].keys()

        def fn(x, a, b):
            # [FUNCTION DOC] HELPER: Add the attr of each domain named by b to x in place and
            # return x.
            for e in b:
                x += a["domains"][e]["attr"]
            return x

        x = torch.ones(2, 3)
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(x, a, b), fn(x, a, b))

        a["domains"].update({"d3": {"attr": 3}})
        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        self.assertEqual(opt_fn(x, a, b), fn(x, a, b))

    def test_list_setitem(self):
        # [FUNCTION DOC] Compiles with dynamic shapes a function that assigns into a list by a
        # symbolic index and builds ones of that shape; checks index 0 and 1 equal eager.
        def fn(a: int):
            # [FUNCTION DOC] HELPER: Set element a of a three-item list to 5 and return ones with
            # it as shape.
            some_array = [1, 2, 3]
            some_array[a] = 5
            return torch.ones(some_array)

        opt_fn = torch.compile(fullgraph=True, backend="eager", dynamic=True)(fn)
        self.assertEqual(opt_fn(0), fn(0))
        self.assertEqual(opt_fn(1), fn(1))

    def test_list_setitem_slice(self):
        # [FUNCTION DOC] Same as the list setitem test but assigning a one-element list to a slice
        # starting at the symbolic index.
        def fn(a: int):
            # [FUNCTION DOC] HELPER: Replace a slice of a three-item list and return ones of
            # that shape.
            some_array = [1, 2, 3]
            some_array[a : a + 1] = [5]
            return torch.ones(some_array)

        opt_fn = torch.compile(fullgraph=True, backend="eager", dynamic=True)(fn)
        self.assertEqual(opt_fn(0), fn(0))
        self.assertEqual(opt_fn(1), fn(1))

    def test_pow_int(self):
        # [FUNCTION DOC] Compiles torch.pow with an int exponent under dynamic shapes; compares
        # to eager.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return torch.pow of a and b.
            return torch.pow(a, b)

        x = torch.ones(2, 2)
        opt_fn = torch.compile(fullgraph=True, backend="eager", dynamic=True)(fn)
        self.assertEqual(opt_fn(x, 2), fn(x, 2))

    def test_tensor_size_indexed_by_symint(self):
        # [FUNCTION DOC] Indexes the shape of y by the last size of x and adds it to x; compares
        # to eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Add y.shape at index x.shape[-1] to x.
            index = x.shape[-1]
            return x + y.shape[index]

        x = torch.rand(10, 2)
        y = torch.rand(10, 8, 6)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_partials_as_input_partials_lambda(self):
        # [FUNCTION DOC] Passes two functools partials of a lambda as inputs; expects a
        # single frame.
        def fn(f0, f1, x):
            # [FUNCTION DOC] HELPER: Return the product of calling both partials on x.
            return f0(x) * f1(x)

        multiply = lambda x, y: x * y
        lambda0 = functools.partial(multiply, y=3)
        lambda1 = functools.partial(multiply, y=2)

        cnts = torch._dynamo.testing.CompileCounter()
        torch.compile(fn, backend=cnts, fullgraph=True)(
            lambda0, lambda1, torch.randn(2, 2)
        )
        self.assertEqual(cnts.frame_count, 1)

    def test_partials_as_input_partials_mod(self):
        # [FUNCTION DOC] Passes two partials of a SmallNN module as inputs; expects a single frame
        # and eager-equal output.
        def fn(f0, f1, x):
            # [FUNCTION DOC] HELPER: Return the product of calling both partials on x.
            return f0(x) * f1(x)

        lambda0 = functools.partial(SmallNN(), y=torch.randn(2, 2))
        lambda1 = functools.partial(SmallNN(), y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        dynamo_result = torch.compile(fn, backend=cnts, fullgraph=True)(
            lambda0, lambda1, x
        )
        self.assertEqual(cnts.frame_count, 1)

        eager_result = fn(lambda0, lambda1, x)
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_as_input_UDF(self):
        # [FUNCTION DOC] Passes two partials of udf_mul as inputs; expects one frame and
        # eager-equal output.
        def fn(f0, f1, x):
            # [FUNCTION DOC] HELPER: Return the product of calling both partials on x.
            return f0(x) * f1(x)

        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()
        x = torch.randn(2, 2)
        dynamo_result = torch.compile(fn, backend=cnts, fullgraph=True)(
            lambda0, lambda1, x
        )
        self.assertEqual(cnts.frame_count, 1)

        eager_result = fn(lambda0, lambda1, x)
        self.assertEqual(eager_result, dynamo_result)

    def test_partials_graph_break_reconstruct(self):
        # [FUNCTION DOC] Creates partials of two functions with keyword y, hits a print graph
        # break, then multiplies results; asserts eager equality and the recorded first graph for
        # static or dynamic shapes.
        def fn(udf_mul_0, udf_mul_1, x):
            # [FUNCTION DOC] HELPER: Build two keyword partials, print, and return torch.mul of
            # their results.
            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_mul_1, y=x)

            print("break")
            return torch.mul(lambda0(x), lambda1(x))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch.compile(fn, backend=cnts)(udf_mul, udf_mul, x)

        eager_result = fn(udf_mul, udf_mul, x)
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_
        mul_1: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_2: "f32[2, 2]" = torch.mul(mul, mul_1);  mul = mul_1 = None
        return (mul_2,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[s9, s9]", s9: "Sym(s9)"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        mul: "f32[s9, s9]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_
        mul_1: "f32[s9, s9]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_2: "f32[s9, s9]" = torch.mul(mul, mul_1);  mul = mul_1 = None
        return (mul_2,)
""",
            )

    def test_partials_graph_break_reconstruct_mix(self):
        # [FUNCTION DOC] Same as the previous test with one keyword partial and one positional
        # partial (udf_mul and udf_add); checks result and first graph text.
        def fn(udf_mul_0, udf_add_1, x):
            # [FUNCTION DOC] HELPER: Build keyword and positional partials, print, and multiply
            # their results.
            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_add_1, x)

            print("break")
            return torch.mul(lambda0(x), lambda1(x))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch.compile(fn, backend=cnts)(udf_mul, udf_add, x)

        eager_result = fn(udf_mul, udf_add, x)
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        add: "f32[2, 2]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[2, 2]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[s9, s9]", s9: "Sym(s9)"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        add: "f32[s9, s9]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_

        mul: "f32[s9, s9]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[s9, s9]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )

    def test_partials_graph_break_reconstruct_mix_no_source(self):
        # [FUNCTION DOC] Like the mixed reconstruct test but one partial wraps a lambda created
        # inside the compiled function (no source); checks result and first graph text.
        def fn(udf_mul_0, x):
            # [FUNCTION DOC] HELPER: Create a lambda adding two values inside the function, build
            # partials, print, and multiply their results.
            udf_add_1 = lambda x, y: x + y

            lambda0 = functools.partial(udf_mul_0, y=x)
            lambda1 = functools.partial(udf_add_1, x)

            print("break")
            return torch.mul(lambda0(x), lambda1(x))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch.compile(fn, backend=cnts)(udf_mul, x)

        eager_result = fn(udf_mul, x)
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[2, 2]"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        add: "f32[2, 2]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_

        mul: "f32[2, 2]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[2, 2]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_lambda0_keywords_y_: "f32[s9, s9]", s9: "Sym(s9)"):
        l_lambda0_keywords_y_ = L_lambda0_keywords_y_

        add: "f32[s9, s9]" = l_lambda0_keywords_y_ + l_lambda0_keywords_y_

        mul: "f32[s9, s9]" = l_lambda0_keywords_y_ * l_lambda0_keywords_y_;  l_lambda0_keywords_y_ = None

        mul_1: "f32[s9, s9]" = torch.mul(mul, add);  mul = add = None
        return (mul_1,)
""",
            )

    def test_partials_graph_break_reconstruct_args_and_kwargs(self):
        # [FUNCTION DOC] Partials with both positional args and keywords of udf_mul2, no graph
        # break; checks eager equality and the exact first graph text for static or dynamic shapes.
        def fn(udf_mul_0, x):
            # [FUNCTION DOC] HELPER: Build partials with positional and keyword arguments, call
            # them and multiply the results.
            lambda0 = functools.partial(udf_mul_0, x, 4, z=x)
            lambda1 = functools.partial(udf_mul_0, 4, z=x)

            return torch.mul(lambda0(), lambda1(5))

        backend = EagerAndRecordGraphs()
        cnts = CompileCounterWithBackend(backend)
        x = torch.randn(2, 2)
        dynamo_result = torch.compile(fn, backend=cnts)(udf_mul2, x)

        eager_result = fn(udf_mul2, x)
        self.assertEqual(eager_result, dynamo_result)
        if torch._dynamo.config.assume_static_by_default:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[2, 2]"):
        l_x_ = L_x_

        mul: "f32[2, 2]" = l_x_ * 4
        mul_1: "f32[2, 2]" = 20 * l_x_
        mul_2: "f32[2, 2]" = mul * l_x_;  mul = l_x_ = None

        mul_3: "f32[2, 2]" = torch.mul(mul_2, mul_1);  mul_2 = mul_1 = None
        return (mul_3,)
""",
            )
        else:
            self.assertExpectedInline(
                normalize_gm(backend.graphs[0].print_readable(print_output=False)),
                """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[s77, s77]", s77: "Sym(s77)"):
        l_x_ = L_x_

        mul: "f32[s77, s77]" = l_x_ * 4
        mul_1: "f32[s77, s77]" = 20 * l_x_
        mul_2: "f32[s77, s77]" = mul * l_x_;  mul = l_x_ = None

        mul_3: "f32[s77, s77]" = torch.mul(mul_2, mul_1);  mul_2 = mul_1 = None
        return (mul_3,)
""",
            )

    def test_partials_recompilation(self):
        # [FUNCTION DOC] Tracks recompilation counts when partials are passed in: swapping equal
        # partials does not recompile; changed tensor size, new function, different keyword or
        # positional keys, or a non-partial input each trigger a recompile (counts 1 to 4).
        def fn(f0, f1, x):
            # [FUNCTION DOC] HELPER: Return the product of calling both functions on x.
            return f0(x) * f1(x)

        lambda0 = functools.partial(udf_mul, y=torch.randn(2, 2))
        lambda1 = functools.partial(udf_mul, y=torch.randn(2, 2))

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.randn(2, 2)
        fn = torch.compile(fn, backend=cnts, fullgraph=True)
        fn(lambda0, lambda1, x)
        self.assertEqual(cnts.frame_count, 1)

        fn(lambda1, lambda0, x)
        self.assertEqual(
            cnts.frame_count, 1
        )  # No recompile! Tensor and udf_mul guarded

        lambda2 = functools.partial(udf_mul, y=torch.randn(3, 3))
        x = torch.randn(3, 3)
        fn(lambda2, lambda2, x)
        self.assertEqual(cnts.frame_count, 2)  # Recompile! Tensor size changed

        multiply = lambda x, y: x * y
        lambda3 = functools.partial(multiply, y=torch.randn(3, 3))
        x = torch.randn(3, 3)
        fn(lambda3, lambda3, x)

        self.assertEqual(cnts.frame_count, 3)  # Recompile! func id changed

        def fn2(f0, f1, args):
            # [FUNCTION DOC] HELPER: Return the product of calling both functions with the
            # unpacked args.
            return f0(*args) * f1(*args)

        cnts = torch._dynamo.testing.CompileCounter()

        x = torch.randn(2, 2)
        fn2 = torch.compile(fn2, backend=cnts, fullgraph=True)
        fn2(lambda0, lambda1, [x])
        self.assertEqual(cnts.frame_count, 1)  # start over

        lambda4 = functools.partial(multiply, y=3, x=torch.randn(3, 3))
        fn2(lambda4, lambda4, [])

        self.assertEqual(cnts.frame_count, 2)  # Recompile! Different kwarg keys

        lambda5 = functools.partial(multiply, 1)
        x = torch.randn(3, 3)
        fn2(lambda5, lambda5, [x])

        self.assertEqual(cnts.frame_count, 3)  # Recompile! Different arg keys

        lambda6 = lambda x: x + x
        fn2(lambda6, lambda6, [x])
        self.assertEqual(
            cnts.frame_count, 4
        )  # Recompile! input is no longer a functools partial

    def test_manual_seed(self):
        # [FUNCTION DOC] Compiled function seeds and draws random ints; two successive pairs of
        # calls must yield equal results.
        @torch.compile(backend="eager")
        def foo():
            # [FUNCTION DOC] HELPER: Seed the generator with 3 and return five random ints.
            torch.manual_seed(3)
            return torch.randint(0, 5, (5,))

        self.assertEqual(foo(), foo())
        self.assertEqual(foo(), foo())

    @unittest.skipIf(not torch.cuda.is_available(), "requires cuda")
    def test_cuda_manual_seed(self):
        # [FUNCTION DOC] Skipped without CUDA. For each cuda manual seed function (four variants)
        # under inductor fallback_random, compiles a function seeding and drawing cuda randoms and
        # asserts repeated calls produce equal results. Resets Dynamo for every variant.
        import torch._inductor.config as inductor_config

        seed_fns = (
            torch.cuda.manual_seed,
            torch.cuda.manual_seed_all,
            torch.cuda.random.manual_seed,
            torch.cuda.random.manual_seed_all,
        )

        with inductor_config.patch("fallback_random", True):
            for seed_fn in seed_fns:
                with self.subTest(seed_fn=f"{seed_fn.__module__}.{seed_fn.__name__}"):
                    torch._dynamo.reset()

                    @torch.compile  # noqa: UNSPECIFIED_BACKEND
                    def foo():
                        # [FUNCTION DOC] HELPER: Call the current seed function with 3 and return a
                        # cuda random tensor.
                        seed_fn(3)
                        return torch.rand(4, device="cuda")

                    self.assertEqual(foo(), foo())
                    self.assertEqual(foo(), foo())

    def test_partial_across_graph_break_uninvoked(self):
        # [FUNCTION DOC] Compiles with dynamic shapes a function that creates partials, calls an
        # inner function with a graph break, then rebinds the partial; just checks it runs.
        from functools import partial

        def bar(x, **kwargs):
            # [FUNCTION DOC] HELPER: Return x plus x (extra keyword arguments are ignored).
            return x + x

        @torch.compile(backend="eager", dynamic=True)
        def foo(x, i):
            # [FUNCTION DOC] HELPER: Compiled function defining an inner function and rebinding op
            # between calls.
            def inner():
                # [FUNCTION DOC] HELPER: Print a message (graph break) and call the current op
                # on x.
                print("this is a graph_break")
                return op(x)

            op = partial(bar, dim=10)
            x = inner()
            op = partial(bar, other=10)
            return inner() + x

        foo(torch.rand(1), 10)

    def test_no_recompile_inner_function(self):
        # [FUNCTION DOC] Calls a compiled function with an inner closure and a graph break three
        # times; expects only one frame compiled (no recompile).
        def forward(inp):
            # [FUNCTION DOC] HELPER: Define an inner closure, print (graph break), call it with a
            # random tensor.
            def g(y):
                # [FUNCTION DOC] HELPER: Return the captured inp plus y.
                return inp + y

            print("graph break")
            return g(torch.rand([1]))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(forward, backend=cnts)

        input = torch.rand([2])
        _ = opt_fn(input)
        _ = opt_fn(input)
        _ = opt_fn(input)
        # Should not have recompiled
        self.assertEqual(cnts.frame_count, 1)

    def test_no_recompile_inner_lambda(self):
        # [FUNCTION DOC] Same as the inner function test but with a lambda closure; expects
        # one frame.
        def forward(inp):
            # [FUNCTION DOC] HELPER: Create a lambda closure, print (graph break) and call it.
            g = lambda y: inp + y
            print("graph break")
            return g(torch.rand([1]))

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(forward, backend=cnts)

        input = torch.rand([2])
        _ = opt_fn(input)
        _ = opt_fn(input)
        _ = opt_fn(input)
        # Should not have recompiled
        self.assertEqual(cnts.frame_count, 1)

    def test_complex_closure(self):
        # [FUNCTION DOC] Compiles a function returning a doubly nested closure and calls the
        # result; asserts it equals the sum of the two inputs.
        @torch.compile(backend="eager")
        def forward(y):
            # [FUNCTION DOC] HELPER: Define nested closures and return the result of calling a.
            def a():
                # [FUNCTION DOC] HELPER: Return the inner closure x.
                def x(z):
                    # [FUNCTION DOC] HELPER: Return the captured y plus z.
                    return y + z

                return x

            return a()

        input1 = torch.rand([2])
        input2 = torch.rand([2])
        res = forward(input1)(input2)
        self.assertTrue(same(res, input1 + input2))

    def test_non_inlined_closure(self):
        # [FUNCTION DOC] Checks a closure invoked through a graph-breaking inner function sees the
        # rebinding of a captured lambda: the result is input1 doubled.
        @torch.compile(backend="eager")
        def program(x, y):
            # [FUNCTION DOC] HELPER: Rebind a captured lambda between two calls of an
            # inner function.
            one = lambda x, y: x + y

            def inner():
                # [FUNCTION DOC] HELPER: Force a graph break, then call the currently bound lambda.
                # Force no inlining
                torch._dynamo.graph_break()
                return one(x, y)

            res = inner()
            one = lambda x, y: x - y
            res += inner()
            return res

        input1 = torch.randn(1)
        input2 = torch.randn(1)

        self.assertTrue(same(program(input1, input2), input1 + input1))

    @parametrize("int_or_float", ("int", "float"))
    def test_np_constant_collections_as_input(self, int_or_float):
        # [FUNCTION DOC] Parametrized over int and float. Calls a compiled function with numpy
        # dtype names, dtype types, dtype objects and iinfo/finfo objects as arguments; each result
        # must equal the eager result.
        info_func = getattr(np, f"{int_or_float[0]}info")
        dt_string_arg = f"{int_or_float}16"
        np_dt_attr = getattr(np, dt_string_arg)

        dt_args = [dt_string_arg, np_dt_attr]
        arg_variants_iter = itertools.chain(
            dt_args, map(np.dtype, dt_args), map(info_func, dt_args)
        )

        def func(a, b, info_or_dt):
            # [FUNCTION DOC] HELPER: Return a plus the max value of the numpy info for the
            # given dtype.
            return a + info_func(info_or_dt).max

        opt_fn = torch.compile(func, backend="eager")

        a = torch.randn(2)
        b = torch.randn(2)
        eager_result = func(a, b, dt_args[0])

        for arg in arg_variants_iter:
            opt_result = opt_fn(a, b, arg)
            self.assertTrue(same(opt_result, eager_result))

    @parametrize(
        "typ, info_func",
        [
            (int, np.iinfo),
            (float, np.finfo),
        ],
        name_fn=lambda t, _: t.__name__,
    )
    def test_np_constant_collections_guards(self, typ, info_func):
        # [FUNCTION DOC] Parametrized over int/float with iinfo/finfo. Checks guards on numpy
        # dtype and info arguments: equal arguments cause one compile; a different dtype causes a
        # second compile.
        def func_info(a, info):
            # [FUNCTION DOC] HELPER: Return a plus the max of an info object.
            return a + info.max

        def func_dtype(a, dt):
            # [FUNCTION DOC] HELPER: Return a plus the max of the info for a dtype.
            return a + info_func(dt).max

        dt_args = [
            np.dtype(typ),
            np.ones((1,), dtype=typ).dtype,
            np.dtype(np.dtype(typ).name),
            np.dtype(typ.__name__),
        ]
        cnts_1 = torch._dynamo.testing.CompileCounter()
        opt_fn_dtype = torch.compile(func_dtype, backend=cnts_1)
        a = torch.zeros(3, dtype=typ)
        for arg in dt_args:
            opt_fn_dtype(a, arg)
        # each should produce an identical arg
        self.assertEqual(cnts_1.frame_count, 1)

        cnts_2 = torch._dynamo.testing.CompileCounter()
        opt_fn_info = torch.compile(func_info, backend=cnts_2)
        info_args = [info_func(dt) for dt in dt_args]
        for arg in info_args:
            opt_fn_info(a, arg)

        # each should produce an identical arg
        self.assertEqual(cnts_2.frame_count, 1)

        if typ is float:
            dt_extra = np.dtype(np.float16)
        else:
            dt_extra = np.dtype(np.int16)
        info_extra = info_func(dt_extra)

        eager_result_dtype = func_dtype(a, dt_extra)
        compile_result_dtype = opt_fn_dtype(a, dt_extra)
        self.assertEqual(cnts_1.frame_count, 2)
        self.assertEqual(eager_result_dtype, compile_result_dtype)

        eager_result_info = func_info(a, info_extra)
        compile_result_info = opt_fn_info(a, info_extra)
        self.assertEqual(cnts_2.frame_count, 2)
        self.assertEqual(eager_result_info, compile_result_info)

    def test_compare_constant_and_tensor(self):
        # [FUNCTION DOC] For each comparison operator (lt, le, gt, ge, ne, eq, is_, is_not)
        # compiles a function comparing the constant -10 with a tensor and compares to eager.
        for op in [
            operator.lt,
            operator.le,
            operator.gt,
            operator.ge,
            operator.ne,
            operator.eq,
            operator.is_,
            operator.is_not,
        ]:
            with self.subTest(op=op):

                def fn(x):
                    # [FUNCTION DOC] HELPER: Apply the loop operator to the constant -10 and x.
                    return op(-10, x)

                opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)

                x = torch.randn(10)
                self.assertEqual(opt_fn(x), fn(x))

    def test_pos(self):
        # [FUNCTION DOC] Compiles with fullgraph and dynamic shapes the unary plus operator
        # function times unary plus on inputs; checks several tensor and scalar input combinations
        # vs eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return unary plus of x times unary plus of y.
            return operator.pos(x) * +y

        opt_fn = torch.compile(fullgraph=True, dynamic=True, backend="eager")(fn)

        def test(x, y):
            # [FUNCTION DOC] HELPER: Assert the compiled and eager results are equal for
            # the inputs.
            self.assertEqual(opt_fn(x, y), fn(x, y))

        test(torch.ones(4), 1)
        test(1, torch.ones(4))
        test(-1, -1)
        test(-1.1, 1.1)
        test(True, False)
        test(torch.ones(4, dtype=torch.float32), 1.1)

    def test_index(self):
        # [FUNCTION DOC] Compiles a function using operator.index on ints and bools, for dynamic
        # and static modes (Dynamo reset in each pass), comparing to eager.
        def fn(x, t):
            # [FUNCTION DOC] HELPER: Convert x with operator.index and multiply t by it
            # (result discarded).
            v = operator.index(x)
            torch.mul(t, v)

        def test(a, b):
            # [FUNCTION DOC] HELPER: Assert the compiled and eager results for a and b are equal.
            self.assertEqual(opt_fn(a, b), fn(a, b))

        for dynamic in [True, False]:
            torch._dynamo.reset()
            opt_fn = torch.compile(fn, dynamic=dynamic, backend="eager")
            t = torch.ones(1)
            test(10, t)
            test(-100, t)
            test(10, t)
            test(False, t)
            test(True, t)

    def test_truth(self):
        # [FUNCTION DOC] Compiles a function combining operator.truth and a bool conversion for a
        # range of ints, floats, bools and tensors, comparing with eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return the truth value of x and the bool conversion of y.
            return operator.truth(x) and bool(y)

        opt_fn = torch.compile(dynamic=False, backend="eager")(fn)

        def test(x, y):
            # [FUNCTION DOC] HELPER: Assert the compiled and eager results are equal.
            self.assertEqual(opt_fn(x, y), fn(x, y))

        test(1, 100)
        test(-1.1, True)
        test(-1.1, 1.1)
        test(True, False)
        test(torch.ones(1), 1)
        test(torch.zeros(1), 1)
        test(torch.ones(1), torch.ones(1))

    def test_unary_fold_op(self):
        # [FUNCTION DOC] For several unary operators (abs, neg, pos, truth) compiles with fullgraph
        # a function mapping the operator over a constant range and compares to eager.
        for op in (operator.abs, abs, operator.neg, operator.pos, operator.truth):
            with self.subTest(op=op):

                def fn():
                    # [FUNCTION DOC] HELPER: Return the list of the loop operator applied to a
                    # range from minus 10 up to 10.
                    a = range(-10, 10)
                    return list(map(op, a))

                opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
                self.assertEqual(opt_fn(), fn())

    def test_unary_fold_op_seq(self):
        # [FUNCTION DOC] Compiles a function that maps operator.length_hint over tuples built from
        # ranges (constant folding of a sequence unary op); compares to eager.
        for op in (operator.length_hint,):
            with self.subTest(op=op):

                def fn():
                    # [FUNCTION DOC] HELPER: Return a tuple of length hints of tuples built
                    # from ranges.
                    a = [tuple(range(-10, i)) for i in range(10)]
                    return tuple(map(op, a))

                opt_fn = torch.compile(fn, fullgraph=True, backend="eager")
                self.assertEqual(opt_fn(), fn())

    def test_operator_concat(self):
        # [FUNCTION DOC] For list and tuple types, compiles operator.concat and compares to eager.
        for seq_type in (list, tuple):
            with self.subTest(seq_type=seq_type):

                def fn(a, b):
                    # [FUNCTION DOC] HELPER: Return the concatenation of a and b via the
                    # operator module.
                    return operator.concat(a, b)

                opt_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
                a = seq_type([1, 2, 3])
                b = seq_type([4, 5, 6])
                self.assertEqual(opt_fn(a, b), fn(a, b))

    def test_operator_iconcat(self):
        # [FUNCTION DOC] Checks operator.iconcat on two lists yields the joined list under compile.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return the in-place concatenation of a and b via the
            # operator module.
            return operator.iconcat(a, b)

        opt_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        self.assertEqual(opt_fn([1, 2, 3], [4, 5, 6]), [1, 2, 3, 4, 5, 6])

    def test_attrgetter(self):
        # [FUNCTION DOC] For several attribute-name tuples compiles an attrgetter applied to two
        # tensors and compares to eager.
        for attrs in (
            ("shape",),
            ("data.shape",),
            ("device", "shape"),
            ("device", "shape", "data.shape"),
        ):
            with self.subTest(attrs=attrs):

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Build an attrgetter from the loop variable attrs and
                    # apply it to x and y.
                    getter = operator.attrgetter(*attrs)
                    return getter(x), getter(y)

                opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)

                x = torch.randn(3, 4)
                y = torch.randn(3, 4)
                self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_itemgetter(self):
        # [FUNCTION DOC] For several item specs (indices and slices) compiles an itemgetter applied
        # to two tensors and compares to eager.
        for items in (
            (0,),
            (slice(1, 3),),
            (0, 1),
            (slice(1, 3), 0, 1),
        ):
            with self.subTest(items=items):

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Build an itemgetter from the loop variable items and
                    # apply to x and y.
                    getter = operator.itemgetter(*items)
                    return getter(x), getter(y)

                opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)

                x = torch.randn(3, 4)
                y = torch.randn(3, 4)
                self.assertEqual(opt_fn(x, y), fn(x, y))

    def test_methodcaller(self):
        # [FUNCTION DOC] For several method names with args and kwargs compiles a methodcaller
        # applied to two tensors and compares to eager.
        for name, args, kwargs in (
            ("size", (), {}),
            ("size", (0,), {}),
            ("add", (torch.randn(3, 4),), {}),
            ("add", (torch.randn(3, 4),), {"alpha": 2.0}),
        ):
            with self.subTest(name=name, args=args, kwargs=kwargs):

                def fn(x, y):
                    # [FUNCTION DOC] HELPER: Build a methodcaller from the loop values and apply it
                    # to x and y.
                    caller = operator.methodcaller(name, *args, **kwargs)
                    return caller(x), caller(y)

                opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)

                x = torch.randn(3, 4)
                y = torch.randn(3, 4)
                self.assertEqual(opt_fn(x, y), fn(x, y))

    def gen_random_range_args(self):
        # [FUNCTION DOC] HELPER: Return a random list of one to three ints in [-10, 10] suitable as
        # range arguments, replacing a zero step by one.
        args_count = random.randint(1, 3)
        args = [random.randint(-10, 10) for _ in range(args_count)]
        if args_count == 3 and args[2] == 0:
            args[2] = 1
        return args

    def test_list_iterator_graph_break(self):
        # [FUNCTION DOC] Compiles a function whose list iterator is advanced across an explicit
        # graph break; asserts the sum 1 plus 3 plus 5 added to x.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Take an item from a list iterator, graph break, take two more.
            it = [1, 3, 5].__iter__()
            y = x + next(it)
            torch._dynamo.graph_break()
            return y + next(it) + next(it)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x + 1 + 3 + 5)

    def test_range_iterator_graph_break(self):
        # [FUNCTION DOC] Same as the list iterator test but with a range iterator.
        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Take an item from a range iterator, graph break, take
            # two more.
            it = range(1, 7, 2).__iter__()
            y = x + next(it)
            torch._dynamo.graph_break()
            return y + next(it) + next(it)

        x = torch.tensor([1.0])
        y = fn(x)
        self.assertEqual(y, x + 1 + 3 + 5)

    def test_range_iterator_graph_break_2(self):
        # [FUNCTION DOC] A range iterator passed to a function excluded from compilation which
        # consumes two items; the compiled caller then expects the next item 7. Asserts the total.
        @torch.compiler.disable
        def g(y, it):
            # [FUNCTION DOC] HELPER: Compiler-disabled function adding the next two items of the
            # iterator to y.
            return y + next(it) + next(it)

        @torch.compile(backend="eager")
        def fn(x):
            # [FUNCTION DOC] HELPER: Advance a range iterator, call the disabled function, check
            # the next is 7.
            it = range(1, 10, 2).__iter__()
            y = x + next(it)
            z = g(y, it)
            k = next(it)
            assert k == 7  # noqa: S101
            return z + k

        x = torch.tensor([1.0])
        z = fn(x)
        self.assertEqual(z, x + 1 + 3 + 5 + 7)

    @make_test
    def test_range_iterator(a, b):
        # [FUNCTION DOC] Checks the range iterator object has the range_iterator type.
        it = range(5).__iter__()
        if isinstance(it, range_iterator):
            return a + b
        return a - b

    @make_test
    def test_range_iterator_2(a, b):
        # [FUNCTION DOC] Checks the iterator of a range has the range_iterator type (comment notes
        # it should pass once there is a single call_iter path).
        # should pass once we stop having three different paths on call_iter
        it = iter(range(5))
        if isinstance(it, range_iterator):
            return a + b
        return a - b

    def test_range_length(self):
        # [FUNCTION DOC] Compares the length of Python range objects with
        # RangeVariable.range_length for fixed edge cases (empty, step, negative step) and 100
        # random argument sets.
        def test(*args, expected=None):
            # [FUNCTION DOC] HELPER: Build a range and RangeVariable from args; assert lengths
            # agree, and equal expected when given.
            r = range(*args)
            range_variable = RangeVariable([ConstantVariable.create(v) for v in args])

            self.assertEqual(len(r), range_variable.range_length())

            if expected is not None:
                self.assertEqual(len(r), expected)

        test(1, 1, 1, expected=0)
        test(1, 0, expected=0)
        test(-10, expected=0)

        test(4, expected=4)
        test(10, expected=10)

        # step >1
        test(1, 10, 2, expected=5)

        # negative step
        test(10, 1, -1, expected=9)
        test(10, 1, -3)

        # Fuzz testing
        for _ in range(100):
            args = self.gen_random_range_args()
            print("testing :", args)
            test(*args)

    def test_indexed_range(self):
        # [FUNCTION DOC] Compares range indexing with the RangeVariable apply-index result on fixed
        # cases and random in-range indexes.
        def test(range, index, expected=None):
            # [FUNCTION DOC] HELPER: Build a RangeVariable from a range; assert its constant at the
            # index matches the real range, and expected if given.
            range_variable = RangeVariable(
                [
                    ConstantVariable.create(v)
                    for v in [range.start, range.stop, range.step]
                ]
            )

            self.assertEqual(
                range[index],
                range_variable.apply_index(None, index).as_python_constant(),
            )

            if expected is not None:
                self.assertEqual(range[index], expected)

        test(range(10), 1, expected=1)
        test(range(10, 20, 2), 1, expected=12)

        # Fuzz testing
        for _ in range(100):
            range_args = self.gen_random_range_args()
            r = range(*range_args)

            if len(r) == 0:
                continue

            index = random.randint(0, len(r) - 1)

            print("testing:", r, index)
            test(r, index)

    def test_sliced_range(self):
        # [FUNCTION DOC] Compares slicing of ranges with the RangeVariable apply-slice result for
        # fixed slices and 100 random slices.
        def test(range, slice, expected=None):
            # [FUNCTION DOC] HELPER: Build a RangeVariable; assert the sliced constant matches
            # range slicing and the expected range if given.
            range_variable = RangeVariable(
                [
                    ConstantVariable.create(v)
                    for v in [range.start, range.stop, range.step]
                ]
            )

            self.assertEqual(
                range[slice],
                range_variable.apply_slice(slice).as_python_constant(),
            )

            if expected is not None:
                self.assertEqual(
                    range[slice],
                    expected,
                )

        test(range(10), slice(1, 10, 2), expected=range(1, 10, 2))
        test(range(10), slice(None, 10, None), expected=range(10))
        test(range(10), slice(-1, 7, None), expected=range(9, 7))
        test(range(10), slice(-1, 7, 2), expected=range(9, 7, 2))
        test(range(1, 10, 2), slice(3, 7, 2), expected=range(7, 11, 4))
        test(range(1, 10, 2), slice(-3, 7, 2), expected=range(5, 11, 4))
        test(range(-1, -5, -3), slice(5, None, -3), expected=range(-4, 2, 9))

        def rand_slice():
            # [FUNCTION DOC] HELPER: Return a slice with one to three randomly chosen parts; the
            # step is nonzero.
            def flip_coin():
                # [FUNCTION DOC] HELPER: Return true about one time in ten (random).
                # 1 out of 10
                return random.randint(1, 10) == 5

            def r_item(allow_zero=True):
                # [FUNCTION DOC] HELPER: Return a random int in [-10, 10] (nonzero if disallowed),
                # occasionally None.
                i = random.randint(-10, 10)
                if not allow_zero and i == 0:
                    i = 1
                if flip_coin():
                    i = None
                return i

            arg_count = random.randint(1, 3)

            if arg_count == 1:
                return slice(r_item())
            elif arg_count == 2:
                return slice(r_item(), r_item())
            else:
                return slice(r_item(), r_item(), r_item(False))

        # Fuzz testing
        for _ in range(100):
            range_args = self.gen_random_range_args()
            r = range(*range_args)
            # generate random slice
            s = rand_slice()

            print("testing:", r, s)
            test(r, s)

    def test_range_with_slice_index(self):
        # [FUNCTION DOC] Compiles a function iterating a sliced range and multiplying an
        # accumulator; compares to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Loop over a sliced range updating acc and return x times acc.
            acc = 1
            for k in range(2)[1::2]:
                acc *= acc * k
            return x * acc

        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        x = torch.ones(1)
        self.assertEqual(opt_fn(x), fn(x))

    def test_range_with_index(self):
        # [FUNCTION DOC] Compiles a function indexing a range; compares to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Multiply acc by an indexed range element and return x
            # times acc.
            acc = 1
            acc *= acc * range(10, 20, 2)[2]
            return x * acc

        opt_fn = torch.compile(fullgraph=True, backend="eager")(fn)
        x = torch.ones(1)
        self.assertEqual(opt_fn(x), fn(x))

    def test_rand_inlined(self):
        # [FUNCTION DOC] Compiles with dynamic shapes a function using python random ints for list
        # indices and sizes and creates an empty tensor; just checks it runs.
        @torch.compile(backend="eager", dynamic=True)
        def fn():
            # [FUNCTION DOC] HELPER: Use random ints to alter a size list and create an
            # empty tensor.
            idx_size = [10]
            idx_size[random.randint(0, 0)] = random.randint(1, 8)
            t = tuple(idx_size)
            src_size = [random.randint(1, 5) + s for s in idx_size]  # noqa: F841
            idx = torch.empty(t)  # noqa: F841

        fn()

    def test_rand_tensor_partial(self):
        # [FUNCTION DOC] Compiles a function that builds tensors via nested partials of torch.rand
        # with namedtuple shapes; just checks it runs.
        from collections import namedtuple
        from functools import partial

        SdpaShape = namedtuple(
            "Sdpa_Shape", ["batch", "num_heads", "seq_len", "head_dim"]
        )

        @torch.compile(backend="eager")
        def func():
            # [FUNCTION DOC] HELPER: Build q and kv random tensors via partials and add them.
            make_tensor = partial(torch.rand, dtype=torch.float16, requires_grad=True)

            bsz, num_heads, seq_len_q, seq_len_kv, head_dim = (16, 16, 128, 128, 16)
            make_q_tensor = partial(
                make_tensor, SdpaShape(bsz, num_heads, seq_len_q, head_dim)
            )
            make_kv_tensor = partial(
                make_tensor, SdpaShape(bsz, num_heads, seq_len_kv, head_dim)
            )
            t1 = make_q_tensor()
            t2 = make_kv_tensor()
            t3 = t1 + t2  # noqa: F841

        func()

    def test_to(self):
        # [FUNCTION DOC] Compiles a function moving a ones tensor to the meta device; just checks
        # it runs.
        @torch.compile(backend="eager")
        def fn():
            # [FUNCTION DOC] HELPER: Create ones and move to meta device.
            t = torch.ones(2)
            y = t.to("meta")  # noqa: F841

        fn()

    def test_elipsis(self):
        # [FUNCTION DOC] Compiles a function assigning into numpy arrays via index tuples with
        # Ellipsis; asserts the results for several index forms.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(a, ind, val):
            # [FUNCTION DOC] HELPER: Assign val into a at ind and return a.
            a[ind] = val
            return a

        arr = np.zeros(4)
        self.assertEqual(fn(arr, np.s_[...], np.ones(4)), np.ones(4))

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[0, ...], np.zeros(2)), np.array([[0, 0], [2, 2]])
        )

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[1, ...], np.zeros(2)), np.array([[1, 1], [0, 0]])
        )

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[..., 0], np.array([3, 3])), np.array([[3, 1], [3, 2]])
        )

        arr = np.array([[1, 1], [2, 2]])
        self.assertEqual(
            fn(arr, np.s_[..., 1], np.array([3, 3])), np.array([[1, 3], [2, 3]])
        )

    def test_round(self):
        # [FUNCTION DOC] Compiles a function adding the rounded constant to t and compares
        # to eager.
        def fn(t):
            # [FUNCTION DOC] HELPER: Return t plus round of a float constant to 7 digits.
            return t + round(1.00002000011, 7)

        t = torch.randn(2)
        e = fn(t)
        g = torch.compile(fn, backend="eager", fullgraph=True)(t)
        self.assertEqual(e, g)

    @unittest.skipIf(sys.platform == "darwin", "No mkldnn on MacOS")
    def test_quantize_per_tensor(self):
        # [FUNCTION DOC] Skipped on macOS. Quantizes a tensor per-tensor to quint8 and compares the
        # compiled and eager results.
        def fn(t, scale, zero_point):
            # [FUNCTION DOC] HELPER: Return torch.quantize_per_tensor of t with the scale and zero
            # point as quint8.
            return torch.quantize_per_tensor(t, scale, zero_point, torch.quint8)

        scale = torch.tensor(2.0)
        zero_point = torch.tensor(10.0)
        t = torch.rand((2, 2)) * scale + zero_point

        result = fn(t, scale, zero_point)
        compiled_fn = torch.compile(fn, fullgraph=True)  # noqa: UNSPECIFIED_BACKEND
        compiled_result = compiled_fn(t, scale, zero_point)
        self.assertEqual(compiled_result, result)

    def test_map_return(self):
        # [FUNCTION DOC] Compiles a function returning a map object and asserts the result is
        # a map.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return a map adding 1 over [a, b].
            return map(lambda x: x + 1, [a, b])

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        m = opt_fn(torch.randn(3, 3), torch.randn(3, 3))
        self.assertIsInstance(m, map)

    @make_test
    def test_map_max(a, b):
        # [FUNCTION DOC] Takes the max of a map of sums over two tensors.
        return max(map(lambda x: x.sum(), [a, b]))

    @make_test
    def test_map_max_const(a):
        # [FUNCTION DOC] Takes the max of a map over constants and returns it with a plus 1.
        return max(map(lambda x: x, [1, 2, 3])), a + 1

    @make_test
    def test_map_list(a, b):
        # [FUNCTION DOC] Converts a map over [a, b] to a list.
        return list(map(lambda x: x + 1, [a, b]))

    @make_test
    def test_map_tuple(a, b):
        # [FUNCTION DOC] Converts a map over [a, b] to a tuple.
        return tuple(map(lambda x: x + 1, [a, b]))

    @make_test
    def test_map_iter(a, b):
        # [FUNCTION DOC] Wraps a map in iter and returns its next item.
        it = iter(map(lambda x: x + 1, [a, b]))
        return next(it)

    @make_test
    def test_map_zip_dict(a):
        # [FUNCTION DOC] Builds a dict by zipping mapped keys with lists of maps; returns the first
        # item of the map under key 3 and a plus 1.
        d = dict(
            zip(
                map(lambda x: x + 1, [0, 1, 2]),
                [map(lambda x: x - 1, [y]) for y in [3, 4, 5]],
            )
        )
        return list(d[3])[0], a + 1  # noqa: RUF015

    @make_test
    def test_map_dict_fromkeys(a):
        # [FUNCTION DOC] Calls dict.fromkeys on a map; returns it with a plus 1.
        return dict.fromkeys(map(lambda x: x + 1, [0, 1])), a + 1

    @make_test
    def test_map_set(a):
        # [FUNCTION DOC] Builds a set from a map; returns it with a plus 1.
        return set(map(lambda x: x + 1, [0, 1])), a + 1

    # test_map_sum defined earlier

    @make_test
    def test_map_reduce(a, b):
        # [FUNCTION DOC] Reduces a map with an addition lambda.
        return functools.reduce(lambda x, y: x + y, map(lambda x: x + 1, [a, b]))

    @make_test
    def test_map_sorted(a):
        # [FUNCTION DOC] Sorts a map of ints; returns it with a plus 1.
        return sorted(map(lambda x: x + 1, [0, 4, 3, 1, 2])), a + 1

    @make_test
    def test_map_list_extend(a, b, c):
        # [FUNCTION DOC] Extends a list with a map over [b, c]; returns the list.
        l = [a]
        l.extend(map(lambda x: x + 1, [b, c]))
        return l

    @make_test
    def test_map_list_slice_assign(a, b, c, d, e):
        # [FUNCTION DOC] Assigns a map to a slice of a list; returns the list.
        l = [a, b, c]
        l[1:2] = map(lambda x: x + 1, [d, e])
        return l

    @make_test
    def test_map_deque_extendleft(a, b, c):
        # [FUNCTION DOC] Extends a deque on the left with a map; returns the deque.
        # KNOWN ISSUE: a later method in this class has the same name and replaces this one, so
        # this version never runs.
        d = collections.deque([a])
        d.extendleft(map(lambda x: x + 1, [b, c]))
        return d

    @make_test
    def test_map_str_join(a):
        # [FUNCTION DOC] Joins a map of strings; returns it with a plus 1.
        return "".join(map(lambda x: x, ["a", "b", "c"])), a + 1

    def test_map_with_graph_break(self):
        # [FUNCTION DOC] Compiles a function using a map whose function mutates a closure variable
        # across a graph break; asserts eager equality and three frames.
        def f(a):
            # [FUNCTION DOC] HELPER: Increment a, build a map, take next before and after a
            # graph break.
            a += 1

            def g(x):
                # [FUNCTION DOC] HELPER: Increment the enclosing a and return x plus 1.
                nonlocal a
                a += 1
                return x + 1

            m = map(g, [1, 2, 3, 4, 5])
            a += next(m)  # won't graph break
            torch._dynamo.graph_break()
            a += next(m)  # will graph break
            return a

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        self.assertEqual(f(torch.ones(3, 3)), opt_f(torch.ones(3, 3)))
        self.assertEqual(cnts.frame_count, 3)

    def test_map_reconstruct(self):
        # [FUNCTION DOC] Checks a map object returned from compiled code is reconstructed as a map
        # with the same contents as eager.
        def fn(a):
            # [FUNCTION DOC] HELPER: Return a map over zipped pairs and a plus 1.
            return map(lambda x: x[0] + x[1], zip([1, 2, 3], [1, 2, 3])), a + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        m = opt_fn(torch.ones(3, 3))[0]
        self.assertIsInstance(m, map)
        self.assertEqual(list(m), list(fn(torch.ones(3, 3))[0]))

    def test_zip_reconstruct(self):
        # [FUNCTION DOC] Checks a zip object returned from compiled code is reconstructed as a zip
        # with the same contents as eager.
        def fn(a):
            # [FUNCTION DOC] HELPER: Return a zip over a list and a map and a plus 1.
            return zip([1, 2, 3], map(lambda x: x + 1, [1, 2, 3])), a + 1

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        m = opt_fn(torch.ones(3, 3))[0]
        self.assertIsInstance(m, zip)
        self.assertEqual(list(m), list(fn(torch.ones(3, 3))[0]))

    @make_test
    def test_map_partial_unpack(a, b):
        # [FUNCTION DOC] Zips a two-item list with a four-item map whose function mutates a
        # nonlocal counter, so only part of the map is consumed; returns a plus y.
        y = 1

        def f(x):
            # [FUNCTION DOC] HELPER: Increment the enclosing y and return x.
            nonlocal y
            y += 1
            return x

        l = list(zip([a, b], map(f, [1, 2, 3, 4])))  # noqa: F841
        return a + y

    @make_test
    def test_map_call_function_ex(a, b):
        # [FUNCTION DOC] Calls a function with unpacked map results.
        def f(x, y):
            # [FUNCTION DOC] HELPER: Return x plus y.
            return x + y

        return f(*map(lambda x: x + 1, [a, b]))

    @make_test
    def test_map_unpack_twice(a, b):
        # [FUNCTION DOC] Converts a map to a list twice; the second list is empty since it
        # is exhausted.
        m = map(lambda x: x + 1, [a, b])
        l1 = list(m)
        l2 = list(m)
        return l1, l2

    @make_test
    def test_enumerate(a, b):
        # [FUNCTION DOC] Enumerates [a, b] starting at 1 and returns the list with a plus 1.
        return list(enumerate([a, b], start=1)), a + 1

    @make_test
    def test_map_enumerate(a, b):
        # [FUNCTION DOC] Enumerates a map starting at 1 and returns the list with a plus 1.
        return list(enumerate(map(lambda x: x + 1, [a, b]), start=1)), a + 1

    @make_test
    def test_map_infinite(a, b):
        # [FUNCTION DOC] Maps a two-argument lambda over a list and an infinite count; the
        # shorter wins.
        return list(map(lambda x, y: x + y, [a, b], itertools.count(3)))

    @make_test
    def test_map_unpack_vars(a, b):
        # [FUNCTION DOC] Unpacks a map into two variables and sums them.
        x, y = map(lambda x: x + 1, [a, b])
        return x + y

    @make_test
    def test_map_list_extend_2(a):
        # [FUNCTION DOC] Extends a list with a map over a range whose function reads the list's
        # last item while it is being extended; returns a plus 1 and the list.
        y = [1]

        def inner(z):
            # [FUNCTION DOC] HELPER: Return z plus the last item of the captured list.
            return z + y[-1]

        y.extend(map(inner, range(3)))
        return a + 1, y

    @make_test
    def test_map_deque_extendleft(a):
        # [FUNCTION DOC] Extends a deque on the left with a map whose function reads its first
        # item; returns a plus 1 and the deque.
        y = collections.deque([1])

        def inner(z):
            # [FUNCTION DOC] HELPER: Return z plus the first item of the captured deque.
            return z + y[0]

        y.extendleft(map(inner, range(3)))
        return a + 1, y

    def test_unsqueeze_inplace(self):
        # [FUNCTION DOC] Compares in-place unsqueeze via the Tensor class and via the method on two
        # equal tensors; checks the compiled outputs are equal.
        def fn(x):
            # [FUNCTION DOC] HELPER: Call the unbound Tensor.unsqueeze_ on x at dim 1 and add 1.
            return torch.Tensor.unsqueeze_(x, dim=1) + 1

        def self_fn(x):
            # [FUNCTION DOC] HELPER: Call x.unsqueeze_ at dim 1 and add 1.
            return x.unsqueeze_(dim=1) + 1

        v = torch.ones([3])
        # identical tensor since modify inplace
        v2 = torch.ones([3])
        opt_fn = torch.compile(fn, backend="eager")
        opt_self_fn = torch.compile(self_fn, backend="eager")
        self.assertEqual(v, v2)
        self.assertEqual(opt_fn(v), opt_self_fn(v2))

    def test_enumerate_custom(self):
        # [FUNCTION DOC] Compiles a loop over enumerate of a user iterator class and compares
        # to eager.
        class MyClass:
            # [CLASS DOC] HELPER: Stateful iterator object: returns itself from iter, resets a
            # counter to 1 and yields increasing ints until the counter exceeds 3, then stops.
            def __iter__(self):
                # [FUNCTION DOC] HELPER: Reset the counter attribute a to 1 and return self.
                self.a = 1
                return self

            def __next__(self):
                # [FUNCTION DOC] HELPER: Raise StopIteration when a is above 3; else increment a
                # and return it.
                if self.a > 3:
                    raise StopIteration
                self.a += 1
                return self.a

        def fn(x):
            # [FUNCTION DOC] HELPER: Add index plus item for each element of enumerate over a
            # MyClass instance.
            for i, it in enumerate(MyClass()):
                x += i + it
            return x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(torch.ones(3, 3)), opt_fn(torch.ones(3, 3)))

    @unittest.skip("https://github.com/pytorch/pytorch/pull/146527 exposed a bug")
    def test_enumerate_reconstruct(self):
        # [FUNCTION DOC] Skipped (PR 146527 exposed a bug). Would check an enumerate object
        # returned from compiled code is reconstructed as enumerate with the same contents.
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Return enumerate over [a, b] starting at 1.
            return enumerate([a, b], start=1)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        inps = (torch.randn(3, 3), torch.randn(3, 3))
        it1 = fn(*inps)
        it2 = opt_fn(*inps)
        self.assertIsInstance(it2, enumerate)
        self.assertEqual(list(it1), list(it2))

    def test_returning_recursive_func(self):
        # [FUNCTION DOC] Compiles a function returning a closure that returns itself; asserts the
        # returned function is its own result.
        @torch.compile(backend="eager", fullgraph=True)
        def run(x):
            # [FUNCTION DOC] HELPER: Define a self-returning inner function; return x plus 1 and
            # the function.
            def f():
                # [FUNCTION DOC] HELPER: Return the enclosing function f itself.
                return f

            return x + 1, f

        res, f = run(torch.zeros(1))
        self.assertTrue(same(res, torch.ones(1)))
        self.assertTrue(f is f())

    def test_functools_partial_binding(self):
        # [FUNCTION DOC] Compiles a function that calls an lru_cache-decorated method on a small
        # object and reads the mutated attribute; compares to eager.
        class Foo:
            # [CLASS DOC] HELPER: Holds an attribute x; its incr method (cached with lru_cache)
            # adds val to x.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x on the instance.
                self.x = x

            @functools.lru_cache  # noqa: B019
            def incr(self, val):
                # [FUNCTION DOC] HELPER: Add val to the attribute x (cached per call arguments).
                self.x += val

        def fn(x):
            # [FUNCTION DOC] HELPER: Make a Foo holding 4, call incr with 3, return x plus its
            # attribute x.
            f = Foo(4)
            f.incr(3)
            return x + f.x

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_wrapper_user_function_hasattr(self):
        # [FUNCTION DOC] Checks that a function wrapped by lru_cache and then by a functools.wraps
        # based decorator at trace time works without a graph break from hasattr on its dict.
        # WrapperUserFunctionVariable (e.g. lru_cache-wrapped fn) passed to a
        # decorator that calls functools.wraps at tracing time should not graph
        # break on hasattr(fn, '__dict__').
        @functools.lru_cache
        def cached_fn(x):
            # [FUNCTION DOC] HELPER: lru_cache function returning x times 2.
            return x * 2

        def retry(func):
            # [FUNCTION DOC] HELPER: Decorator factory returning a wrapper (functools.wraps)
            # around func.
            @functools.wraps(func)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped func with all arguments.
                return func(*args, **kwargs)

            return wrapper

        def fn(x):
            # [FUNCTION DOC] HELPER: Wrap cached_fn with retry and call it on x.
            return retry(cached_fn)(x)

        x = torch.tensor(2.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_wraps_on_lru_cache_preserves_name(self):
        # [FUNCTION DOC] Checks functools.wraps on an lru_cache function copies name, qualname and
        # doc attributes at trace time; the compiled function raises AssertionError otherwise.
        # functools.wraps copies __name__ and __qualname__ from the wrapped fn;
        # when applied to an lru_cache-wrapped function at trace time,
        # WrapperUserFunctionVariable must expose those attributes.
        @functools.lru_cache
        def my_op(x):
            # [FUNCTION DOC] HELPER: lru_cache function returning x plus 1, with a docstring.
            """my docstring"""
            return x + 1

        def apply_wraps(func):
            # [FUNCTION DOC] HELPER: Return an inner function wrapped with the functools wraps
            # decorator on func.
            @functools.wraps(func)
            def inner(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped func with all arguments.
                return func(*args, **kwargs)

            return inner

        def fn(x):
            # [FUNCTION DOC] HELPER: Wrap my_op, check name, qualname and doc match the original
            # (raising AssertionError otherwise), and call the wrapper.
            wrapped = apply_wraps(my_op)
            if wrapped.__name__ != "my_op":
                raise AssertionError(f"Expected 'my_op', got {wrapped.__name__!r}")
            if wrapped.__qualname__ != my_op.__qualname__:
                raise AssertionError(
                    f"Expected {my_op.__qualname__!r}, got {wrapped.__qualname__!r}"
                )
            if wrapped.__doc__ != "my docstring":
                raise AssertionError(
                    f"Expected 'my docstring', got {wrapped.__doc__!r}"
                )
            return wrapped(x)

        x = torch.tensor(1.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_wraps_on_lru_cache_copies_annotations(self):
        # [FUNCTION DOC] Checks functools.wraps on an lru_cache function with annotations works
        # when traced; compares compiled and eager.
        # functools.wraps should copy __annotations__ from an lru_cache-wrapped fn.
        @functools.lru_cache
        def annotated_fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: lru_cache function with annotations returning x times 3.
            return x * 3

        def decorator(func):
            # [FUNCTION DOC] HELPER: Decorator returning a functools.wraps wrapper around func.
            @functools.wraps(func)
            def inner(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped func with all arguments.
                return func(*args, **kwargs)

            return inner

        def fn(x):
            # [FUNCTION DOC] HELPER: Decorate annotated_fn and call it on x.
            return decorator(annotated_fn)(x)

        x = torch.tensor(2.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_wraps_on_nested_fn(self):
        # [FUNCTION DOC] Checks functools.wraps on a locally defined function works without a graph
        # break and copies its name (see the checks in the function body).
        # functools.wraps applied to a locally defined (NestedUserFunctionVariable)
        # function should also work without graph breaks.
        def fn(x):
            # [FUNCTION DOC] HELPER: Define inner_op and wrapper, then verify the
            # copied attributes.
            def inner_op(t):
                # [FUNCTION DOC] HELPER: Return t times 2 (has a docstring).
                """inner doc"""
                return t * 2

            @functools.wraps(inner_op)
            def wrapper(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call inner_op with all arguments.
                return inner_op(*args, **kwargs)

            if wrapper.__name__ != "inner_op":
                raise AssertionError(f"Expected 'inner_op', got {wrapper.__name__!r}")
            if wrapper.__doc__ != "inner doc":
                raise AssertionError(f"Expected 'inner doc', got {wrapper.__doc__!r}")
            return wrapper(x)

        x = torch.tensor(3.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_wraps_stacked_on_lru_cache(self):
        # [FUNCTION DOC] Stacks two functools.wraps decorator layers over an lru_cache function and
        # calls the result under fullgraph; compares compiled to eager.
        # Stacking two functools.wraps layers over an lru_cache-wrapped fn.
        @functools.lru_cache
        def base_fn(x):
            # [FUNCTION DOC] HELPER: lru_cache function returning x minus 1.
            return x - 1

        def outer_decorator(func):
            # [FUNCTION DOC] HELPER: Decorator returning a wraps-based wrapper around func.
            @functools.wraps(func)
            def middle(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped func with all arguments.
                return func(*args, **kwargs)

            return middle

        def inner_decorator(func):
            # [FUNCTION DOC] HELPER: Decorator returning a wraps-based wrapper around func.
            @functools.wraps(func)
            def innermost(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Call the wrapped func with all arguments.
                return func(*args, **kwargs)

            return innermost

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply both decorators to base_fn and call the result on x.
            return inner_decorator(outer_decorator(base_fn))(x)

        x = torch.tensor(5.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_lru_cache_dunder_name_access(self):
        # [FUNCTION DOC] Checks the name attribute of an lru_cache function read during tracing is
        # the original name; the compiled function raises AssertionError otherwise.
        # Accessing __name__ on an lru_cache-wrapped function during tracing
        # should return the original function's name as a constant.
        @functools.lru_cache
        def compute(x):
            # [FUNCTION DOC] HELPER: lru_cache function returning x plus 10.
            return x + 10

        def fn(x):
            # [FUNCTION DOC] HELPER: Read the name of compute, verify it, then call compute on x.
            name = compute.__name__
            if name != "compute":
                raise AssertionError(f"Expected 'compute', got {name!r}")
            return compute(x)

        x = torch.tensor(1.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_lru_cache_dunder_doc_access(self):
        # [FUNCTION DOC] Checks the doc attribute of an lru_cache function read during tracing
        # equals its docstring; the compiled function raises AssertionError otherwise.
        # Accessing __doc__ on an lru_cache-wrapped function during tracing.
        @functools.lru_cache
        def documented_fn(x):
            # [FUNCTION DOC] HELPER: lru_cache function returning x squared; has a docstring.
            """returns x squared"""
            return x**2

        def fn(x):
            # [FUNCTION DOC] HELPER: Read the doc of documented_fn, verify it, then call it on x.
            doc = documented_fn.__doc__
            if doc != "returns x squared":
                raise AssertionError(f"Expected 'returns x squared', got {doc!r}")
            return documented_fn(x)

        x = torch.tensor(3.0)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_functools_cache_guard(self):
        # [FUNCTION DOC] Compiles a call to an lru_cache-decorated method (with a defaulted
        # argument) on a captured instance and compares with eager.
        class Foo:
            # [CLASS DOC] HELPER: Class with an lru_cache-decorated method run.
            @functools.lru_cache  # noqa: B019
            def run(self, val, c=1.0):
                # [FUNCTION DOC] HELPER: Return val times c times 2 (cached per arguments).
                return val * c * 2

        f = Foo()

        def fn(x):
            # [FUNCTION DOC] HELPER: Call run on the captured instance.
            return f.run(x)

        x = torch.randn(2)
        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertEqual(fn(x), opt_fn(x))

    def test_torch_get_device_module(self):
        # [FUNCTION DOC] Tests torch.get_device_module under fullgraph: default, cpu and explicit
        # device calls match eager; unknown keyword, both positional and keyword device, and an
        # invalid name each raise Unsupported; replacing torch.get_device_module via patch is
        # honored; a variant calling synchronize on the module compiles with a graph break.
        def f1():
            # [FUNCTION DOC] HELPER: Return the default, cpu and accelerator device modules.
            mod1 = torch.get_device_module()
            mod2 = torch.get_device_module("cpu")
            mod3 = torch.get_device_module(torch.device(device_type))
            return mod1, mod2, mod3

        self.assertEqual(f1(), torch.compile(f1, backend="eager", fullgraph=True)())

        @torch.compile(backend="eager", fullgraph=True)
        def f2():
            # [FUNCTION DOC] HELPER: Call get_device_module with an unsupported keyword.
            torch.get_device_module(foo="cpu")

        with self.assertRaises(Unsupported):
            f2()

        @torch.compile(backend="eager", fullgraph=True)
        def f3():
            # [FUNCTION DOC] HELPER: Call get_device_module with both positional and
            # keyword device.
            torch.get_device_module("cpu", device="cpu")

        with self.assertRaises(Unsupported):
            f3()

        @torch.compile(backend="eager", fullgraph=True)
        def f4():
            # [FUNCTION DOC] HELPER: Call get_device_module with an invalid device name.
            torch.get_device_module("asdf")

        with self.assertRaises(Unsupported):
            f4()

        # test for changing torch.get_device_module() (super rare case due to lru_cache)
        @torch.compile(backend="eager", fullgraph=True)
        def f5():
            # [FUNCTION DOC] HELPER: Return the default device module.
            return torch.get_device_module()

        f5()
        new_device = (
            "cpu"
            if torch._C._get_accelerator() == torch.device(device_type)
            else device_type
        )

        old_get_device_module = torch.get_device_module

        def new_get_device_module(device=None):
            # [FUNCTION DOC] HELPER: Replacement that delegates to the original if device is given,
            # else returns the module for the alternate device.
            if device:
                return old_get_device_module(device)
            return getattr(torch, new_device)

        # NOTE: torch.get_device_module.__wrapped__ is guarded on, but not
        # torch.get_device_module
        with patch("torch.get_device_module", new_get_device_module):
            print(torch.get_device_module())
            self.assertEqual(f5(), getattr(torch, new_device))

        # synchronize causes a graph break, so no fullgraph=True
        @torch.compile(backend="eager")
        def f6():
            # [FUNCTION DOC] HELPER: Get the default device module, call synchronize on it and
            # return it.
            mod = torch.get_device_module()
            mod.synchronize()
            return mod

        f6()

    def test_torch_source(self):
        # [FUNCTION DOC] Declares torch global and temporarily rebinds it to the int 1 in the
        # module globals while a compiled function that uses a saved reference to get_device_module
        # runs; expects a module return and restores the global in a finally clause.
        global torch

        g = torch.get_device_module

        @torch.compile(backend="eager", fullgraph=True)
        def f():
            # [FUNCTION DOC] HELPER: Return the result of calling the captured get_device_module.
            return g()

        try:
            old_torch = torch
            torch = 1
            self.assertEqual(torch, 1)
            self.assertIsInstance(f(), types.ModuleType)
        finally:
            torch = old_torch

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_wrap_triton_handled_during_tracing(self):
        # [FUNCTION DOC] Skipped without a GPU. Wraps a triton sine kernel with
        # torch.library.wrap_triton inside a function, compiles with fullgraph and compares to
        # torch.sin on a device tensor.
        import triton
        import triton.language as tl

        # [FUNCTION DOC] HELPER: Triton kernel; each program loads a block of x (masked by
        # n_elements), computes the sine and stores it to y.
        @triton.jit
        def sin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, tl.sin(x), mask=mask)

        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocate output, wrap the kernel, launch it with a grid and
            # block size 128.
            out = torch.empty_like(x)
            n_elements = x.numel()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrapped = torch.library.wrap_triton(sin_kernel)
            wrapped[grid](x, out, n_elements, BLOCK_SIZE=128)

            return out

        compiled = torch.compile(fn, fullgraph=True, backend="eager")

        x = torch.randn(1024, device=device_type)
        result = compiled(x)

        self.assertEqual(result, torch.sin(x))

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_capture_triton_handled_during_tracing(self):
        # [FUNCTION DOC] Skipped without a GPU. Same as the wrap_triton test but
        # using capture_triton.
        import triton
        import triton.language as tl

        # [FUNCTION DOC] HELPER: Triton kernel; each program loads a block of x (masked by
        # n_elements), computes the sine and stores it to y.
        @triton.jit
        def sin_kernel(x_ptr, y_ptr, n_elements, BLOCK_SIZE: "tl.constexpr"):
            pid = tl.program_id(axis=0)
            offsets = pid * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
            mask = offsets < n_elements
            x = tl.load(x_ptr + offsets, mask=mask)
            tl.store(y_ptr + offsets, tl.sin(x), mask=mask)

        def fn(x: torch.Tensor) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Allocate output, capture the kernel, launch it with a grid and
            # block size 128.
            out = torch.empty_like(x)
            n_elements = x.numel()

            grid = lambda meta: (triton.cdiv(n_elements, meta["BLOCK_SIZE"]),)

            wrapped = torch._library.capture_triton(sin_kernel)
            wrapped[grid](x, out, n_elements, BLOCK_SIZE=128)

            return out

        compiled = torch.compile(fn, fullgraph=True, backend="eager")

        x = torch.randn(1024, device=device_type)
        result = compiled(x)

        self.assertEqual(result, torch.sin(x))

    def test_property_descriptor_on_instance(self):
        # [FUNCTION DOC] Checks that a property defined on a class is evaluated on an instance in
        # compiled code (returns stored value plus 1).
        class Foo:
            # [CLASS DOC] HELPER: Stores _x; the property x returns _x plus 1.
            def __init__(self, x):
                # [FUNCTION DOC] HELPER: Store x in the private attribute.
                self._x = x

            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property returning the stored value plus 1.
                return self._x + 1

        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Return the property x of the object.
            return obj.x

        self.assertEqual(fn(Foo(torch.tensor(5))), torch.tensor(6))

    def test_property_descriptor_on_class(self):
        # [FUNCTION DOC] Checks accessing a property on the class (not an instance) yields a
        # property object inside compiled code.
        class Foo:
            # [CLASS DOC] HELPER: Class with a property x returning 42.
            @property
            def x(self):
                # [FUNCTION DOC] HELPER: Property returning 42.
                return 42

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Return whether Foo.x is a property object.
            return isinstance(Foo.x, property)

        self.assertTrue(fn())

    def test_tuplegetter_on_instance(self):
        # [FUNCTION DOC] Reads namedtuple fields via the tuple getter descriptors on an instance in
        # compiled code; expects 7.
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])

        @torch.compile(backend="eager", fullgraph=True)
        def fn(p):
            # [FUNCTION DOC] HELPER: Return p.x plus p.y.
            return p.x + p.y

        p = Point(torch.tensor(3), torch.tensor(4))
        self.assertEqual(fn(p), torch.tensor(7))

    def test_tuplegetter_doc_on_class(self):
        # [FUNCTION DOC] Reads the doc of a namedtuple field descriptor on the class; expects it to
        # contain the word Alias.
        from collections import namedtuple

        Point = namedtuple("Point", ["x", "y"])

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Return the doc attribute of the field descriptor.
            return Point.x.__doc__

        self.assertIn("Alias", fn())

    def test_getset_descriptor_on_instance(self):
        # [FUNCTION DOC] Reads the class attribute of an int and a str inside compiled code via a
        # getset descriptor; expects int and str.
        @torch.compile(backend="eager", fullgraph=True)
        def fn(obj):
            # [FUNCTION DOC] HELPER: Return the class of obj.
            return obj.__class__

        self.assertEqual(fn(42), int)
        self.assertEqual(fn("hello"), str)

    def test_type_getset_descriptor_metaclass_shadow(self):
        # [FUNCTION DOC] Checks that a metaclass defining properties named like type internals
        # (dict and mro) is used for ordinary attribute reads, while calling the raw type
        # descriptors bypasses it. Expects 3.0 and 10.0.
        class Meta(type):
            # [CLASS DOC] HELPER: Metaclass whose dict property returns a mapping with shadow set
            # to 1 and whose mro property returns a one-element tuple.
            @property
            def __dict__(cls):
                # [FUNCTION DOC] HELPER: Return a fake class dict with key shadow set to 1.
                return {"shadow": 1}

            @property
            def __mro__(cls):
                # [FUNCTION DOC] HELPER: Return a fake one-element mro tuple.
                return ("shadow",)

        class Foo(metaclass=Meta):
            # [CLASS DOC] HELPER: Class using the shadowing metaclass; has class attribute marker
            # equal to 7.
            marker = 7

        @torch.compile(backend="eager", fullgraph=True)
        def normal_lookup(x):
            # [FUNCTION DOC] HELPER: Add the shadowed dict entry and shadowed mro length to x.
            return x + Foo.__dict__["shadow"] + len(Foo.__mro__)

        @torch.compile(backend="eager", fullgraph=True)
        def direct_descriptor_call(x):
            # [FUNCTION DOC] HELPER: Read the true class dict and mro through the raw type
            # descriptors and add marker and mro length to x.
            cls_dict = type.__dict__["__dict__"].__get__(Foo)
            cls_mro = type.__dict__["__mro__"].__get__(Foo)
            return x + cls_dict["marker"] + len(cls_mro)

        x = torch.tensor(1.0)
        self.assertEqual(normal_lookup(x), torch.tensor(3.0))
        self.assertEqual(direct_descriptor_call(x), torch.tensor(10.0))

    def test_member_descriptor_isinstance_on_class(self):
        # [FUNCTION DOC] Checks that the slot descriptor on a class is a member descriptor type in
        # compiled code.
        class A:
            # [CLASS DOC] HELPER: Class using slots with one slot x.
            __slots__ = ("x",)

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Return whether A.x is a member descriptor.
            return isinstance(A.x, types.MemberDescriptorType)

        self.assertTrue(fn())


def udf_mul(x, y):
    # [FUNCTION DOC] HELPER: Return x times y.
    return x * y


def udf_mul2(x, y, z):
    # [FUNCTION DOC] HELPER: Return x times y times z.
    return x * y * z


def udf_add(x, y):
    # [FUNCTION DOC] HELPER: Return x plus y.
    return x + y


class SmallNN(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a forward that concatenates x and y along dim 1 and applies
    # ReLU twice.
    def forward(self, x, y):
        # [FUNCTION DOC] HELPER: Concatenate x and y on dim 1, apply ReLU twice and return.
        combined = torch.cat((x, y), dim=1)
        out = torch.nn.ReLU()(combined)
        out = torch.nn.ReLU()(out)
        return out


def udf_module(mod, x, y):
    # [FUNCTION DOC] HELPER: Call the module on x and y.
    return mod(x, y)


def global_func_with_default_tensor_args(
    x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))
):
    # [FUNCTION DOC] HELPER: Function with default tensor arguments (positional and keyword-only)
    # that are shared across calls; add 1 in place to each and return both.
    x.add_(1)
    kw_x.add_(1)
    return x, kw_x


class ModuleWithDefaultTensorArgsMethod(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward has default tensor arguments (shared across calls).
    def forward(self, x=torch.zeros((2, 2)), *, kw_x=torch.zeros((1, 2))):
        # [FUNCTION DOC] HELPER: Add 1 in place to each default tensor and return both.
        x.add_(1)
        kw_x.add_(1)
        return x, kw_x


class WrapperModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a ModuleWithDefaultTensorArgsMethod instance as m.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initialize the module and create the inner module m.
        super().__init__()
        self.m = ModuleWithDefaultTensorArgsMethod()

    def forward(self):
        # [FUNCTION DOC] HELPER: Call the inner module with no arguments.
        return self.m()


class DefaultsTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo tests for function and method default argument handling and several
    # dataclass, set, frozenset and tensor identity behaviors.
    def test_func_default_tensor_args(self):
        # [FUNCTION DOC] Alternates eager and compiled calls of a function mutating default tensor
        # arguments (original objects are shared); checks accumulated values. Compiled once (1
        # frame, 2 ops); patching __defaults__ or __kwdefaults__ with new tensors triggers
        # recompiles (frames 2, 3).
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """

        def func():
            # [FUNCTION DOC] HELPER: Call the global function with default tensor arguments.
            return global_func_with_default_tensor_args()

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_func = torch.compile(func, backend=cnts)
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = func()
            else:
                x, kw_x = compiled_func()
            # the inner func mutates += 1 each call
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # Calling compiled_func twice does not recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # But with a change to the guarded default tensor, we do recompile
        with patch.object(
            global_func_with_default_tensor_args,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_func()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            global_func_with_default_tensor_args,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_func()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    @torch._dynamo.config.patch(assume_dunder_attributes_remain_unchanged=False)
    def test_meth_default_tensor_args(self):
        # [FUNCTION DOC] Method version of the default tensor argument test: a wrapper module whose
        # inner forward has default tensors; checks shared mutation and recompiles after patching
        # the defaults on the forward function. Runs with the dunder-attributes-unchanged
        # assumption off.
        """
        Tests that we indeed reference (and mutate) "the one" default tensor arg
        stored on the globally allocated function object, both from the orig and
        compiled function
        """
        mod = WrapperModule()
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_mod = torch.compile(mod, backend=cnts)
        for i in range(4):
            if i % 2 == 0:
                x, kw_x = mod()
            else:
                x, kw_x = compiled_mod()
            # the inner func mutates += 1 each call
            self.assertTrue(same(x, torch.ones_like(x) + i))
            self.assertTrue(same(kw_x, torch.ones_like(kw_x) + i))
        # Calling compiled_func twice does not recompile
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 2)

        # But with a change to the guarded default tensor, we do recompile
        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__defaults__",
            (torch.ones((3, 4, 5)),),
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 2)
        self.assertEqual(cnts.op_count, 4)

        with patch.object(
            ModuleWithDefaultTensorArgsMethod.forward,
            "__kwdefaults__",
            {"kw_x": torch.ones((3, 4, 5))},
        ):
            x, kw_x = compiled_mod()
        self.assertEqual(cnts.frame_count, 3)
        self.assertEqual(cnts.op_count, 6)

    def test_guard_on_constant_func_defaults(self):
        # [FUNCTION DOC] Checks a compiled function taking a callable recompiles when given a
        # closure with the same code but a different default value (frame count 1 then 2).
        """
        When a compiled function is re-invoked with a closure whose
        __code__ is the same but __defaults__ differ (e.g. a different
        constant default arg), Dynamo must recompile instead of reusing
        the stale graph.
        """
        cnts = torch._dynamo.testing.CompileCounter()

        def make_adder(offset):
            # [FUNCTION DOC] HELPER: Return a function that adds a default offset to its argument.
            def adder(x, _offset=offset):
                # [FUNCTION DOC] HELPER: Add the default offset to x.
                return x + _offset

            return adder

        @torch.compile(backend=cnts)
        def call_adder(x, fn):
            # [FUNCTION DOC] HELPER: Call fn on x.
            return fn(x)

        x = torch.ones(4)

        adder0 = make_adder(0)
        result0 = call_adder(x, adder0)
        self.assertEqual(result0, x + 0)
        self.assertEqual(cnts.frame_count, 1)

        # Same __code__, different __defaults__ → must recompile
        adder5 = make_adder(5)
        result5 = call_adder(x, adder5)
        self.assertEqual(result5, x + 5)
        self.assertEqual(cnts.frame_count, 2)

    def test_guard_on_pos_default_mapping(self):
        # [FUNCTION DOC] Compiles a caller that passes a keyword argument to a function with
        # defaults; patching the defaults to one value makes the missing positional argument raise
        # an error whose message mentions the missing required argument b.
        cnts = torch._dynamo.testing.CompileCounter()

        def target(x, b=1, c=2):
            # [FUNCTION DOC] HELPER: Return x plus b (c unused).
            return x + b

        @torch.compile(backend=cnts)
        def caller(x):
            # [FUNCTION DOC] HELPER: Call target with keyword c equal to 3.
            return target(x, c=3)

        x = torch.ones(())
        self.assertEqual(caller(x), x + 1)
        self.assertEqual(cnts.frame_count, 1)

        with patch.object(target, "__defaults__", (1,)):
            expected_msg = "missing .* required positional argument: 'b'"
            self.assertRaisesRegex(Exception, expected_msg, caller, x)

    def test_func_default_torch_args(self):
        # [FUNCTION DOC] Checks function defaults that are torch dtype, Size and device give the
        # same dtype, device and size when compiled; one frame and one op.
        """
        Tests other types of torch types as function default (size, dtype, device)
        """

        def func_with_default_torch_args(
            dt=torch.float16, ds=torch.Size((1, 2, 3)), dd=torch.device("cpu")
        ):
            # [FUNCTION DOC] HELPER: Return a ones tensor with the default size, dtype and device.
            return torch.ones(ds, dtype=dt, device=dd)

        def func():
            # [FUNCTION DOC] HELPER: Call the function using its default arguments.
            return func_with_default_torch_args()

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_func = torch.compile(func, backend=cnts)
        out = func()
        compiled_out = compiled_func()
        self.assertEqual(out.dtype, compiled_out.dtype)
        self.assertEqual(out.device, compiled_out.device)
        self.assertEqual(out.size(), compiled_out.size())
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 1)

    def test_shadowed_dataclass_field_property_matches_eager(self):
        # [FUNCTION DOC] Checks a dataclass whose field name is shadowed by a read-only
        # property: the constructor fails with an attribute error, and compiled code must raise
        # a matching error.
        @dataclass
        class Data:
            # [CLASS DOC] HELPER: Dataclass with fields value and edges, where edges is then
            # redefined as a read-only property returning random numbers.
            value: torch.Tensor
            edges: torch.Tensor

            @property
            def edges(self) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Property returning a random tensor of three elements.
                return torch.randn(3)

        def fn(x):
            # [FUNCTION DOC] HELPER: Construct Data with both fields and return the sum of value.
            data = Data(value=x, edges=x)
            return data.value.sum()

        x = torch.randn(3)
        expected_msg = (
            r"(property 'edges' of .*Data' object has no setter"
            r"|can't set attribute(?: 'edges')?)"
        )
        self.assertRaisesRegex(AttributeError, expected_msg, fn, x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        self.assertRaisesRegex(Exception, expected_msg, opt_fn, x)

    def test_dataclass_factory(self):
        # [FUNCTION DOC] Tests dataclass default factories, methods, field assignment and returning
        # a dataclass from compiled code; one frame, five ops, and the results equal eager.
        @dataclass
        class Output:
            # [CLASS DOC] HELPER: Dataclass with int scalar (default 2), dict named_tensors and
            # list lists (both from default factories) and a scale method.
            scalar: int = 2
            named_tensors: dict[str, torch.Tensor] = field(default_factory=dict)
            lists: list[torch.Tensor] = field(default_factory=list)

            def scale(self):
                # [FUNCTION DOC] HELPER: Return scalar times 2.
                return self.scalar * 2

        def fn(x):
            # [FUNCTION DOC] HELPER: Construct dataclass instances with default and explicit
            # values, mutate one, and return it with a tensor expression.
            # Check default dict assignment
            a = Output(1)
            # Check that dataclass methods can be inlined
            scaled_value = a.scale()

            # Check that normal assignment works
            b = Output(5, named_tensors={"x": x})

            # Check default int assignment
            c = Output()

            # Check that the default members are properly initialized
            if isinstance(a.named_tensors, dict):
                x = torch.sin(x)

            # Change dataclass
            c.scalar = 6
            c.named_tensors["x"] = x

            # Return dataclaass as well to check reconstruction
            return c, torch.cos(x) * scaled_value + b.named_tensors["x"] + c.scalar

        cnts = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        x = torch.randn(4)
        eager_dataclass, out = fn(x)
        compiled_dataclass, compiled_out = compiled_fn(x)
        self.assertEqual(eager_dataclass.scalar, compiled_dataclass.scalar)
        self.assertEqual(
            eager_dataclass.named_tensors["x"], compiled_dataclass.named_tensors["x"]
        )
        self.assertTrue(same(out, compiled_out))
        self.assertEqual(cnts.frame_count, 1)
        self.assertEqual(cnts.op_count, 5)

    def test_dataclass_nested(self):
        # [FUNCTION DOC] Checks inheritance between dataclasses with a default factory field; the
        # constructed derived object provides outer_a multiplied with x.
        @dataclass
        class Base:
            # [CLASS DOC] HELPER: Dataclass with two int fields.
            outer_a: int
            outer_b: int

        @dataclass
        class Derived(Base):
            # [CLASS DOC] HELPER: Dataclass extending Base with a list field from a
            # default factory.
            inner_a: Any = field(default_factory=list)

        def fn(x):
            # [FUNCTION DOC] HELPER: Build a Derived with Base fields 1 and 2 and return outer_a
            # times x.
            l = Derived(1, 2)
            return l.outer_a * x

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_pydantic_dataclass_construction(self):
        # [FUNCTION DOC] Checks a dataclass with a custom init and a pydantic marker constructed in
        # compiled code causes a graph break: zero frames compile, and exactly one counter entry
        # mentions the Pydantic dataclass constructor.
        @torch._dynamo.disable
        def populate(self, x, y):
            # [FUNCTION DOC] HELPER: Dynamo-disabled function that stores x and y on the object.
            self.x = x
            self.y = y

        @dataclass(init=False)
        class Point:
            # [CLASS DOC] HELPER: Dataclass with tensor fields x and y with a hand-written init
            # that calls the disabled populate; marked with the pydantic sentinel attribute.
            x: torch.Tensor
            y: torch.Tensor
            # Pydantic uses this sentinel on decorated dataclasses.
            __is_pydantic_dataclass__ = True

            def __init__(self, x, y):
                # [FUNCTION DOC] HELPER: Fill the fields via the disabled helper.
                populate(self, x, y)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Build a Point and return the sum of its fields.
            p = Point(x=x, y=y)
            return p.x + p.y

        torch._dynamo.reset()
        counters.clear()
        cnts = torch._dynamo.testing.CompileCounter()
        compiled_fn = torch.compile(fn, backend=cnts)
        x = torch.randn(4)
        y = torch.randn(4)

        self.assertTrue(same(fn(x, y), compiled_fn(x, y)))
        self.assertEqual(cnts.frame_count, 0)
        self.assertEqual(cnts.op_count, 0)
        # Skipping the whole frame records a second follow-on graph break, so
        # assert on the specific pydantic entry rather than the raw count.
        self.assertEqual(
            [
                count
                for msg, count in counters["graph_break"].items()
                if "Pydantic dataclass constructor" in msg
            ],
            [1],
        )

    def test_listlike_of_tensors_contains_constant(self):
        # [FUNCTION DOC] For set and list: compiles a function that mutates a tensor in place,
        # wraps it in the container and tests membership of the int 1; compares to eager.
        for listlike in [set, list]:

            def fn(x):
                # [FUNCTION DOC] HELPER: Add 1 in place to x, build the container, return whether 1
                # is in it.
                x.add_(1)
                s = listlike([x])
                res = 1 in s
                return res

            opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
            x = torch.randn(1)
            ref = opt_fn(x)
            res = fn(x)
            self.assertEqual(ref, res)

    def test_cast_tensor_single_elem(self):
        # [FUNCTION DOC] With scalar-output capture enabled, compiles int and float conversions of
        # a single-element tensor for several value types and compares to eager; a two-element
        # tensor raises ValueError in eager and TorchRuntimeError when compiled.
        with torch._dynamo.config.patch({"capture_scalar_outputs": True}):
            for t, val in [
                (float, 1.0),
                (float, 1),
                (float, True),
                (int, 1),
                (int, False),
                # (int, 1.0), # fails due to a >= 0 comparison in sym_int
            ]:  # , bool, complex]: no casting for sym_bool, no sym_complex

                def fn(x):
                    # [FUNCTION DOC] HELPER: Add 1 and convert with the loop's type.
                    x = x + 1
                    return t(x)

                opt_fn = torch.compile(
                    fn, backend="eager", fullgraph=True, dynamic=False
                )
                x = torch.tensor([val])
                res = fn(x)
                ref = opt_fn(x)
                self.assertEqual(ref, res)

                # Cannot handle non single-elem
                with self.assertRaises(ValueError):
                    fn(torch.tensor([val] * 2))
                with self.assertRaises(torch._dynamo.exc.TorchRuntimeError):
                    opt_fn(torch.tensor([val] * 2))

    def test_set_construction(self):
        # [FUNCTION DOC] Builds a set with a tensor after in-place add, adds the tensor again and
        # returns the set size; compares to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Add 1 in place, build a set, add the same object and return
            # the length.
            y = x.add_(1)
            s = set({x})
            s.add(y)
            return len(s)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    @parametrize("_type", [set, frozenset], name_fn=lambda t: t.__name__)
    def test_set_call___init__(self, _type):
        # [FUNCTION DOC] Parametrized over set and frozenset. Builds the type from strings, calls
        # init again with new strings and branches on whether banana remains (kept for frozenset).
        @make_test
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Create the collection, re-initialize it, and return a plus b
            # if banana is in it, else a minus b.
            s = _type({"apple", "banana", "cherry"})
            s.__init__({"google", "microsoft", "apple"})
            # frozenset should remain the same while set gets updated
            if "banana" in s:
                return a + b
            else:
                return a - b

        fn(self)

    @parametrize(
        "method_name",
        [
            "copy",
            "difference",
            "intersection",
            "symmetric_difference",
            "union",
        ],
    )
    def test_frozenset_return_type(self, method_name):
        # [FUNCTION DOC] Parametrized over copy, difference, intersection, symmetric_difference and
        # union. Checks that these operations on a frozenset return a frozenset.
        @make_test
        def fn(a, b):
            # [FUNCTION DOC] HELPER: Apply the method to frozensets; a plus b if the result is a
            # frozenset, else a minus b.
            set1 = frozenset({"apple", "banana", "cherry"})
            set2 = frozenset({"google", "microsoft", "apple"})
            if method_name == "copy":
                result = set1.copy()
            else:
                result = getattr(set1, method_name)(set2)
            if type(result) is frozenset:
                x = a + b
            else:
                x = a - b
            return x

        fn(self)

    def test_frozenset_construction(self):
        # [FUNCTION DOC] Constructs a frozenset from a tensor and then from the frozenset; compares
        # the length with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Build frozensets in two steps and return the length.
            s = frozenset({x})
            t = frozenset(s)
            return len(t)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_frozenset_reconstruction(self):
        # [FUNCTION DOC] Uses an empty frozenset as a dict key after a graph break to check
        # frozenset reconstruction; compares to eager.
        d = {}
        f = frozenset()
        d[f] = torch.randn(4)

        def fn(x):
            # [FUNCTION DOC] HELPER: Create an empty frozenset, graph break and index the dict
            # by it.
            k = frozenset()
            torch._dynamo.graph_break()
            return d[k] * x

        opt_fn = torch.compile(fn, backend="eager")
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_frozenset_reconstruction2(self):
        # [FUNCTION DOC] Returns a frozenset containing a tensor from compiled code; compares
        # to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return x plus 1 and a frozenset of one and x.
            s = frozenset([1, x])
            return x + 1, s

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        res = fn(x)
        ref = opt_fn(x)
        self.assertEqual(ref, res)

    def test_is_tensor_tensor(self):
        # [FUNCTION DOC] Checks the identity check between tensors in compiled code (dynamic
        # shapes) for distinct and identical tensors.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return x times 2 if x is y, else x plus y.
            if x is y:
                return x * 2
            else:
                return x + y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2)
        y = torch.ones(2)

        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_not_tensor_tensor(self):
        # [FUNCTION DOC] Checks the negative identity check between tensors for distinct and
        # identical tensors.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Return x times 2 if x is not y, else x plus y.
            if x is not y:
                return x * 2
            else:
                return x + y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        x = torch.zeros(2)
        y = torch.ones(2)

        self.assertEqual(fn(x, y), fn_opt(x, y))
        self.assertEqual(fn(x, x), fn_opt(x, x))

    def test_is_mutated_tensor_tensor(self):
        # [FUNCTION DOC] Checks that the result of an in-place add on a tensor is the same object.
        # KNOWN ISSUE: a later method in this class has the same name, so this one never runs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Add 1 in place and return whether the result is x.
            y = x.add_(1)
            return x is y

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor_across_graph_break(self):
        # [FUNCTION DOC] Compiles with dynamic shapes (no fullgraph) a function that checks
        # identity of a tensor and its in-place result before and after a graph break; compares
        # with eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Do in-place adds around a graph break and track the
            # identity condition.
            y = x.add_(1)
            cond = x is y
            x.add_(1)
            # The real tensor values are recovered when graph breaking.
            # Hence we recover the invariant.
            torch._dynamo.graph_break()
            x.add_(1)
            return x is y, cond

        fn_opt = torch.compile(backend="eager", dynamic=True)(fn)

        z = torch.ones(4)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_mutated_tensor_tensor(self):
        # [FUNCTION DOC] Checks the result of an in-place add is the same object, compiled
        # with fullgraph and dynamic shapes (this definition replaces an earlier one with the
        # same name).
        def fn(x):
            # [FUNCTION DOC] HELPER: Add 1 in place and return whether the result is x.
            y = x.add_(1)
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_mutated_tensor_tensor(self):
        # [FUNCTION DOC] Checks identity between a clone created inside the compiled function and
        # the result of its in-place add.
        def fn(x):
            # [FUNCTION DOC] HELPER: Clone x, add in place and return whether the result is
            # the clone.
            z = x.clone()
            y = z.add_(1)
            return y is z

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_vmapped_mutated_tensor_tensor(self):
        # [FUNCTION DOC] Checks identity of results of a vmapped in-place op against clones created
        # inside compiled code.
        def fn(z):
            # [FUNCTION DOC] HELPER: Clone z, apply vmapped acos_ in place and compare identities.
            x = z.clone()
            y = torch.vmap(torch.Tensor.acos_)(x)
            _ = y is z
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_vmapped_mutated_tensor_tensor(self):
        # [FUNCTION DOC] Checks identity of a vmapped in-place op result and its input.
        def fn(x):
            # [FUNCTION DOC] HELPER: Apply vmapped acos_ to x and return whether the result is x.
            y = torch.vmap(torch.Tensor.acos_)(x)
            return y is x

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        z = torch.ones(4, 1)

        self.assertEqual(fn(z), fn_opt(z))

    def test_is_init_in_compile_vmapped_mutated_tensor_tensor_multi_arg(self):
        # [FUNCTION DOC] Checks identity chains for a vmapped function with two in-place outputs,
        # for different inputs and for the same tensor passed twice.
        def fn(y, z):
            # [FUNCTION DOC] HELPER: Clone both inputs, vmap an in-place function and return the
            # identity chain.
            a = y.clone()
            b = z.clone()

            def g(a, b):
                # [FUNCTION DOC] HELPER: Apply in-place acos to both arguments and return the pair.
                return a.acos_(), b.acos_()

            c, d = torch.vmap(g)(a, b)
            return a is c is b is d

        fn_opt = torch.compile(backend="eager", fullgraph=True, dynamic=True)(fn)

        y = torch.ones(4, 2)
        z = torch.ones(4, 10)

        self.assertEqual(fn(y, z), fn_opt(y, z))
        self.assertEqual(fn(y, y), fn_opt(y, y))

    def test_in_set_would_fail_broadcast(self):
        # [FUNCTION DOC] Checks set membership of tensors with mismatched shapes does not raise
        # from broadcasting; also checks that aliasing the arguments causes a recompile (frames 1
        # then 2).
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        tensor_list = set()
        tensor_list.add(param2)
        if param in tensor_list:
            raise AssertionError("Expected param not in tensor_list")

        def fn(param, param2):
            # [FUNCTION DOC] HELPER: Add 1 in place to param, build a set with param2 and
            # test membership.
            param.add_(1)
            tensor_list = set([param2])
            return param in tensor_list

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        self.assertEqual(cnts.frame_count, 1)
        # Test aliased
        self.assertEqual(opt_fn(param, param), fn(param, param))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_in_set_inplace(self):
        # [FUNCTION DOC] Same membership test with tensors modified by method and by function form
        # of the in-place op; aliased arguments cause a recompile.
        param = torch.zeros(5)
        param2 = torch.zeros(5, 10)

        tensor_list = set()
        tensor_list.add(param2)
        if param in tensor_list:
            raise AssertionError("Expected param not in tensor_list")

        def fn(param, param2):
            # [FUNCTION DOC] HELPER: Add in place twice and return membership of both results.
            y = param.add_(1)  # Tensor method
            z = torch.Tensor.add_(y, 1)  # torch function
            tensor_list = set([param2])
            return y in tensor_list and z in tensor_list

        cnts = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(fn, backend=cnts, fullgraph=True)
        self.assertEqual(opt_fn(param, param2), fn(param, param2))
        self.assertEqual(cnts.frame_count, 1)
        # Test aliased
        self.assertEqual(opt_fn(param, param), fn(param, param))
        self.assertEqual(cnts.frame_count, 2)  # Recompiles

    def test_reconstructed_name(self):
        # [FUNCTION DOC] Checks the name attribute of a nested function passed to a Dynamo-disabled
        # function is the same in compiled and eager runs.
        lst = []

        @torch._dynamo.disable
        def disallowed(g):
            # [FUNCTION DOC] HELPER: Dynamo-disabled; append the name of g to the captured list.
            lst.append(g.__name__)

        def f():
            # [FUNCTION DOC] HELPER: Define a nested function g and pass it to the
            # disabled function.
            def g():
                # [FUNCTION DOC] HELPER: Return an empty tuple.
                return ()

            disallowed(g)

        opt_f = torch.compile(f, backend="eager")
        opt_f()
        f()
        self.assertEqual(len(lst), 2)
        self.assertEqual(lst[0], lst[1])

    def test_zip_strict(self):
        # [FUNCTION DOC] Tests strict zip: results and zip type match eager; with mismatched
        # lengths fullgraph raises Unsupported mentioning zip, while graph-break mode falls back
        # and raises ValueError.
        def fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Accumulate products with a strict zip and return x and an
            # unstrict zip.
            x = x.clone()
            for y, z in zip(ys, zs, strict=True):
                x += y * z
            return x, zip(ys, zs)

        opt_fn = torch.compile(fn, backend="eager")
        nopython_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.ones(3)
        ys = [1.0, 2.0, 3.0]
        zs = [2.0, 5.0, 8.0]

        ref = fn(x, ys, zs)
        res = opt_fn(x, ys, zs)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(list(ref[1]), list(res[1]))
        self.assertIsInstance(res[1], zip)

        # If nopython, should raise Unsupported
        with self.assertRaisesRegex(Unsupported, "zip()"):
            nopython_fn(x, ys[:1], zs)

        with self.assertRaisesRegex(Unsupported, "zip()"):
            nopython_fn(x, ys, zs[:1])

        # Should cause fallback if allow graph break
        with self.assertRaisesRegex(ValueError, "zip()"):
            opt_fn(x, ys[:1], zs)

        with self.assertRaisesRegex(ValueError, "zip()"):
            opt_fn(x, ys, zs[:1])

    def test_map_strict(self):
        # [FUNCTION DOC] Tests strict map. Before Python 3.14 the strict keyword raises TypeError
        # (or Unsupported with fullgraph). From 3.14, results match eager and mismatched lengths
        # raise Unsupported (fullgraph) or ValueError; a map returned from Dynamo stays strict.
        def fn(x, ys, zs):
            # [FUNCTION DOC] HELPER: Accumulate products using a strict map and return x and a
            # strict map.
            x = x.clone()
            for y, z in map(lambda a, b: (a, b), ys, zs, strict=True):
                x += y * z
            return x, map(lambda a, b: a + b, ys, zs, strict=True)

        opt_fn = torch.compile(fn, backend="eager")
        nopython_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = torch.ones(3)
        ys = [1.0, 2.0, 3.0]
        zs = [2.0, 5.0, 8.0]

        if sys.version_info < (3, 14):
            with self.assertRaises(TypeError):
                opt_fn(x, ys, zs)
            torch._dynamo.reset()
            with self.assertRaises((TypeError, torch._dynamo.exc.Unsupported)):
                nopython_fn(x, ys, zs)
            return

        ref = fn(x, ys, zs)
        res = opt_fn(x, ys, zs)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(list(ref[1]), list(res[1]))
        self.assertIsInstance(res[1], map)

        # If nopython, should raise UserError
        with self.assertRaisesRegex(Unsupported, "map()"):
            nopython_fn(x, ys[:1], zs)

        with self.assertRaisesRegex(Unsupported, "map()"):
            nopython_fn(x, ys, zs[:1])

        # Should cause fallback if allow graph break
        with self.assertRaisesRegex(ValueError, "map()"):
            opt_fn(x, ys[:1], zs)

        with self.assertRaisesRegex(ValueError, "map()"):
            opt_fn(x, ys, zs[:1])

        # Check strict is set by testing a map returned from dynamo
        opt_map_fn = torch.compile(
            lambda ys, zs: map(lambda a, b: a + b, ys, zs, strict=True), backend="eager"
        )
        strict_map_from_dynamo = opt_map_fn(ys[:1], zs)
        with self.assertRaises(ValueError):
            list(strict_map_from_dynamo)

    @unittest.skipIf(sys.version_info < (3, 14), "strict requires Python 3.14+")
    def test_map_strict_with_graph_break(self):
        # [FUNCTION DOC] Skipped before Python 3.14. A strict map whose function mutates a closure
        # variable, with a graph break between next calls; asserts eager equality and 3 frames.
        def f(a):
            # [FUNCTION DOC] HELPER: Increment a, create a strict map, take next before and after a
            # graph break.
            a += 1

            def g(x, y):
                # [FUNCTION DOC] HELPER: Increment the enclosing a and return x plus y.
                nonlocal a
                a += 1
                return x + y

            m = map(g, [1, 2, 3, 4, 5], [1, 2, 3, 4, 5], strict=True)
            a += next(m)  # won't graph break
            torch._dynamo.graph_break()
            a += next(m)  # will graph break
            return a

        cnts = torch._dynamo.testing.CompileCounter()
        opt_f = torch.compile(f, backend=cnts)
        self.assertEqual(f(torch.ones(3, 3)), opt_f(torch.ones(3, 3)))
        self.assertEqual(cnts.frame_count, 3)

    def test_fn_with_attr(self):
        # [FUNCTION DOC] Compiles a function that reads an attribute set on the function object;
        # changing the attribute and recompiling gives the other branch (frame counts 1 then 2).
        def fn(x):
            # [FUNCTION DOC] HELPER: Return relu of x times 2 if the function attribute pred is
            # true, else abs of x plus 3.
            if fn.pred:
                return torch.relu(x * 2)
            else:
                return torch.abs(x + 3)

        t = torch.ones(3)
        counter = torch._dynamo.testing.CompileCounter()
        fn.pred = True
        opt_fn_0 = torch.compile(fullgraph=True, backend=counter)(fn)
        self.assertEqual(opt_fn_0(t), fn(t))
        self.assertEqual(counter.frame_count, 1)
        fn.pred = False
        opt_fn_1 = torch.compile(fullgraph=True, backend=counter)(fn)
        self.assertEqual(opt_fn_1(t), fn(t))
        self.assertEqual(counter.frame_count, 2)

    def test_str_handler_for_user_defined_object(self):
        # [FUNCTION DOC] Checks str of a user object with and without a custom string method is the
        # same in compiled and eager runs (the default form compared without memory address).
        """
        Confirms handler behaviour for `str` is the same between eager and dynamo.
        Compares a user defined object with custom `__str__` method and without.
        """

        class CustomStr:
            # [CLASS DOC] HELPER: Class with a string method returning ok.
            def __str__(self):
                # [FUNCTION DOC] HELPER: Return the string ok.
                return "ok"

        def foo_custom_str(x):
            # [FUNCTION DOC] HELPER: Return x and the string of a new CustomStr.
            a = CustomStr()
            return x, str(a)

        eager_custom_str = foo_custom_str(torch.ones(4))
        dynamo_custom_str = torch.compile(
            foo_custom_str, fullgraph=True, backend="eager"
        )(torch.ones(4))

        self.assertEqual(eager_custom_str[1], dynamo_custom_str[1])
        self.assertEqual(eager_custom_str[1], "ok")

        class DefaultStr:
            # [CLASS DOC] HELPER: Empty class using the default object string form.
            pass

        def foo_default_str(x):
            # [FUNCTION DOC] HELPER: Return x and the string of a new DefaultStr.
            a = DefaultStr()
            return x, str(a)

        eager_default_str = foo_default_str(torch.ones(4))
        dynamo_default_str = torch.compile(
            foo_default_str, fullgraph=True, backend="eager"
        )(torch.ones(4))

        # Check that the tensor output from eager and dynamo modes are the same
        self.assertEqual(eager_default_str[0], dynamo_default_str[0])

        # Check that the class name (without memory address) is the same in both modes
        eager_class_name = eager_default_str[1].split(" object at")[0]
        dynamo_class_name = dynamo_default_str[1].split(" object at")[0]
        self.assertEqual(eager_class_name, dynamo_class_name)

    def test_pybind_object(self):
        # [FUNCTION DOC] Uses a pybind GuardDebugInfo object whose result attribute is true or
        # false to choose cos or sin of x in compiled code; compares to eager for both.
        def fn(x, pybind_obj):
            # [FUNCTION DOC] HELPER: Return cos of x if the object's result attribute is truthy,
            # else sin.
            if pybind_obj.result:
                return torch.cos(x)
            return torch.sin(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        pybind_obj = torch._C._dynamo.guards.GuardDebugInfo(True, ["a==1"], 0)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, pybind_obj), fn(x, pybind_obj))

        pybind_obj = torch._C._dynamo.guards.GuardDebugInfo(False, ["a==1"], 1)
        x = torch.randn(4)
        self.assertEqual(opt_fn(x, pybind_obj), fn(x, pybind_obj))

    def test_tree_map(self):
        # [FUNCTION DOC] Applies pytree tree_map_only with an indexing function over a tuple of
        # tensors in compiled code; compares to eager.
        def fn(a, b, index):
            # [FUNCTION DOC] HELPER: Define a nested function and return the result of calling it
            # with index.
            def call(index):
                # [FUNCTION DOC] HELPER: Return tree_map_only applied to the tuple of a and b,
                # selecting by index.
                mapped_attributes = torch.utils._pytree.tree_map_only(
                    torch.Tensor,
                    lambda x: x[index],
                    (a, b),
                )
                return mapped_attributes

            return call(index)

        a = torch.randn(4, 2, 5)
        b = torch.randn(4, 2, 5, 5)
        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager")
        compiled_res = compiled_fn(a, b, torch.tensor([2]))
        reference_res = fn(a, b, torch.tensor([2]))
        self.assertTrue(same(compiled_res, reference_res))

    def test_fx_map_aggregate(self):
        # [FUNCTION DOC] Applies fx map_aggregate to a nested list and a dict of tensors; compares
        # values and result types with eager.
        def fn(inputs, f):
            # [FUNCTION DOC] HELPER: Return the map_aggregate of inputs with function f.
            return torch.fx.node.map_aggregate(inputs, f)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)

        x = [torch.randn(4), [torch.randn(4), torch.randn(4)]]

        def f(y):
            # [FUNCTION DOC] HELPER: Return y times 2.
            return y * 2

        ref = fn(x, f)
        res = opt_fn(x, f)
        self.assertEqual(ref, res)
        # Check type(res) is immutable_list
        self.assertTrue(type(ref) is type(res))

        x = {
            "a": torch.randn(4),
            "b": [torch.randn(4), torch.randn(4)],
        }
        ref = fn(x, f)
        res = opt_fn(x, f)
        self.assertEqual(ref, res)
        self.assertTrue(type(ref) is type(res))

    def test_fx_immutable_list_mutation_not_allowed(self):
        # [FUNCTION DOC] Checks that appending to an fx immutable list raises TypeError (caught)
        # and map_aggregate still gives the eager result.
        def fn(inputs, x, f=lambda x: x * 2):
            # [FUNCTION DOC] HELPER: Make an immutable list, try appending (TypeError swallowed),
            # then map_aggregate.
            immutable_inputs = torch.fx.immutable_collections.immutable_list(inputs)
            try:
                immutable_inputs.append(x)
            except TypeError:
                pass
            return torch.fx.node.map_aggregate(immutable_inputs, f)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        inputs = [torch.randn(4), [torch.randn(4), torch.randn(4)]]
        x = torch.randn(4)

        self.assertEqual(fn(inputs, x), opt_fn(inputs, x))

    def test_udf_tuple(self):
        # [FUNCTION DOC] Compiles a function using a tuple subclass with an overridden contains
        # method and extra method; checks values and that the overridden method ran (checked flag).
        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Tuple subclass with a method returning twice the length and a
            # contains override that records checked and delegates.
            def len_mulitply_2(self):
                # [FUNCTION DOC] HELPER: Return twice the length.
                return len(self) * 2

            def __contains__(self, val):
                # [FUNCTION DOC] HELPER: Record that membership was checked and delegate to
                # the parent.
                # Ensure that overridden method is traced
                self.checked = True
                return super().__contains__(val)

        def fn(x, tup):
            # [FUNCTION DOC] HELPER: Choose cos or sin by membership of 3 and scale by the
            # doubled length.
            if 3 in tup:
                x = torch.cos(x)
            else:
                x = torch.sin(x)
            return x * tup.len_mulitply_2()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_tup = MyTuple([1, 2, 3])
        ref = fn(x, ref_tup)
        res_tup = MyTuple([1, 2, 3])
        res = opt_fn(x, res_tup)
        self.assertEqual(ref, res)
        self.assertTrue(ref_tup.checked)
        self.assertTrue(res_tup.checked)

    def test_udf_tuple_construction(self):
        # [FUNCTION DOC] Constructs a tuple subclass instance in compiled code and returns it with
        # x; compares with eager.
        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Empty tuple subclass.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Build a MyTuple, choose cos/sin by membership of 3 and return
            # x and tuple.
            tup = MyTuple([1, 2, 3])
            if 3 in tup:
                x = torch.cos(x)
            else:
                x = torch.sin(x)
            return x, tup

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_x, ref_tup = fn(x)
        res_x, res_tup = opt_fn(x)
        self.assertEqual(ref_x, res_x)
        self.assertEqual(ref_tup, res_tup)

    def test_udf_tuple_construction_custom_new(self):
        # [FUNCTION DOC] Same, with a tuple subclass whose new method ignores arguments and always
        # builds (1, 2, 3).
        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Tuple subclass whose new method always builds the contents (1,
            # 2, 3).
            def __new__(cls, *args, **kwargs):
                # [FUNCTION DOC] HELPER: Build the instance with fixed contents regardless
                # of arguments.
                return super().__new__(cls, [1, 2, 3])

        def fn(x):
            # [FUNCTION DOC] HELPER: Build a MyTuple with no args, choose cos/sin by membership
            # of 3.
            tup = MyTuple()
            if 3 in tup:
                x = torch.cos(x)
            else:
                x = torch.sin(x)
            return x, tup

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_x, ref_tup = fn(x)
        res_x, res_tup = opt_fn(x)
        self.assertEqual(ref_x, res_x)
        self.assertEqual(ref_tup, res_tup)

    def test_udf_tuple_equality(self):
        # [FUNCTION DOC] Checks inequality of two equal tuple subclass instances in compiled code.
        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Tuple subclass whose new method takes an items argument.
            def __new__(cls, items):
                # [FUNCTION DOC] HELPER: Build the instance from items.
                return super().__new__(cls, items)

        def fn(x, my_tuple_1, my_tuple_2):
            # [FUNCTION DOC] HELPER: Return x plus 1 if the tuples differ, else x minus 1.
            """Returns different values based on equality check"""
            if my_tuple_1 != my_tuple_2:
                return x + 1.0
            else:
                return x - 1.0

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(1)
        my_tuple_1 = MyTuple([1, -1])
        my_tuple_2 = MyTuple([1, -1])
        ref_x = fn(x, my_tuple_1, my_tuple_2)
        res_x = opt_fn(x, my_tuple_1, my_tuple_2)
        self.assertEqual(ref_x, res_x)

    def test_udf_namedtuple(self):
        # [FUNCTION DOC] Compiles addition of NamedTuple subclass instances with a custom add
        # method and compares the fields with eager.
        class MyTuple(NamedTuple):
            # [CLASS DOC] HELPER: NamedTuple with tensor fields a and b.
            a: torch.Tensor
            b: torch.Tensor

        class PairTensor(MyTuple):
            # [CLASS DOC] HELPER: MyTuple subclass with a custom new and an add method that
            # adds fieldwise.
            def __new__(cls, a, b):
                # [FUNCTION DOC] HELPER: Create an instance from a and b.
                return super().__new__(cls, a, b)

            def __add__(self, other):
                # [FUNCTION DOC] HELPER: Return a new PairTensor with fieldwise sums.
                return PairTensor(self.a + other.a, self.b + other.b)

        def fn(pair1, pair2):
            # [FUNCTION DOC] HELPER: Return pair1 plus pair2.
            return pair1 + pair2

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        pair1 = PairTensor(torch.randn(4), torch.randn(2, 8))
        pair2 = PairTensor(torch.randn(1), torch.randn(2, 1))
        ref = fn(pair1, pair2)
        res = opt_fn(pair1, pair2)
        self.assertEqual(ref.a, res.a)
        self.assertEqual(ref.b, res.b)

    def test_udf_tuple_reconstruction(self):
        # [FUNCTION DOC] Builds tuple subclass or plain tuple instances through the base tuple
        # new inside compiled code (with an attribute for the subclass) and checks type, values
        # and attribute.
        class MyTuple(tuple):  # noqa: SLOT001
            # [CLASS DOC] HELPER: Empty tuple subclass.
            pass

        def fn(x, klass):
            # [FUNCTION DOC] HELPER: Double x, build an instance of klass with it, and set attr on
            # subclass instances.
            x = x * 2
            sc_tuple = tuple.__new__(klass, [x])
            if isinstance(sc_tuple, MyTuple):
                sc_tuple.attr = 3
            return sc_tuple

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x, MyTuple)
        res = opt_fn(x, MyTuple)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, MyTuple))
        self.assertEqual(ref.attr, res.attr)

        ref = fn(x, tuple)
        res = opt_fn(x, tuple)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, tuple))

    def test_udf_list(self):
        # [FUNCTION DOC] Compiles a function using a list subclass with overridden contains and
        # getitem (the latter raising) and an extra method; checks results, lengths and that
        # contains ran.
        class MyList(list):
            # [CLASS DOC] HELPER: List subclass with a method returning twice the length, a
            # contains override that records checked and a getitem override that always
            # raises RuntimeError.
            def len_mulitply_2(self):
                # [FUNCTION DOC] HELPER: Return twice the length.
                return len(self) * 2

            def __contains__(self, val):
                # [FUNCTION DOC] HELPER: Record that membership was checked and delegate to
                # the parent.
                # Ensure that overridden method is traced
                self.checked = True
                return super().__contains__(val)

            def __getitem__(self, idx):
                # [FUNCTION DOC] HELPER: Always raise RuntimeError; used to verify reconstruction
                # never calls it.
                # Tests that the reconstruction logic does not call the
                # overridden __getitem__ method.
                raise RuntimeError("Should not be called")

        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Choose cos/sin by membership of 3, append 4 and scale by the
            # doubled length.
            if 3 in lst:
                x = torch.cos(x)
            else:
                x = torch.sin(x)
            lst.append(4)
            return x * lst.len_mulitply_2()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_lst = MyList([1, 2, 3])
        ref = fn(x, ref_lst)
        res_lst = MyList([1, 2, 3])
        res = opt_fn(x, res_lst)
        self.assertEqual(ref, res)
        self.assertEqual(len(ref_lst), len(res_lst))
        self.assertTrue(ref_lst.checked)
        self.assertTrue(res_lst.checked)

    def test_udf_list_slice(self):
        # [FUNCTION DOC] Compiles a function appending to a list subclass instance and slicing it;
        # checks results and lengths against eager.
        class MyList(list):
            # [CLASS DOC] HELPER: List subclass with a method returning twice the length.
            def len_mulitply_2(self):
                # [FUNCTION DOC] HELPER: Return twice the length.
                return len(self) * 2

        def fn(x, lst):
            # [FUNCTION DOC] HELPER: Append 4, then scale x by doubled length and the sum of
            # slice 1..3.
            lst.append(4)
            return x * lst.len_mulitply_2() * sum(lst[1:3])

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref_lst = MyList([1, 2, 3])
        ref = fn(x, ref_lst)
        res_lst = MyList([1, 2, 3])
        res = opt_fn(x, res_lst)
        self.assertEqual(ref, res)
        self.assertEqual(len(ref_lst), len(res_lst))

    def test_udf_list_reconstruction(self):
        # [FUNCTION DOC] Builds list subclass or plain list instances through the base list new
        # inside compiled code and checks type, values and an attribute set on the subclass.
        class MyList(list):
            # [CLASS DOC] HELPER: Empty list subclass.
            # def __new__(cls, *args, **kwargs):
            #     return super().__new__(cls, *args, **kwargs)
            pass

        def fn(x, klass):
            # [FUNCTION DOC] HELPER: Double x, build an empty instance of klass, append x, set attr
            # on subclass.
            x = x * 2
            sc_list = list.__new__(klass)
            sc_list.append(x)
            if isinstance(sc_list, MyList):
                sc_list.attr = 3
            return sc_list

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x, MyList)
        res = opt_fn(x, MyList)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, MyList))
        self.assertEqual(ref.attr, res.attr)

        ref = fn(x, list)
        res = opt_fn(x, list)
        self.assertEqual(ref, res)
        self.assertTrue(isinstance(res, list))

    def test_sys_recursionlimit(self):
        # [FUNCTION DOC] Compiles a function using the Python recursion limit constant; compares
        # to eager.
        def fn(x):
            # [FUNCTION DOC] HELPER: Return sin of x times the recursion limit.
            return x.sin() * sys.getrecursionlimit()

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_keyword(self):
        # [FUNCTION DOC] Compiles a function using the keyword module iskeyword on a string
        # argument; tested with None (a keyword) and dynamo (not).
        def fn(x, word):
            # [FUNCTION DOC] HELPER: Return sin of x if word is a keyword, else cos of x.
            if keyword.iskeyword(word):
                return torch.sin(x)
            return torch.cos(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        word = "None"
        self.assertEqual(fn(x, word), opt_fn(x, word))
        word = "dynamo"
        self.assertEqual(fn(x, word), opt_fn(x, word))

    def test_func_attrs(self):
        # [FUNCTION DOC] Compiles a function reading function attributes: missing attribute
        # handling, code and defaults; compares with eager.
        def f(x=4, y=2):
            # [FUNCTION DOC] HELPER: Function with two default arguments that does nothing.
            pass

        def fn(x):
            # [FUNCTION DOC] HELPER: Apply sin if the missing attribute raises, then scale x by the
            # number of defaults and the code argument count.
            try:
                f.dynamo + 1
            except AttributeError:
                x = torch.sin(x)

            code = f.__code__
            defaults = f.__defaults__
            return x * len(defaults) * code.co_argcount

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        self.assertEqual(fn(x), opt_fn(x))

    def test_functional_compile(self):
        # [FUNCTION DOC] Collects the public functions of torch.functional and checks torch.compile
        # can wrap each into a callable (functions are not called).
        def get_torch_functional_functions():
            # [FUNCTION DOC] HELPER: Return the set of functions listed in torch.functional's
            # public list.
            s = set()
            for name in torch.functional.__all__:
                method = getattr(torch.functional, name)
                s.add(method)
            return s

        functions = get_torch_functional_functions()
        self.assertTrue(len(functions) > 0)
        for func in functions:
            compiled_func = torch.compile(func, backend="eager")
            self.assertTrue(callable(compiled_func))

    def test_skip_function_call_very_weird_value(self):
        # [FUNCTION DOC] Builds a SkipFunctionVariable for an object whose qualname lookup raises
        # AttributeError and asserts calling it raises Unsupported.
        class weird:
            # [CLASS DOC] HELPER: Class whose attribute access for the qualname
            # raises AttributeError.
            def __getattribute__(self, name):
                # [FUNCTION DOC] HELPER: Raise AttributeError for the qualname attribute; None for
                # other names.
                if name == "__qualname__":
                    raise AttributeError("test")

        w = weird()
        a = SkipFunctionVariable(value=w)
        with self.assertRaises(Unsupported):
            a.call_function(None, [], {})

    def test_unsupported_msg_in_bind_args_error(self):
        # [FUNCTION DOC] Checks that a bind-args failure in a compiled function raises Unsupported
        # whose message includes the source line of the failing call.
        class BadClass:
            # [CLASS DOC] HELPER: Class whose init requires a cls argument.
            """Class that requires 'cls' argument."""

            def __init__(self, cls):
                # [FUNCTION DOC] HELPER: Store cls on the instance.
                self.cls = cls

        @contextlib.contextmanager
        def my_context():
            # [FUNCTION DOC] HELPER: Context manager that yields once.
            yield

        @torch.compile(backend="eager", fullgraph=True)
        def fn():
            # [FUNCTION DOC] HELPER: Inside a context, construct BadClass without its
            # required argument.
            with my_context():
                # Error: Missing required positional argument 'cls'
                obj = BadClass()  # This will raise TypeError
                return obj

        with self.assertRaisesRegex(Unsupported, "obj = BadClass()"):
            fn()

    def test_inspect_method_source(self):
        # [FUNCTION DOC] Checks that calling inspect.signature on a bound module method works in
        # compiled code; compares to eager.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a check method and forward both returning x times 2.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Initialize the module.
                super().__init__()

            def check(self, x):
                # [FUNCTION DOC] HELPER: Return x times 2.
                return x * 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x times 2.
                return x * 2

        mod = Mod()

        def fn(x):
            # [FUNCTION DOC] HELPER: Inspect the signature of the module's check method and call
            # the module.
            inspect.signature(mod.check).parameters.items()
            return mod(x)

        opt_fn = torch.compile(fn, backend="eager", fullgraph=True)
        x = torch.randn(4)
        ref = fn(x)
        res = opt_fn(x)
        self.assertEqual(ref, res)

    def test_property_class_transmute(self):
        # [FUNCTION DOC] Adds a property (built from a callable object) to a module's dynamically
        # created subclass by changing its class; compiled forward reads it and returns x plus 1.
        class PropertyGetter:
            # [CLASS DOC] HELPER: Callable object that ignores its argument and returns True.
            def __call__(self, obj):
                # [FUNCTION DOC] HELPER: Return True.
                return True

        p = property(PropertyGetter())

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward returns x plus 1 if the property p is
            # truthy, else raises.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x plus 1 if self.p is truthy, else
                # raise RuntimeError.
                if self.p:
                    return x + 1
                else:
                    raise RuntimeError("whoops")

        mod = Mod()
        mod.__class__ = type(mod.__class__.__name__, (mod.__class__,), {"p": p})

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(1)
        self.assertEqual(opt_mod(x), x + 1)

    def test_property_functools_partial(self):
        # [FUNCTION DOC] Uses a property whose getter is a functools partial with a keyword;
        # compiled module forward reads it and returns x plus 1.
        def p_getter(obj, *, delta: int):
            # [FUNCTION DOC] HELPER: Return whether the flag attribute (default 0) plus delta
            # is positive.
            # Use instance state + a bound constant
            return (getattr(obj, "flag", 0) + delta) > 0

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a flag attribute and a property p whose getter is
            # a partial.
            def __init__(self, flag: int):
                # [FUNCTION DOC] HELPER: Initialize the module and store flag.
                super().__init__()
                self.flag = flag

            # fget is a functools.partial object
            p = property(functools.partial(p_getter, delta=1))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return x plus 1 if property p is truthy, else
                # raise RuntimeError.
                if self.p:  # calls p_getter(self, delta=1)
                    return x + 1
                else:
                    raise RuntimeError("whoops")

        mod = Mod(flag=1)

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        x = torch.randn(1)
        self.assertEqual(opt_mod(x), x + 1)

    def test_full_with_tensor_fill_value(self):
        # [FUNCTION DOC] Regression test for torch.full with a tensor fill value: repeated compiled
        # calls with different values must use each value; also checks several dtypes and scalar
        # fill values.
        """Test that torch.full works correctly with dynamic tensor fill_value"""

        # Test with tensor fill_value (the bug case)
        def func_tensor(x):
            # [FUNCTION DOC] HELPER: Return a two-element float64 tensor filled with x.
            return torch.full((2,), x, dtype=torch.float64)

        func_compiled = torch.compile(func_tensor, backend="eager")

        # Test with different values
        x1 = torch.tensor(5.0, dtype=torch.float64)
        x2 = torch.tensor(10.0, dtype=torch.float64)

        result1 = func_compiled(x1)
        expected1 = torch.full((2,), x1, dtype=torch.float64)
        self.assertEqual(result1, expected1)

        # This is where the bug occurred - second call reused first value
        result2 = func_compiled(x2)
        expected2 = torch.full((2,), x2, dtype=torch.float64)
        self.assertEqual(result2, expected2)

        # Test with different dtypes
        for dtype in [torch.float32, torch.float64, torch.int32, torch.int64]:

            def func_typed(x):
                # [FUNCTION DOC] HELPER: Return a three-element tensor with the loop dtype filled
                # with x.
                return torch.full((3,), x, dtype=dtype)

            func_typed_compiled = torch.compile(func_typed, backend="eager")
            x_typed = torch.tensor(7, dtype=dtype)
            result = func_typed_compiled(x_typed)
            expected = torch.full((3,), x_typed, dtype=dtype)
            self.assertEqual(result, expected)

        # Test with non-tensor fill_value (scalar) to ensure we didn't break existing behavior
        def func_scalar(size):
            # [FUNCTION DOC] HELPER: Return a float32 tensor of the given size filled with 42.
            return torch.full((size,), 42.0, dtype=torch.float32)

        func_scalar_compiled = torch.compile(func_scalar, backend="eager")

        result_scalar = func_scalar_compiled(5)
        expected_scalar = torch.full((5,), 42.0, dtype=torch.float32)
        self.assertEqual(result_scalar, expected_scalar)

        # Test with different scalar values
        def func_scalar_param():
            # [FUNCTION DOC] HELPER: Return two float32 tensors filled with two
            # different constants.
            # Test multiple calls with different hardcoded scalar values
            a = torch.full((2,), 3.14, dtype=torch.float32)
            b = torch.full((2,), 2.71, dtype=torch.float32)
            return a, b

        func_scalar_param_compiled = torch.compile(func_scalar_param, backend="eager")
        result_a, result_b = func_scalar_param_compiled()

        self.assertEqual(result_a, torch.full((2,), 3.14, dtype=torch.float32))
        self.assertEqual(result_b, torch.full((2,), 2.71, dtype=torch.float32))

    def test_full_with_parameter_fill_value_raises(self):
        # [FUNCTION DOC] Checks torch.full with a Parameter fill value raises TypeError in eager
        # and Unsupported when compiled with fullgraph.
        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with a Parameter fill_value used in forward to fill
            # a tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Create a scalar Parameter fill_value.
                super().__init__()
                self.fill_value = torch.nn.Parameter(torch.tensor(2.5))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Return torch.full with the Parameter as fill value.
                return torch.full((x.shape[0], x.shape[1], 2), self.fill_value)

        mod = Mod()
        x = torch.randn(3, 4)

        with self.assertRaises(TypeError):
            mod(x)

        opt_mod = torch.compile(mod, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            opt_mod(x)

    def test_full_with_parameter_subclass_fill_value_raises(self):
        # [FUNCTION DOC] Same for an instance of a Parameter subclass: TypeError eager,
        # Unsupported compiled.
        class MyParam(torch.nn.Parameter):
            # [CLASS DOC] HELPER: Empty Parameter subclass.
            pass

        def func(x):
            # [FUNCTION DOC] HELPER: Return a two-element tensor filled with x.
            return torch.full((2,), x)

        x = MyParam(torch.tensor(2.5))

        with self.assertRaises(TypeError):
            func(x)

        func_compiled = torch.compile(func, backend="eager", fullgraph=True)
        with self.assertRaises(torch._dynamo.exc.Unsupported):
            func_compiled(x)

    def test_full_with_tensor_subclass_fill_value(self):
        # [FUNCTION DOC] Checks torch.full with a tensor subclass fill value gives the eager result
        # when compiled with fullgraph.
        class MyTensor(torch.Tensor):
            # [CLASS DOC] HELPER: Empty Tensor subclass.
            pass

        def func(x):
            # [FUNCTION DOC] HELPER: Return a two-element tensor filled with x.
            return torch.full((2,), x)

        x = torch.tensor(5.0).as_subclass(MyTensor)
        expected = func(x)

        func_compiled = torch.compile(func, backend="eager", fullgraph=True)
        result = func_compiled(x)

        self.assertEqual(result, expected)


instantiate_parametrized_tests(FunctionTests)
instantiate_parametrized_tests(DefaultsTests)

if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
