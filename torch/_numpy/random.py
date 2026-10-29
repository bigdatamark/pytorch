# mypy: ignore-errors
# [MODULE DOC] Partial numpy.random API (seed, random_sample/sample/random, rand,
# randn, normal, uniform, randint, shuffle, choice) implemented on torch random
# generation, with no reproducibility guarantee relative to numpy. Each function is
# wrapped by a stream decorator which, when the dynamo config flag for using the numpy
# random stream is on, delegates to real numpy and wraps array results; otherwise uses
# torch. Default dtype is the default float dtype. No import-time side effects;
# defines the export list.

"""Wrapper to mimic (parts of) np.random API surface.

NumPy has strict guarantees on reproducibility etc; here we don't give any.

Q: default dtype is float64 in numpy

"""

from __future__ import annotations

import functools
from math import sqrt

import torch

from . import _dtypes_impl, _util
from ._normalizations import array_or_scalar, ArrayLike, normalizer


__all__ = [
    "seed",
    "random_sample",
    "sample",
    "random",
    "rand",
    "randn",
    "normal",
    "choice",
    "randint",
    "shuffle",
    "uniform",
]


def use_numpy_random():
    # [FUNCTION DOC] HELPER: Return the dynamo config flag saying whether to use numpy's
    # own random stream. Imports the config lazily to avoid reference cycles.
    # local import to avoid ref cycles
    import torch._dynamo.config as config

    return config.use_numpy_random_stream


def deco_stream(func):
    # [FUNCTION DOC] HELPER: Decorator choosing the random backend per call. Returns a
    # wrapper that runs the torch implementation unless the numpy-stream flag is set.
    @functools.wraps(func)
    def inner(*args, **kwds):
        # [FUNCTION DOC] HELPER: Wrapper. If numpy stream is disabled, calls the torch
        # implementation. Otherwise unwraps ndarray arguments (positional and keyword) to
        # numpy arrays, calls the numpy.random function of the same name, and wraps a numpy
        # array result into the library ndarray (scalars and None pass through). Imports numpy
        # lazily.
        if not use_numpy_random():
            return func(*args, **kwds)
        else:
            import numpy

            from ._ndarray import ndarray

            f = getattr(numpy.random, func.__name__)

            # numpy funcs accept numpy ndarrays, unwrap
            args = tuple(
                arg.tensor.numpy() if isinstance(arg, ndarray) else arg for arg in args
            )
            kwds = {
                key: val.tensor.numpy() if isinstance(val, ndarray) else val
                for key, val in kwds.items()
            }

            value = f(*args, **kwds)

            # `value` can be either numpy.ndarray or python scalar (or None)
            if isinstance(value, numpy.ndarray):
                value = ndarray(torch.as_tensor(value))

            return value

    return inner


@deco_stream
def seed(seed=None):
    # [FUNCTION DOC] Seed the global torch random generator when seed is not None;
    # otherwise does nothing. Side effect: mutates global RNG state. Returns None.
    if seed is not None:
        torch.random.manual_seed(seed)


@deco_stream
def random_sample(size=None):
    # [FUNCTION DOC] Uniform samples in [0, 1) of shape size (scalar when size is None) in
    # the default float dtype. Returns a scalar for empty size, else an ndarray.
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_()
    return array_or_scalar(values, return_scalar=size == ())


def rand(*size):
    # [FUNCTION DOC] Uniform [0, 1) samples with shape given by positional dimension
    # arguments; no arguments yields a scalar. Delegates to random_sample.
    if size == ():
        size = None
    return random_sample(size)


sample = random_sample
random = random_sample


@deco_stream
def uniform(low=0.0, high=1.0, size=None):
    # [FUNCTION DOC] Uniform samples in [low, high) of shape size (scalar when None) in
    # the default float dtype. Returns a scalar or ndarray.
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).uniform_(low, high)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def randn(*size):
    # [FUNCTION DOC] Standard normal samples with shape given by positional dimension
    # arguments in the default float dtype; scalar when no dimensions given.
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.randn(size, dtype=dtype)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def normal(loc=0.0, scale=1.0, size=None):
    # [FUNCTION DOC] Normal samples with mean loc and standard deviation scale and shape
    # size (scalar when None), default float dtype. Returns a scalar or ndarray.
    if size is None:
        size = ()
    dtype = _dtypes_impl.default_dtypes().float_dtype
    values = torch.empty(size, dtype=dtype).normal_(loc, scale)
    return array_or_scalar(values, return_scalar=size == ())


@deco_stream
def shuffle(x):
    # [FUNCTION DOC] Shuffle a tensor or ndarray in place along its first axis using a
    # random permutation. Raises NotImplementedError for any other type (lists
    # unsupported). Side effect: mutates the argument and the RNG state.
    # no @normalizer because we do not cast e.g. lists to tensors
    from ._ndarray import ndarray

    if isinstance(x, torch.Tensor):
        tensor = x
    elif isinstance(x, ndarray):
        tensor = x.tensor
    else:
        raise NotImplementedError("We do not random.shuffle lists in-place")

    perm = torch.randperm(tensor.shape[0])
    xp = tensor[perm]
    tensor.copy_(xp)


@deco_stream
def randint(low, high=None, size=None):
    # [FUNCTION DOC] Random integers in [low, high) (or [0, low) when high is None) with
    # shape size (int promoted to a one-element tuple; scalar when None). Returns an int
    # scalar or ndarray.
    if size is None:
        size = ()
    if not isinstance(size, (tuple, list)):
        size = (size,)
    if high is None:
        low, high = 0, low
    values = torch.randint(low, high, size=size)
    return array_or_scalar(values, int, return_scalar=size == ())


@deco_stream
@normalizer
def choice(a: ArrayLike, size=None, replace=True, p: ArrayLike | None = None):
    # [FUNCTION DOC] Sample from a 1D array (or arange of a when a has a single element)
    # with or without replacement, optionally with probabilities p (default uniform). size
    # may be None (one draw), an int, or a sequence, in which case the result is reshaped.
    # Raises ValueError if probabilities do not sum to 1 within a tolerance based on
    # machine epsilon. Returns the sampled elements.
    # KNOWN ISSUE: a one-element array is always treated as a count to take arange of, and
    #   a non-integer a is not rejected as numpy does; with no size the result is a
    #   length-one array rather than a scalar.
    # https://stackoverflow.com/questions/59461811/random-choice-with-pytorch
    if a.numel() == 1:
        a = torch.arange(a)

    # TODO: check a.dtype is integer -- cf np.random.choice(3.4) which raises

    # number of draws
    if size is None:
        num_el = 1
    elif _util.is_sequence(size):
        num_el = 1
        for el in size:
            num_el *= el
    else:
        num_el = size

    # prepare the probabilities
    if p is None:
        p = torch.ones_like(a) / a.shape[0]

    # cf https://github.com/numpy/numpy/blob/main/numpy/random/mtrand.pyx#L973
    atol = sqrt(torch.finfo(p.dtype).eps)
    if abs(p.sum() - 1.0) > atol:
        raise ValueError("probabilities do not sum to 1.")

    # actually sample
    indices = torch.multinomial(p, num_el, replacement=replace)

    if _util.is_sequence(size):
        indices = indices.reshape(size)

    samples = a[indices]

    return samples
