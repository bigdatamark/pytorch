# [MODULE DOC] Tooling for compile-only cache warming: a FakeTensorMode wrapper that returns sentinel scalars for
# data-dependent scalar extraction, a context manager that enables compile-only mode, and a live query function.
# In this mode kernels are compiled and exported to the persistent cache but not launched.
# No import-time side effects besides defining the sentinel constants.
# Copyright (c) 2026, Tri Dao.
"""Tooling for ``--compile-only`` cache warming.

Background
----------

Each ``@cute_op``-wrapped QuACK kernel registers a ``register_fake`` that, under
:data:`quack.cache.COMPILE_ONLY` ``= True`` + ``FakeTensorMode``, invokes the
underlying body of the op *without* actually launching a kernel. The body's
call to ``_compile_*`` exports a ``.o`` file to the persistent cache via
:func:`quack.cache.jit_cache`. Subsequent real runs load the ``.o`` in ~1 ms
instead of paying a ~500 ms CuTe-DSL compile.

This module exposes a thin context manager and a ``FakeTensorMode`` subclass
that fix the one rough edge of that workflow: ``aten._local_scalar_dense``
(which backs ``.item()``, ``.tolist()``, ``int(t)``, ``float(t)``) raises
``DataDependentOutputException`` under stock ``FakeTensorMode``. Many tests
shape their inputs with patterns like::

    seq_lens = torch.randint(8192 - 1024, 8192 + 1024, (num_groups,), device=...)
    total_m = seq_lens.sum().item()
    A = torch.randn(total_m, k, ...)
    gemm(A, B, cu_seqlens_m=cu_seqlens_m, ...)

…which bails at ``.item()`` and never reaches the kernel dispatch. The compile
signature for ``_compile_gemm`` is shape-key-independent for the batch dim
(it always uses ``cute.sym_int()``), so returning *any* small concrete int
from ``.item()`` lets the test setup proceed far enough to fire the compile.

Usage
-----

Driver-script style (warm the cache from a CLI tool)::

    from .cache import compile_only_mode
    from my_lib import my_kernel

    with compile_only_mode():
        for shape in shapes_to_warm:
            A, B = make_inputs(shape)        # FakeTensors
            my_kernel(A, B)                  # compiles, doesn't launch

Pytest style — use the reusable plugin in your project's ``conftest.py``::

    pytest_plugins = ["quack.testing.pytest_plugin"]

…then run ``pytest --compile-only`` to populate the cache.

Low-level style (only when neither of the above fits) — push the depth
counter manually::

    import quack.cache
    from .cache import CompileOnlyFakeTensorMode

    token = quack.cache._COMPILE_ONLY_DEPTH.set(
        quack.cache._COMPILE_ONLY_DEPTH.get() + 1
    )
    fake_mode = CompileOnlyFakeTensorMode()
    fake_mode.__enter__()
    try:
        # ... your code here ...
    finally:
        fake_mode.__exit__(None, None, None)
        quack.cache._COMPILE_ONLY_DEPTH.reset(token)

Direct assignment to ``quack.cache.COMPILE_ONLY`` is **forbidden** — the
module's custom ``__setattr__`` raises :class:`AttributeError` to prevent
the old leak-on-reset failure mode. See :mod:`quack.cache` for background.
"""

from __future__ import annotations

import contextlib
from typing import Iterator

import torch
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.utils._python_dispatch import TorchDispatchMode


# Sentinel values returned from intercepted ``.item()`` / ``.tolist()`` /
# ``int(t)`` / ``float(t)`` calls. They only need to be valid Python scalars
# of the right kind so test setup that uses them as shape arguments,
# divisors, or stddev factors keeps working. The compile-key signatures of
# ``@cute_op``-decorated kernels do not depend on these values — the batch
# dim is always a ``cute.sym_int()`` in the compile key.
#
# Chose ``1`` (not e.g. ``256``) deliberately: any test code that turns out
# to depend on the concrete returned value should surface as a test-time
# failure under ``--compile-only``, not silently pass with a quietly-aligned
# magic number. The remaining usage patterns we surveyed (``total_k /
# num_groups`` as a stddev scale, ``total_m * 2`` as a buffer size) all
# behave fine with ``1``.
_INT_SENTINEL = 1
_FLOAT_SENTINEL = 1.0


class _LocalScalarDenseSentinel(TorchDispatchMode):
    # [CLASS DOC] HELPER: Dispatch mode that intercepts the local-scalar-dense op (behind item, tolist, int and float conversions)
    # and returns a fixed sentinel (float sentinel for floating dtypes, integer sentinel otherwise).
    # All other ops are forwarded to the next mode in the stack.
    """Internal: intercept ``aten._local_scalar_dense`` and return a sentinel.

    Implemented as a separate :class:`TorchDispatchMode` rather than as a
    ``FakeTensorMode`` subclass. Two benefits:

    * Uses the documented ``__torch_dispatch__`` protocol — stable across
      PyTorch versions — instead of overriding ``FakeTensorMode.dispatch``
      (an internal method whose signature has shifted in past releases).
    * Cleanly composes with whatever ``FakeTensorMode`` configuration the
      user provides (dynamic shapes, ``allow_non_fake_inputs``, etc.); we
      no longer have to mirror ``FakeTensorMode``'s public surface in a
      subclass.

    Any op other than ``_local_scalar_dense`` is delegated to the next mode
    in the dispatch stack via ``func(*args, **kwargs)``, which from inside a
    ``FakeTensorMode`` context routes naturally to fake dispatch.
    """

    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: If func is the local-scalar-dense op return the sentinel matching the first argument's dtype kind;
        # otherwise call func with the given args and kwargs (kwargs defaults to empty) and return its result.
        kwargs = kwargs or {}
        if func is torch.ops.aten._local_scalar_dense.default:
            t = args[0]
            return _FLOAT_SENTINEL if t.dtype.is_floating_point else _INT_SENTINEL
        return func(*args, **kwargs)


class CompileOnlyFakeTensorMode:
    # [CLASS DOC] Context manager combining a stock FakeTensorMode (constructor kwargs are forwarded to it) with the
    # sentinel dispatch mode, so test setup using data-dependent scalars can proceed to kernel dispatch under fake tensors.
    # State: the inner fake mode and the sentinel override. Entering returns self; exiting unwinds in reverse order.
    """Context manager: stock :class:`FakeTensorMode` + sentinel override.

    The override (:class:`_LocalScalarDenseSentinel`) returns a sentinel value
    from ``aten._local_scalar_dense`` (the op behind ``.item()``,
    ``.tolist()``, ``int(t)``, ``float(t)``) so data-dependent test setup
    proceeds far enough to dispatch the kernel and populate the ``.o`` cache.

    Compile-key signatures don't depend on the sentinel value.

    Forwards ``**fake_kwargs`` to the inner :class:`FakeTensorMode`
    constructor, so callers that need e.g. ``shape_env=ShapeEnv()`` or
    ``allow_non_fake_inputs=True`` can pass them through.
    """

    def __init__(self, **fake_kwargs):
        # [FUNCTION DOC] Create the inner FakeTensorMode from the forwarded keyword arguments and the sentinel override mode.
        self._fake_mode = FakeTensorMode(**fake_kwargs)
        self._override = _LocalScalarDenseSentinel()

    def __enter__(self):
        # [FUNCTION DOC] Enter the fake tensor mode first, then the sentinel override; return self.
        self._fake_mode.__enter__()
        self._override.__enter__()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] Exit the sentinel override first, then the fake mode (reverse order), passing the exception info to both.
        # Exit inner override first, then outer FakeTensorMode (LIFO).
        self._override.__exit__(exc_type, exc_val, exc_tb)
        self._fake_mode.__exit__(exc_type, exc_val, exc_tb)


@contextlib.contextmanager
def compile_only_mode() -> Iterator[CompileOnlyFakeTensorMode]:
    # [FUNCTION DOC] Context manager enabling compile-only mode. Increments the context-variable depth counter, enters a new
    # CompileOnlyFakeTensorMode and yields it; on exit (including on exceptions) leaves the fake mode and resets the counter
    # to its previous value, so nesting is safe. While active, jit_cache returns a no-op kernel instead of the real one.
    # KNOWN ISSUE: the docstring refers to a helper in the cache package for the no-op launch, but the no-op kernel is defined in the jit submodule.
    """Enter the QuACK compile-only cache-warming context.

    Inside the ``with`` block:

    * :func:`is_compile_only` returns ``True`` (the depth counter is pushed),
      so the ``@cute_op`` ``register_fake`` hooks fire the underlying kernel
      body (which compiles and exports ``.o``) and the launch path no-ops via
      :func:`quack.cache._noop_kernel`.
    * A :class:`CompileOnlyFakeTensorMode` is active, so all tensor
      allocations are fake (no GPU memory) and ``.item()`` &c. return
      sentinel values instead of raising.

    Yields the active :class:`CompileOnlyFakeTensorMode` so callers can
    inspect or augment it if needed.

    Nested calls compose correctly: each ``with`` pushes the depth counter
    one level deeper and pops back to the prior depth on exit, so callers
    can compose the context inside outer compile-only regions without ever
    leaking ``False`` back to the outer scope. The token-based push/pop is
    structurally leak-free: even an exception inside the block pops to the
    exact prior depth, never to ``0``.

    Example::

        from .cache import compile_only_mode

        with compile_only_mode():
            for cfg in parametrize_grid():
                run_kernel(*build_inputs(cfg))   # compiles, doesn't launch
    """
    import torch._vendor.quack.cache as _state

    token = _state._COMPILE_ONLY_DEPTH.set(_state._COMPILE_ONLY_DEPTH.get() + 1)
    fake_mode = CompileOnlyFakeTensorMode()
    fake_mode.__enter__()
    try:
        yield fake_mode
    finally:
        fake_mode.__exit__(None, None, None)
        _state._COMPILE_ONLY_DEPTH.reset(token)


def is_compile_only() -> bool:
    # [FUNCTION DOC] Return true if compile-only mode is currently active (depth counter above zero). Reads live state on every call.
    """Return ``True`` if the QuACK compile-only mode is currently active.

    Always returns the live stack depth as a bool; no capture-at-import
    hazard at any callsite. Prefer this over ``quack.cache.COMPILE_ONLY``
    in new code:

    * ``quack.cache.COMPILE_ONLY`` attribute reads work via the legacy
      ``__getattr__`` proxy and *are* live, but rely on PEP 562 module
      hooks that don't survive ``from quack.cache import COMPILE_ONLY``
      (that captures the value once at import time).
    * :func:`is_compile_only` is a regular function call, so it's safe to
      use anywhere — in decorator arguments, at module top level, inside
      lazy imports, etc.
    """
    import torch._vendor.quack.cache as _state

    return _state._COMPILE_ONLY_DEPTH.get() > 0
