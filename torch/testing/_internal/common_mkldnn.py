# mypy: ignore-errors

# [MODULE DOC]
# Test helpers for exercising mkldnn (oneDNN) reduced-precision float32 modes on CPU.
# Provides availability checks for bf16/tf32 fast paths, context managers that switch the
# mkldnn matmul and conv fp32 precision settings (and the test's tolerance), and a decorator
# that runs a test once in strict mode and once per supported reduced-precision mode.
# No import-time side effects beyond imports.

import contextlib
import functools
import inspect

import torch


def bf32_is_not_fp32():
    # [FUNCTION DOC] HELPER: Reports whether float32 ops can actually run in bf16 on this machine.
    # Returns False if mkldnn is unavailable or the mkldnn bf16 support check fails;
    # otherwise True. No arguments, no side effects.
    if not torch.backends.mkldnn.is_available():
        return False
    if not torch.ops.mkldnn._is_mkldnn_bf16_supported():
        return False
    return True


def tf32_is_not_fp32():
    # [FUNCTION DOC] HELPER: Reports whether float32 ops can run in reduced "tf32" mode here.
    # Returns False if mkldnn is unavailable or the CPU lacks AMX fp16 support; otherwise
    # True. No arguments, no side effects.
    # KNOWN ISSUE: the gate checks AMX fp16 support, not anything specific to tf32 itself.
    if not torch.backends.mkldnn.is_available():
        return False
    if not torch.cpu._is_amx_fp16_supported():
        return False
    return True


@contextlib.contextmanager
def reduced_f32_off():
    # [FUNCTION DOC] HELPER: Context manager forcing strict ieee fp32 for mkldnn matmul and conv.
    # Saves the current matmul and conv fp32 precision settings, sets both to ieee, yields,
    # and restores the saved values on exit even if the body raises. Mutates global backend
    # settings for the duration of the block.
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "ieee"
        torch.backends.mkldnn.conv.fp32_precision = "ieee"
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision


@contextlib.contextmanager
def bf32_on(self, bf32_precision=1e-2):
    # [FUNCTION DOC] HELPER: Context manager enabling bf16 mode for fp32 mkldnn matmul and conv.
    # Args: self is a test case whose precision attribute (comparison tolerance) is loosened
    # to bf32_precision for the block. Saves matmul/conv precision settings and the test
    # tolerance, applies the new values, yields, and restores all of them on exit even on
    # error. Mutates global backend settings and the test instance.
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "bf16"
        torch.backends.mkldnn.conv.fp32_precision = "bf16"
        self.precision = bf32_precision
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision
        self.precision = old_precision


@contextlib.contextmanager
def tf32_on(self, tf32_precision=1e-5):
    # [FUNCTION DOC] HELPER: Context manager enabling tf32 mode for fp32 mkldnn matmul and conv.
    # Args: self is a test case whose precision attribute (comparison tolerance) is set to
    # tf32_precision for the block. Saves matmul/conv precision settings and the test
    # tolerance, applies the new values, yields, and restores all of them on exit even on
    # error. Mutates global backend settings and the test instance.
    old_matmul_precision = torch.backends.mkldnn.matmul.fp32_precision
    old_conv_precision = torch.backends.mkldnn.conv.fp32_precision
    old_precision = self.precision
    try:
        torch.backends.mkldnn.matmul.fp32_precision = "tf32"
        torch.backends.mkldnn.conv.fp32_precision = "tf32"
        self.precision = tf32_precision
        yield
    finally:
        torch.backends.mkldnn.matmul.fp32_precision = old_matmul_precision
        torch.backends.mkldnn.conv.fp32_precision = old_conv_precision
        self.precision = old_precision


# This is a wrapper that wraps a test to run this test three times, one with
# reduced_f32 OFF, the others with reduced_f32 ON (including bf32 ON and tf32
# ON). When running with reduced_f32 ON, it will use reduced precision (bf16/
# tf32) as specified by the argument.
def reduced_f32_on_and_off(bf32_precision=1e-2, tf32_precision=1e-5):
    # [FUNCTION DOC] Decorator factory that runs a test under strict and reduced fp32 modes.
    # Args: bf32_precision and tf32_precision are the test tolerances used while the bf16 or
    # tf32 mode is active. Returns a decorator; the decorated test is invoked with all
    # arguments converted to keyword arguments (so the test must accept a self parameter).
    # Side effects: temporarily changes global mkldnn precision settings via the helpers.
    def with_reduced_f32_disabled(self, function_call):
        # [FUNCTION DOC] HELPER: Calls function_call inside the strict ieee mode context.
        # The self argument is unused.
        with reduced_f32_off():
            function_call()

    def with_bf32_enabled(self, function_call):
        # [FUNCTION DOC] HELPER: Calls function_call with bf16 mode on and the test's
        # tolerance set to the captured bf32_precision.
        with bf32_on(self, bf32_precision):
            function_call()

    def with_tf32_enabled(self, function_call):
        # [FUNCTION DOC] HELPER: Calls function_call with tf32 mode on and the test's
        # tolerance set to the captured tf32_precision.
        with tf32_on(self, tf32_precision):
            function_call()

    def wrapper(f):
        # [FUNCTION DOC] HELPER: Wraps test function f, preserving its metadata. Records the
        # parameter names of f so positional arguments can be mapped to keywords. Returns the
        # wrapped function.
        params = inspect.signature(f).parameters
        arg_names = tuple(params.keys())

        @functools.wraps(f)
        def wrapped(*args, **kwargs):
            # [FUNCTION DOC] HELPER: Runs the test, possibly several times. Folds positional
            # args into keyword args by name, then decides whether reduced precision applies:
            # only if a device argument (when present) is cpu and a dtype argument (when
            # present) is float32. If bf16 or tf32 is supported, runs the test once in strict
            # mode, then once per supported reduced mode; otherwise runs it once unchanged.
            # Requires a self keyword in that case. Returns nothing; errors from the test
            # propagate.
            # KNOWN ISSUE: the fold uses non-strict zip, so extra positional args are silently
            # dropped, and a positional value overwrites any same-named keyword value.
            kwargs.update(zip(arg_names, args, strict=False))
            cond = True
            if "device" in kwargs:
                cond = cond and (torch.device(kwargs["device"]).type == "cpu")
            if "dtype" in kwargs:
                cond = cond and (kwargs["dtype"] == torch.float)
            bf32_cond = cond and bf32_is_not_fp32()
            tf32_cond = cond and tf32_is_not_fp32()
            if bf32_cond or tf32_cond:
                with_reduced_f32_disabled(kwargs["self"], lambda: f(**kwargs))
                if bf32_cond:
                    with_bf32_enabled(kwargs["self"], lambda: f(**kwargs))
                if tf32_cond:
                    with_tf32_enabled(kwargs["self"], lambda: f(**kwargs))
            else:
                f(**kwargs)

        return wrapped

    return wrapper
