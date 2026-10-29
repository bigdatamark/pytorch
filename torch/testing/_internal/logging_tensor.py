# mypy: ignore-errors

# [MODULE DOC] Test utilities that log every operator seen at the dispatcher level.
# Provides a wrapper tensor subclass (LoggingTensor), a dispatch mode (LoggingTensorMode),
# a logging handler that formats calls with short tensor ids, and capture_logs helpers
# that collect the formatted lines (optionally with tracebacks) into a list.
# Import-time side effect: creates the logger named LoggingTensor.

import torch
from torch.utils._pytree import tree_map
from collections.abc import Iterator
import logging
import contextlib
import itertools
from torch.utils._dtype_abbrs import dtype_abbrs as _dtype_abbrs
from torch.utils._python_dispatch import TorchDispatchMode
from torch.utils.weak import WeakTensorKeyDictionary
import functools
from torch._C._profiler import gather_traceback, symbolize_tracebacks

logger = logging.getLogger("LoggingTensor")

# How the chain of calls works for LoggingTensor:
# 1. Call torch.sin
# 2. Attempt __torch_function__. In LoggingTensor torch function is disabled so we bypass it entirely
# 3. Enter dispatcher, wind your way through Autograd
# 4. Hit Python dispatch key, call __torch_dispatch__

# This Tensor can work with autograd in two ways:
#  - The wrapped Tensor does not require gradients. In that case, the LoggingTensor
#    can require gradients if the user asks for it as a constructor kwarg.
#  - The wrapped Tensor can require gradients. In that case autograd will be tracked
#    for the wrapped Tensor and the LoggingTensor itself cannot require gradients.
# WARNING: We allow these two possibilities for testing purposes. You should NEVER use both in a single
# test or you might get surprising behavior.

# TODO: TensorBase should work
class LoggingTensor(torch.Tensor):
    # [CLASS DOC] Wrapper tensor subclass holding a real tensor in the elem slot; it owns no
    # memory itself but mirrors size, strides, dtype, layout and device of the wrapped tensor.
    # Every operator reaching dispatch is run on the unwrapped tensors, results are rewrapped,
    # and the call is logged. The class attribute context is a context-manager factory entered
    # around the call (subclasses may override it).
    elem: torch.Tensor

    __slots__ = ['elem']

    context = contextlib.nullcontext

    @staticmethod
    def __new__(cls, elem, *args, **kwargs):
        # [FUNCTION DOC] Builds a wrapper subclass instance mirroring the metadata of elem.
        # The requires_grad keyword (default false) is read from kwargs; extra args are ignored.
        # When the wrapper requires grad, the stored element is a detached alias of elem,
        # otherwise elem itself is stored. Returns the new wrapper tensor.
        # The wrapping tensor (LoggingTensor) shouldn't hold any
        # memory for the class in question, but it should still
        # advertise the same device as before
        r = torch.Tensor._make_wrapper_subclass(
            cls, elem.size(),
            strides=elem.stride(), storage_offset=elem.storage_offset(),
            # TODO: clone storage aliasing
            dtype=elem.dtype, layout=elem.layout,
            device=elem.device, requires_grad=kwargs.get("requires_grad", False)
        )
        # ...the real tensor is held as an element on the tensor.
        r.elem = elem.detach() if r.requires_grad else elem
        return r

    def __repr__(self):
        # [FUNCTION DOC] Returns the standard tensor repr but showing the wrapped element's
        # contents.
        return super().__repr__(tensor_contents=f"{self.elem}")

    @classmethod
    def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Dispatch hook: unwraps instances of this class in args and kwargs, runs
        # func inside the class context, wraps every tensor in the result back into this class,
        # then emits one info record on the LoggingTensor logger carrying the original args,
        # kwargs and result. Returns the wrapped result. Side effect: logging.
        # KNOWN ISSUE: kwargs defaults to None but is passed to tree_map and then double-star
        # unpacked, which would fail if it were really None (callers normally pass a mapping).
        def unwrap(e):
            # [FUNCTION DOC] HELPER: returns the wrapped element for an instance of this class,
            # else the input unchanged.
            return e.elem if isinstance(e, cls) else e

        def wrap(e):
            # [FUNCTION DOC] HELPER: wraps a plain tensor in this class; other values pass through.
            return cls(e) if isinstance(e, torch.Tensor) else e

        with cls.context():
            rs = tree_map(wrap, func(*tree_map(unwrap, args), **tree_map(unwrap, kwargs)))
        logging.getLogger("LoggingTensor").info(
            f"{func.__module__}.{func.__name__}",  # noqa: G004
            extra={"lt_args": args, "lt_kwargs": kwargs, "lt_rs": rs},
        )
        return rs

class LoggingTensorMode(TorchDispatchMode):
    # [CLASS DOC] Dispatch mode that runs each operator unchanged and logs the call (args,
    # kwargs, result) on the LoggingTensor logger. Holds no state of its own.
    def __torch_dispatch__(self, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] Runs func with args and kwargs (empty mapping if none), logs one info
        # record with the call details and result, and returns the result unchanged.
        if kwargs is None:
            kwargs = {}
        rs = func(*args, **kwargs)
        logging.getLogger("LoggingTensor").info(
            f"{func.__module__}.{func.__name__}",  # noqa: G004
            extra={"lt_args": args, "lt_kwargs": kwargs, "lt_rs": rs},
        )
        return rs

class LoggingTensorReentrant(LoggingTensor):
    # [CLASS DOC] LoggingTensor variant whose dispatch context enables reentrant dispatch, so
    # operators run inside dispatch can themselves hit the Python dispatch key.
    context = torch.overrides.enable_reentrant_dispatch

# https://stackoverflow.com/questions/36408496/python-logging-handler-to-append-to-list
class LoggingTensorHandler(logging.Handler):
    # [CLASS DOC] Logging handler that formats LoggingTensor records as text lines and appends
    # them to a caller-provided list. Tensors are shown as numbered short ids assigned on first
    # sight (memo keyed weakly by tensor, counter next_id); optional dtype/shape suffix on
    # results; optionally collects each record's traceback into a second list.
    def __init__(
            self, log_list: list[str], use_shortid_for_all_tensors: bool,
            with_type: bool, tracebacks_list: list | None) -> None:
        # [FUNCTION DOC] Stores the output list, the flag choosing whether all tensors or only
        # LoggingTensor instances get short ids, the type-suffix flag and the optional
        # traceback list (none disables traceback collection); starts id numbering at zero.
        logging.Handler.__init__(self)
        self.log_list = log_list
        self.use_shortid_for_all_tensors = use_shortid_for_all_tensors
        self.tracebacks_list = tracebacks_list
        self.memo = WeakTensorKeyDictionary()
        self.next_id = 0
        self.with_type = with_type

    def _shortid(self, t: torch.Tensor) -> int:
        # [FUNCTION DOC] HELPER: returns the stable small integer id for t, assigning the next
        # unused id the first time the tensor is seen.
        if t not in self.memo:
            self.memo[t] = self.next_id
            self.next_id += 1
        return self.memo[t]

    def _fmt(self, a: object, with_type: bool = False) -> str:
        # [FUNCTION DOC] HELPER: formats one value. Tensors (all tensors, or only LoggingTensor
        # instances depending on the handler flag) become a dollar sign plus short id, with an
        # abbreviated dtype and shape appended when both with_type and the handler flag are set.
        # Any other value is formatted with its repr.
        cond_cls = torch.Tensor if self.use_shortid_for_all_tensors else LoggingTensor
        if isinstance(a, cond_cls):
            maybe_type = ""
            if with_type and self.with_type:
                maybe_type = f": {_dtype_abbrs[a.dtype]}[{', '.join(map(str, a.shape))}]"
            x = f"${self._shortid(a)}{maybe_type}"
            return x
        else:
            return repr(a)

    def emit(self, record):
        # [FUNCTION DOC] Formats a record's stored args, kwargs and result into one line of the
        # form results equals op name applied to args, and appends it to the output list. If
        # traceback collection is on, also appends the traceback attached to the record (which
        # requires the GatherTraceback filter to have run, else an attribute error is raised).
        fmt_args = ", ".join(
            itertools.chain(
                (str(tree_map(self._fmt, a)) for a in record.lt_args),
                (f"{k}={str(tree_map(self._fmt, v))}" for k, v in record.lt_kwargs.items()),
            )
        )
        fmt_rets = tree_map(functools.partial(self._fmt, with_type=True), record.lt_rs)
        self.log_list.append(f'{fmt_rets} = {record.msg}({fmt_args})')
        if self.tracebacks_list is not None:
            self.tracebacks_list.append(record.traceback)

def log_input(name: str, var: object) -> None:
    # [FUNCTION DOC] Emits an info record named input on the LoggingTensor logger that marks
    # var as a named input, so captured logs show where an input tensor was introduced.
    logger.info("input", extra={"lt_args": (name,), "lt_kwargs": {}, "lt_rs": var})

class GatherTraceback(logging.Filter):
    # [CLASS DOC] Logging filter that attaches a freshly gathered traceback (python, script
    # and/or C++ frames per its flags) to every record and always lets the record through.
    def __init__(self, python=True, script=True, cpp=False):
        # [FUNCTION DOC] Records which frame kinds to gather. Does not call the base initializer.
        self.python = python
        self.script = script
        self.cpp = cpp

    def filter(self, record):
        # [FUNCTION DOC] Sets the traceback attribute on the record from a fresh gather and
        # returns true so the record is never dropped.
        record.traceback = gather_traceback(python=self.python, script=self.script, cpp=self.cpp)
        return True

@contextlib.contextmanager
def capture_logs(is_mode=False, python_tb=False, script_tb=False, cpp_tb=False) -> Iterator[list[str]]:
    # [FUNCTION DOC] Context manager that captures LoggingTensor log lines. Yields a list of
    # strings, or a pair of that list and a traceback list when any traceback flag is set.
    # is_mode makes all tensors, not only LoggingTensor, print as short ids. On exit the
    # tracebacks are symbolized in place and the handler is removed.
    # Side effects: sets the logger level to info and disables propagation, and adds a
    # traceback filter when requested.
    # KNOWN ISSUE: propagate is never restored and the GatherTraceback filter is never removed,
    # so both leak past the context; symbolization also runs even without tracebacks.
    collect_traceback = python_tb or script_tb or cpp_tb
    log_list: list[str] = []
    tracebacks_list: list[str] = []
    handler = LoggingTensorHandler(
        log_list,
        with_type=True,
        use_shortid_for_all_tensors=is_mode,
        tracebacks_list=tracebacks_list if collect_traceback else None
    )
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    if collect_traceback:
        logger.addFilter(GatherTraceback(python=python_tb, script=script_tb, cpp=cpp_tb))
    try:
        if collect_traceback:
            yield log_list, tracebacks_list
        else:
            yield log_list
    finally:
        symbolized_tracebacks = symbolize_tracebacks(tracebacks_list)
        tracebacks_list.clear()
        tracebacks_list.extend(symbolized_tracebacks)
        logger.removeHandler(handler)

@contextlib.contextmanager
def capture_logs_with_logging_tensor_mode(python_tb=False, script_tb=False, cpp_tb=False):
    # [FUNCTION DOC] Context manager combining LoggingTensorMode with capture_logs in mode
    # style (all tensors shown as short ids); yields what capture_logs yields.
    with LoggingTensorMode(), capture_logs(True, python_tb, script_tb, cpp_tb) as logs:
        yield logs
