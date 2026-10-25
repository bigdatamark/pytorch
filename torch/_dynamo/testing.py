# [MODULE DOC]
# Helper utilities used across Dynamo/Inductor tests: compile counters, graph-recording
# backends, graph/string normalizers, result collection and loss reduction helpers,
# strided random tensors, python-version skip decorators, expected-failure markers,
# RNG reset, test class generation with patches, and bytecode debug hooks.
# Import-time side effects: optionally imports numpy (np is None if missing) and
# defines module logger and a few constants.
"""Testing utilities and infrastructure for Dynamo.

This module provides a comprehensive set of testing utilities including:
- Test result collection and validation
- Graph manipulation and comparison tools
- Test case management and execution helpers
- Specialized test decorators for different Python versions and features
- RNG state management
- Compilation counting and monitoring
- Debug utilities for bytecode transformation

The utilities in this module are used across Dynamo's test suite to ensure
consistent testing patterns and proper test isolation.
"""

import contextlib
import dis
import functools
import logging
import os.path
import random
import re
import sys
import types
import unittest
from collections.abc import Callable, Generator, Sequence
from typing import Any, overload, TypeVar
from typing_extensions import ParamSpec
from unittest.mock import patch

import torch
from torch import fx
from torch._dynamo.backends.debugging import aot_eager
from torch._dynamo.output_graph import OutputGraph

from . import config, eval_frame, optimize_assert, reset
from .bytecode_transformation import (
    create_instruction,
    debug_checks,
    is_generator,
    transform_code_object,
)
from .guards import CheckFunctionManager, CompileId, GuardedCode
from .types import ConvertFrameReturn, DynamoFrameType, wrap_guarded_code
from .utils import CompileCounterInt, same


np: types.ModuleType | None = None
try:
    import numpy as np
except ModuleNotFoundError:
    np = None


unsupported = eval_frame.unsupported
three = 3

log = logging.getLogger(__name__)

_P = ParamSpec("_P")


def clone_me(x: torch.Tensor | None) -> torch.Tensor | None:
    # [FUNCTION DOC] Returns None for None; otherwise a detached clone of x with the
    # same requires_grad flag.
    if x is None:
        return None
    return x.detach().clone().requires_grad_(x.requires_grad)


def remove_optimized_module_prefix(name: str) -> str:
    # [FUNCTION DOC] Strips a leading "_orig_mod." prefix (added by OptimizedModule) from name.
    return re.sub(r"^_orig_mod[.]", "", name)


def extract_graph_and_tracker(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Compiles fn with fullgraph=True using a capturing backend, calls it
    # with the given args, and returns the FX graph and the region tracker of the
    # current translator's output graph. Executes fn; raises if compilation fails.
    from torch._dynamo.symbolic_convert import InstructionTranslator

    gm = None
    region_tracker = None

    def extract_graph_backend(_gm, *args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Backend that stores the graph module and region tracker
        # in the enclosing scope and returns the graph module unchanged.
        nonlocal gm
        nonlocal region_tracker
        gm = _gm
        region_tracker = InstructionTranslator.current_tx().output.region_tracker
        return _gm

    torch.compile(backend=extract_graph_backend, fullgraph=True)(fn)(*args, **kwargs)
    return gm.graph, region_tracker  # type: ignore[union-attr]


def extract_graph(fn, *args, **kwargs):  # type: ignore[no-untyped-def]
    # [FUNCTION DOC] Compiles and runs fn with AotEagerAndRecordGraphs; returns the
    # result plus lists of dynamo graphs, forward graphs and backward graphs.
    backend = AotEagerAndRecordGraphs()
    result = torch.compile(backend=backend)(fn)(*args, **kwargs)
    return result, backend.graphs, backend.fw_graphs, backend.bw_graphs


def collect_results(
    model: torch.nn.Module, prediction: Any, loss: Any, example_inputs: Any
) -> list[Any]:
    # [FUNCTION DOC] Gathers comparable outputs of a training step into a list:
    # prediction, loss, a mapping of parameter gradients (zeros when grad is None), a
    # mapping of parameters, a mapping of buffers (names stripped of the optimized
    # module prefix), then the grads of tensor example inputs (including those in
    # tuples/lists).
    results = []
    results.append(prediction)
    results.append(loss)
    # if isinstance(loss, torch.Tensor) and loss.item() > 1:
    #     log.warning(
    #         f"High loss value alert - {loss:.2f}. Can result in unstable gradients."
    #     )

    grads = {}
    params = {}
    for name, param in model.named_parameters():
        if isinstance(model, eval_frame.OptimizedModule):
            name = remove_optimized_module_prefix(name)
        param_copy = param
        grad = param.grad
        # Treat None and zero grad as same
        if param.grad is None:
            grad = torch.zeros_like(param)
        grads[name + ".grad"] = grad
        params[name] = param_copy
    results.append(grads)
    results.append(params)
    buffers = {}
    for name, buffer in model.named_buffers():
        if isinstance(model, eval_frame.OptimizedModule):
            name = remove_optimized_module_prefix(name)
        buffers[name] = buffer
    results.append(buffers)
    for example in example_inputs:
        if isinstance(example, (tuple, list)):
            results.extend(inp.grad for inp in example if isinstance(inp, torch.Tensor))
        else:
            if isinstance(example, torch.Tensor):
                results.append(example.grad)
    return results


def requires_bwd_pass(out: Any) -> bool:
    # [FUNCTION DOC] Returns whether out needs a backward pass: a tensor's requires_grad,
    # any() over list/tuple elements, False for None or int. Raises NotImplementedError
    # for other types.
    if isinstance(out, torch.Tensor):
        return out.requires_grad
    elif isinstance(out, (list, tuple)):
        return any(requires_bwd_pass(x) for x in out)
    elif out is None:
        return False
    elif isinstance(out, int):
        return False
    raise NotImplementedError("Don't know how to reduce", type(out))


@overload
def reduce_to_scalar_loss(out: torch.Tensor) -> torch.Tensor: ...


@overload
def reduce_to_scalar_loss(
    out: list[Any] | tuple[Any, ...] | dict[Any, Any],
) -> float: ...


def reduce_to_scalar_loss(out: Any) -> torch.Tensor | float:
    # [FUNCTION DOC] Reduces a model output to a scalar: tensors give sum divided by
    # numel; lists/tuples and dicts give the mean of recursively reduced elements/values;
    # known HF output types reduce their logits; SquashedNormal reduces its mean's sum.
    # Raises NotImplementedError for unsupported types.
    """Reduce the output of a model to get scalar loss"""
    if isinstance(out, torch.Tensor):
        # Mean does not work on integer tensors
        return out.sum() / out.numel()
    elif isinstance(out, (list, tuple)):
        return sum(reduce_to_scalar_loss(x) for x in out) / len(out)
    elif type(out).__name__ in (
        "MaskedLMOutput",
        "Seq2SeqLMOutput",
        "CausalLMOutputWithCrossAttentions",
    ):
        return reduce_to_scalar_loss(out.logits)
    elif type(out).__name__ == "SquashedNormal":
        return out.mean.sum()
    elif isinstance(out, dict):
        return sum(reduce_to_scalar_loss(value) for value in out.values()) / len(
            out.keys()
        )
    raise NotImplementedError("Don't know how to reduce", type(out))


def debug_dir() -> str:
    # [FUNCTION DOC] Returns the path of a "debug" directory next to this package's
    # parent, creating it on disk if missing (I/O side effect).
    path = os.path.join(os.path.dirname(__file__), "../debug")
    if not os.path.exists(path):
        os.mkdir(path)
    return path


def debug_dump(name: str, code: types.CodeType, extra: str = "") -> None:
    # [FUNCTION DOC] HELPER: Writes the code object's info, disassembly and extra text to
    # a file called name inside debug_dir().
    with open(os.path.join(debug_dir(), name), "w") as fd:
        fd.write(
            f"{dis.Bytecode(code).info()}\n\n{dis.Bytecode(code).dis()}\n\n{extra}\n"
        )


def debug_insert_nops(
    frame: DynamoFrameType, cache_size: int, hooks: Any, _: Any, *, skip: int = 0
) -> ConvertFrameReturn:
    # [FUNCTION DOC] HELPER: Frame-conversion callback for debugging jump updates. For
    # non-generator frames it transforms the code object by inserting two NOP
    # instructions at the start, builds a dummy OutputGraph and guard manager, and
    # returns the wrapped guarded code. Generators yield an empty ConvertFrameReturn.
    # Runs under dynamo_timed and the metrics context. Args cache_size, hooks, skip are unused.
    """used to debug jump updates"""

    def insert_nops(instructions: list[Any], code_options: Any) -> None:
        # [FUNCTION DOC] HELPER: Inserts two NOP instructions at the beginning of the list.
        instructions.insert(0, create_instruction("NOP"))
        instructions.insert(0, create_instruction("NOP"))

    metrics_context = torch._dynamo.utils.get_metrics_context()
    with torch._dynamo.utils.dynamo_timed("debug_insert_nops"), metrics_context:
        if is_generator(frame.f_code):
            return ConvertFrameReturn()

        debug_checks(frame.f_code)
        code, _ = transform_code_object(frame.f_code, insert_nops)
        graph = OutputGraph(
            # The nop-insertion path does not inspect code object attributes.
            code_options={},  # type: ignore[arg-type]
            compiler_fn=None,
            root_tx=None,  # type: ignore[arg-type]
            export=False,
            export_constraints=[],
            frame_state={"_id": 0},
            # TODO: shouldn't this be f_locals/f_globals from frame?
            local_scope=locals(),
            global_scope=globals(),
            f_code=frame.f_code,
            torch_function_mode_stack=[],
            package=None,
        )

        return wrap_guarded_code(
            GuardedCode(
                code,
                CheckFunctionManager(frame.f_code, graph).guard_manager,  # type: ignore[arg-type]
                CompileId(frame_id=0, frame_compile_id=0),
            )
        )


class CompileCounter:
    # [CLASS DOC] Eager-like backend that counts compiled graphs (frame_count) and call
    # nodes (op_count) across invocations. Use clear() to reset.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes counters via clear().
        self.frame_count: int | CompileCounterInt = 0
        self.clear()

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Backend entry: increments frame_count, adds the number of nodes
        # whose op contains "call" to op_count, and returns gm.forward.
        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        return gm.forward

    def clear(self) -> None:
        # [FUNCTION DOC] Resets frame_count (to a CompileCounterInt when
        # config.debug_disable_compile_counter is set, else 0) and op_count to 0.
        if config.debug_disable_compile_counter:
            self.frame_count = CompileCounterInt(0)
        else:
            self.frame_count = 0
        self.op_count = 0


class CompileCounterWithBackend:
    # [CLASS DOC] Like CompileCounter but forwards each graph to a named backend and
    # records the graphs (graphs list) alongside frame_count and op_count.
    def __init__(self, backend: str) -> None:
        # [FUNCTION DOC] Stores the backend name and initializes counters via clear().
        self.frame_count: int | CompileCounterInt = 0
        self.backend = backend
        self.graphs: list[torch.fx.GraphModule] = []
        self.clear()

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Counts the frame and call nodes, records gm, and returns the
        # result of the looked-up backend applied to gm and example_inputs.
        from .backends.registry import lookup_backend

        self.frame_count += 1
        for node in gm.graph.nodes:
            if "call" in node.op:
                self.op_count += 1
        self.graphs.append(gm)
        return lookup_backend(self.backend)(gm, example_inputs)

    def clear(self) -> None:
        # [FUNCTION DOC] Resets frame_count (CompileCounterInt when
        # debug_disable_compile_counter is set, else 0), op_count and the graphs list.
        if config.debug_disable_compile_counter:
            self.frame_count = CompileCounterInt(0)
        else:
            self.frame_count = 0
        self.op_count = 0
        self.graphs = []


# Equivalent to backend="eager", but also records graphs that
# we can assert on
class EagerAndRecordGraphs:
    # [CLASS DOC] Eager backend that appends every received graph module to graphs.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes an empty graphs list.
        self.graphs: list[torch.fx.GraphModule] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Records gm and returns gm.forward.
        self.graphs.append(gm)
        return gm.forward


class AotEagerAndRecordGraphs:
    # [CLASS DOC] Backend running aot_eager while recording the dynamo graphs (graphs)
    # and the AOT forward (fw_graphs) and backward (bw_graphs) graphs.
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes empty graphs, fw_graphs and bw_graphs lists.
        self.graphs: list[torch.fx.GraphModule] = []
        self.fw_graphs: list[torch.fx.GraphModule] = []
        self.bw_graphs: list[torch.fx.GraphModule] = []

    def __call__(
        self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> Callable[..., Any]:
        # [FUNCTION DOC] Records gm, then returns aot_eager(gm, example_inputs) with
        # forward/backward compilers that record their graphs and return gm.forward.
        self.graphs.append(gm)

        def fw_compiler(
            gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
        ) -> Callable[..., Any]:
            # [FUNCTION DOC] HELPER: Records the forward graph and returns its forward.
            self.fw_graphs.append(gm)
            return gm.forward

        def bw_compiler(
            gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]
        ) -> Callable[..., Any]:
            # [FUNCTION DOC] HELPER: Records the backward graph and returns its forward.
            self.bw_graphs.append(gm)
            return gm.forward

        return aot_eager(
            gm,
            example_inputs,
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
        )


class InductorAndRecordGraphs:
    # [CLASS DOC] Backend running Inductor's compile_fx while recording the dynamo graphs
    # (graphs) and the graphs passed to the inner Inductor compile (inductor_graphs).
    def __init__(self) -> None:
        # [FUNCTION DOC] Initializes empty graphs and inductor_graphs lists.
        self.graphs: list[torch.fx.GraphModule] = []
        self.inductor_graphs: list[torch.fx.GraphModule] = []

    def __call__(self, gm, example_inputs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] Records gm, temporarily patches compile_fx's _compile_fx_inner
        # to record its first argument before delegating, and returns compile_fx's result.
        import torch._inductor.compile_fx as compile_fx_mod

        self.graphs.append(gm)

        old_compile_fx_inner = compile_fx_mod._compile_fx_inner

        def patched(*args, **kwargs):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Records the first positional arg then calls the original.
            self.inductor_graphs.append(args[0])
            return old_compile_fx_inner(*args, **kwargs)

        with patch.object(compile_fx_mod, "_compile_fx_inner", new=patched):
            return compile_fx_mod.compile_fx(gm, example_inputs)


def strip_comment(code: str) -> str:
    # [FUNCTION DOC] Removes lines that consist only of optional spaces and a comment.
    return re.sub(r"(?m)^ *#.*\n?", "", code)


def remove_trailing_space(code: str) -> str:
    # [FUNCTION DOC] Strips trailing whitespace from every line.
    return "\n".join([line.rstrip() for line in code.split("\n")])


def _squash_blank_lines(code: str) -> str:
    # [FUNCTION DOC] HELPER: Collapses consecutive blank lines into a single blank line.
    lines = code.split("\n")
    result: list[str] = []
    saw_blank = False
    for line in lines:
        if line.strip() == "":
            if saw_blank:
                continue
            saw_blank = True
        else:
            saw_blank = False
        result.append(line)
    return "\n".join(result)


def normalize_gm(gm_str: str) -> str:
    # [FUNCTION DOC] Normalizes printed graph module text for snapshot tests: strips
    # comment lines (which contain system-specific paths), trailing spaces and repeated
    # blank lines; returns the result.
    # strip comments as comments have path to files which may differ from
    # system to system.
    stripped = strip_comment(gm_str)
    no_trailing = remove_trailing_space(stripped)
    return _squash_blank_lines(no_trailing)


def empty_line_normalizer(code: str) -> str:
    # [FUNCTION DOC] Collapses every run of CR/LF characters into one newline, removing
    # empty lines; returns the result.
    """
    Normalize code: remove empty lines.
    """
    normal_code = re.sub(r"[\r\n]+", "\n", code)
    return normal_code


def standard_test(
    self: Any,
    fn: Callable[..., Any],
    nargs: int,
    expected_ops: int | None = None,
    expected_ops_dynamic: int | None = None,
    expected_frame_count: int = 1,
) -> None:
    # [FUNCTION DOC] Standard compile test: runs fn eagerly on two sets of nargs random
    # 10x10 tensors, resets dynamo, runs optimize_assert(CompileCounter) version twice on
    # each set, and asserts results match eager and frame_count equals
    # expected_frame_count (and op_count equals expected_ops if given; the dynamic variant
    # is used when assume_static_by_default is off). self is the calling TestCase. Resets
    # dynamo state before and after.
    if not config.assume_static_by_default and expected_ops_dynamic is not None:
        expected_ops = expected_ops_dynamic

    actual = CompileCounter()

    args1 = [torch.randn(10, 10) for _ in range(nargs)]
    args2 = [torch.randn(10, 10) for _ in range(nargs)]
    correct1 = fn(*args1)
    correct2 = fn(*args2)
    reset()
    opt_fn = optimize_assert(actual)(fn)
    val1a = opt_fn(*args1)
    val2a = opt_fn(*args2)
    val1b = opt_fn(*args1)
    val2b = opt_fn(*args2)
    reset()
    self.assertTrue(same(val1a, correct1))
    self.assertTrue(same(val1b, correct1))
    self.assertTrue(same(val2a, correct2))
    self.assertTrue(same(val2b, correct2))
    self.assertEqual(actual.frame_count, expected_frame_count)
    if expected_ops is not None:
        self.assertEqual(actual.op_count, expected_ops)


def dummy_fx_compile(
    gm: fx.GraphModule, example_inputs: list[torch.Tensor]
) -> Callable[..., Any]:
    # [FUNCTION DOC] Trivial backend returning the graph module's forward unchanged.
    return gm.forward


def format_speedup(
    speedup: float,
    pvalue: float,
    is_correct: bool = True,
    pvalue_threshold: float = 0.1,
) -> str:
    # [FUNCTION DOC] Formats a benchmark result: "ERROR" if not correct; "<speedup>x SAME"
    # if pvalue exceeds the threshold; otherwise "<speedup>x p=<pvalue>" (3 and 2 decimals).
    if not is_correct:
        return "ERROR"
    if pvalue > pvalue_threshold:
        return f"{speedup:.3f}x SAME"
    return f"{speedup:.3f}x p={pvalue:.2f}"


def rand_strided(
    size: Sequence[int],
    stride: Sequence[int],
    dtype: torch.dtype = torch.float32,
    device: str | torch.device = "cpu",
    extra_size: int = 0,
) -> torch.Tensor:
    # [FUNCTION DOC] Creates a tensor with the given size and stride via as_strided over
    # a freshly allocated buffer (sized from the strides plus extra_size when all sizes
    # are positive). Float dtypes get random normal data (fp8 via fp16 cast, float4 via
    # random bytes); non-float dtypes get zeros. Returns the strided view.
    needed_size = extra_size
    if all(s > 0 for s in size):
        # only need to allocate if all sizes are non-zero
        needed_size += (
            sum((shape - 1) * stride for shape, stride in zip(size, stride)) + 1
        )
    if dtype.is_floating_point:
        if dtype == torch.float4_e2m1fn_x2:
            buffer = torch.randint(
                0, 256, (needed_size,), dtype=torch.uint8, device=device
            ).view(torch.float4_e2m1fn_x2)
        elif dtype.itemsize == 1:
            """
            normal distribution kernel is not implemented for fp8..
            Workaround that by creating a fp16 tensor and then cast.
            """
            buffer = torch.randn(needed_size, dtype=torch.float16, device=device).to(
                dtype=dtype
            )
        else:
            buffer = torch.randn(needed_size, dtype=dtype, device=device)
    else:
        buffer = torch.zeros(size=[needed_size], dtype=dtype, device=device)
    return torch.as_strided(buffer, size, stride)


_T = TypeVar("_T")


def check_dynamic_shape_capture() -> bool:
    # [FUNCTION DOC] Returns True when dynamo is configured to treat shapes as dynamic
    # (assume_static_by_default is off).
    # This also mirrors config from `test/dynamo/test_dynamic_shapes.py:make_dynamic_cls`
    return not config.assume_static_by_default


def _make_fn_with_patches(fn: Callable[_P, _T], *patches: Any) -> Callable[_P, _T]:
    # [FUNCTION DOC] HELPER: Returns a wrapper of fn that applies each (module, attr, value)
    # patch via patch.object for the duration of the call.
    @functools.wraps(fn)
    def _fn(*args: _P.args, **kwargs: _P.kwargs) -> _T:
        # [FUNCTION DOC] HELPER: Enters all patches, calls fn with the args, returns its result.
        with contextlib.ExitStack() as stack:
            for module, attr, val in patches:
                stack.enter_context(patch.object(module, attr, val))

            return fn(*args, **kwargs)

    return _fn


def make_test_cls_with_patches(
    cls: type,
    cls_prefix: str,
    fn_suffix: str,
    *patches: Any,
    xfail_prop: str | None = None,
    decorator: Callable[[Callable[..., Any]], Callable[..., Any]] = lambda x: x,
) -> type:
    # [FUNCTION DOC] Builds a new test class named cls_prefix + cls name with the same
    # bases. Each test_ method of cls is copied with fn_suffix appended and wrapped to
    # apply the patches (marked expected failure if it has the xfail_prop attribute, then
    # passed through decorator); non-callable test_ attributes and other attributes are
    # copied as-is. Returns the class.
    DummyTestClass = type(f"{cls_prefix}{cls.__name__}", cls.__bases__, {})
    DummyTestClass.__qualname__ = DummyTestClass.__name__

    for name in dir(cls):
        if name.startswith("test_"):
            fn = getattr(cls, name)
            if not callable(fn):
                setattr(DummyTestClass, name, getattr(cls, name))
                continue
            new_name = f"{name}{fn_suffix}"
            new_fn = _make_fn_with_patches(fn, *patches)
            new_fn.__name__ = new_name
            if xfail_prop is not None and hasattr(fn, xfail_prop):
                new_fn = unittest.expectedFailure(new_fn)
            setattr(DummyTestClass, new_name, decorator(new_fn))
        # NB: Doesn't handle slots correctly, but whatever
        elif not hasattr(DummyTestClass, name):
            setattr(DummyTestClass, name, getattr(cls, name))

    return DummyTestClass


# test Python 3.11+ specific features
def skipIfNotPy311(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Returns fn on Python 3.11+, otherwise unittest.skip applied to fn.
    # KNOWN ISSUE: passes fn as the skip reason instead of calling skip(reason)(fn),
    # so on old Pythons the result is a decorator, not a skipped test.
    if sys.version_info >= (3, 11):
        return fn
    # pyrefly: ignore [bad-return, bad-argument-type]
    return unittest.skip(fn)


def skipIfNotPy312(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Skips fn unless running Python 3.12 or newer.
    if sys.version_info >= (3, 12):
        return fn
    return unittest.skip("Requires Python 3.12+")(fn)


def skipIfOnlyNotPy312(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Skips fn unless running exactly Python 3.12.
    if sys.version_info >= (3, 13) or sys.version_info < (3, 12):
        return unittest.skip("Requires Python 3.12")(fn)
    return fn


def xfailIfPy312(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Marks fn as an expected failure on Python 3.12+; unchanged otherwise.
    if sys.version_info >= (3, 12):
        return unittest.expectedFailure(fn)
    return fn


def skipIfPy312(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Skips fn on Python 3.12+; unchanged otherwise.
    if sys.version_info >= (3, 12):
        return unittest.skip("Not supported in Python 3.12+")(fn)
    return fn


# Controls tests generated in test/inductor/test_torchinductor_dynamic_shapes.py
# and test/dynamo/test_dynamic_shapes.py
def expectedFailureDynamic(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Tags fn with the _expected_failure_dynamic attribute (mutates fn)
    # and returns it.
    fn._expected_failure_dynamic = True  # type: ignore[attr-defined]
    return fn


# Controls tests generated in test/inductor/test_torchinductor_codegen_dynamic_shapes.py
def expectedFailureCodegenDynamic(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Tags fn with the _expected_failure_codegen_dynamic attribute
    # (mutates fn) and returns it.
    fn._expected_failure_codegen_dynamic = True  # type: ignore[attr-defined]
    return fn


# Controls test generated in test/inductor/test_cpp_wrapper.py
def expectedFailureDynamicWrapper(fn: Callable[_P, _T]) -> Callable[_P, _T]:
    # [FUNCTION DOC] Tags fn with the _expected_failure_dynamic_wrapper attribute
    # (mutates fn) and returns it.
    fn._expected_failure_dynamic_wrapper = True  # type: ignore[attr-defined]
    return fn


def reset_rng_state(use_xla: bool = False) -> None:
    # [FUNCTION DOC] Seeds torch, python random and (if available) numpy with 1337; with
    # use_xla also sets the XLA device RNG state. Mutates global RNG state.
    torch.manual_seed(1337)
    random.seed(1337)
    if np:
        np.random.seed(1337)
    if use_xla:
        import torch_xla.core.xla_model as xm

        xm.set_rng_state(1337, str(xm.xla_device()))


def _skipped_function_for_test_reconstruct(
    f: Callable[_P, _T], *args: _P.args, **kwargs: _P.kwargs
) -> _T:
    # [FUNCTION DOC] HELPER: Calls f with the given args and returns the result; lives in
    # a skipped file so tests can exercise reconstruction of skipped functions.
    return f(*args, **kwargs)


_testing_invoke_subgraph_inductor_compile_captured_gms = None


@contextlib.contextmanager
def _testing_capture_invoke_subgraph_inductor_compile_gms() -> Generator[
    list[torch.fx.GraphModule]
]:
    # [FUNCTION DOC] HELPER: Context manager that sets the module-global captured-graphs
    # list to a new empty list, yields it, and resets the global to None on exit.
    """
    Context manager to capture graph modules compiled by invoke_subgraph_inductor_compile.

    Usage:
        with _testing_capture_invoke_subgraph_inductor_compile_gms() as captured_gms:
            # code that triggers invoke_subgraph_inductor_compile
            pass
        # captured_gms will contain the list of captured graph modules
    """
    global _testing_invoke_subgraph_inductor_compile_captured_gms
    # pyrefly: ignore [implicit-any]
    _testing_invoke_subgraph_inductor_compile_captured_gms = []
    try:
        yield _testing_invoke_subgraph_inductor_compile_captured_gms
    finally:
        _testing_invoke_subgraph_inductor_compile_captured_gms = None
