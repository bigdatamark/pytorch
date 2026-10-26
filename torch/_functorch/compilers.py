# [MODULE DOC] Collection of simple AOT compilers and debugging helpers for functorch's
# aot_function/aot_module: TorchScript compiler (ts_compile), no-op and printing compilers,
# a shape/stride-checking interpreter (DebugInterpreter, debug_nop), memory_efficient_fusion,
# and utilities that dump forward/backward/joint graphs to disk (graph_dumper_aot). Import-time
# side effects: builds default_decompositions (a decomposition table for a fixed set of ATen
# ops) and defines the global graph_index counter and a module logger.
from __future__ import annotations

import copy
import logging
import os
import pickle
import random
from contextlib import contextmanager
from functools import partial
from typing import Any, TYPE_CHECKING
from typing_extensions import ParamSpec, TypeVar

import sympy

import torch
import torch.fx as fx
import torch.nn as nn
import torch.utils._pytree as pytree
from torch import SymInt
from torch._decomp import get_decompositions
from torch.fx.experimental.symbolic_shapes import bind_symbols

from .aot_autograd import aot_function, aot_module, make_boxed_compiler
from .compile_utils import strip_overloads
from .partitioners import (
    default_partition,
    draw_graph,
    min_cut_rematerialization_partition,
)


if TYPE_CHECKING:
    from collections.abc import Callable, Generator, Sequence

    from torch.fx.node import Node
    from torch.types import IntLikeType


_P = ParamSpec("_P")
_R = TypeVar("_R")

log = logging.getLogger(__name__)


# These canonicalizations are needed here (and not decompositions), as the ops
# we're trying to canonicalize to CompositeImplicitAutograd.
def _canonicalize(fx_g: fx.GraphModule) -> fx.GraphModule:
    # [FUNCTION DOC] HELPER: Retargets every aten._to_copy call to aten.to in place, recompiles
    # the graph module, and returns it.
    for node in fx_g.graph.find_nodes(
        op="call_function", target=torch.ops.aten._to_copy
    ):
        node.target = torch.ops.aten.to
    fx_g.recompile()
    return fx_g


@contextmanager
def _disable_jit_autocast() -> Generator[None, None, None]:
    # [FUNCTION DOC] HELPER: Context manager that turns off TorchScript autocast mode for its
    # body and restores the previous flag on exit, even on error.
    # pyrefly: ignore [missing-attribute]
    old_jit_autocast_flag = torch._C._jit_set_autocast_mode(False)
    try:
        yield
    finally:
        # pyrefly: ignore [missing-attribute]
        torch._C._jit_set_autocast_mode(old_jit_autocast_flag)


@make_boxed_compiler
def ts_compile(fx_g: fx.GraphModule, inps: Sequence[Any]) -> torch.jit.ScriptModule:
    # [FUNCTION DOC] Boxed compiler: unwraps a lazy graph module, then with JIT autocast
    # disabled strips overloads, turns single-dtype-kwarg _to_copy calls into aten.to, converts
    # device kwargs to their type strings, lints and recompiles, scripts the graph, removes
    # mutation, freezes in eval mode, and optimizes for inference. If no input is a fake tensor,
    # runs the result once on inps as a warmup. Mutates fx_g in place; returns the scripted module.
    """
    Compiles the :attr:`fx_g` with Torchscript compiler.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fx_g(fx.GraphModule): The input Fx graph module to be compiled.

    Returns:
        Torch scripted model.
    """

    from torch.fx._lazy_graph_module import _unwrap_lazy_graph_module

    fx_g = _unwrap_lazy_graph_module(fx_g)

    with _disable_jit_autocast():
        strip_overloads(fx_g)

        for node in fx_g.graph.find_nodes(
            op="call_function", target=torch.ops.aten._to_copy
        ):
            if len(node.args) == 1 and len(node.kwargs) == 1 and "dtype" in node.kwargs:
                node.target = torch.ops.aten.to

        for node in fx_g.graph.nodes:
            new_kwargs = {}
            for k, v in node.kwargs.items():
                if isinstance(v, torch.device):
                    v = v.type
                new_kwargs[k] = v
            node.kwargs = new_kwargs

        fx_g.graph.lint()

        fx_g.recompile()

        f = torch.jit.script(fx_g)

        # pyrefly: ignore [missing-attribute]
        torch._C._jit_pass_remove_mutation(f.graph)

        f = torch.jit.freeze(f.eval())
        f = torch.jit.optimize_for_inference(f)
        if not any(torch._subclasses.fake_tensor.is_fake_tensor(t) for t in inps):
            f(*inps)
    return f


def _draw_graph_compile(
    fx_g: fx.GraphModule, _: Any, name: str, clear_meta: bool = True
) -> fx.GraphModule:
    # [FUNCTION DOC] HELPER: Prints the graph code, renders the graph to a file named by name via
    # draw_graph (optionally clearing node meta), and returns the graph module unchanged.
    print(fx_g.code)
    draw_graph(fx_g, name, clear_meta=clear_meta)
    return fx_g


def draw_graph_compile(
    name: str,
) -> Callable[[fx.GraphModule, list[Any]], fx.GraphModule]:
    # [FUNCTION DOC] Returns a boxed compiler that prints and draws each graph it receives under
    # the given file name and returns the graph unchanged.
    return make_boxed_compiler(partial(_draw_graph_compile, name=name))


@make_boxed_compiler
def nop(fx_g: fx.GraphModule, _: Any) -> fx.GraphModule:
    # [FUNCTION DOC] Boxed no-op compiler: returns the graph module as is, ignoring example inputs.
    """
    Returns the :attr:`fx_g` Fx graph module as it is. This is a no-op compiler
    and can be used to check accuracy.

    .. warning::
        This API is experimental and likely to change.

    """
    return fx_g


class DebugInterpreter(fx.Interpreter):
    # [CLASS DOC] FX interpreter that, after executing each node, checks the real outputs against
    # the node's recorded meta val: dtype, size, and (for dims of size > 1) strides must match.
    # State: symbol_mapping, binding of the graph's symbolic sizes to concrete values, set in run.
    # Raises AssertionError on any mismatch. Slow; intended for debugging.
    def run(
        self,
        *args: Any,
        initial_env: dict[Node, Any] | None = None,
        enable_io_processing: bool = True,
    ) -> Any:
        # [FUNCTION DOC] Binds the module's symbolic shape symbols to the concrete args (stored in
        # symbol_mapping) and then runs the normal interpreter with the same options.
        self.symbol_mapping = bind_symbols(
            # pyrefly: ignore[bad-argument-type]
            self.module,
            *args,
        )
        return super().run(
            *args, initial_env=initial_env, enable_io_processing=enable_io_processing
        )

    def run_node(self, n: Node) -> Any:
        # [FUNCTION DOC] Executes node n, then, if it has a meta val, flattens recorded and real
        # outputs, requires equal counts, and for each real tensor output checks dtype, size and
        # significant strides against the recorded value. Returns the real result. Raises
        # AssertionError with the output index and symbol mapping on any mismatch.
        def subst_symint(ni: IntLikeType) -> int:
            # [FUNCTION DOC] HELPER: Returns ni if not symbolic; otherwise substitutes symbol_mapping
            # into its expression and returns the integer result. Raises if not fully numeric.
            if not isinstance(ni, SymInt):
                return ni
            r = sympy.expand(ni.node.expr.xreplace(self.symbol_mapping))
            if not r.is_number:
                raise AssertionError(f"expected r to be a number, got {r}")
            return int(r)

        def subst_symint_tuple(nis: tuple[IntLikeType, ...]) -> tuple[int, ...]:
            # [FUNCTION DOC] HELPER: Applies subst_symint to each element and returns a tuple of ints.
            return tuple(subst_symint(ni) for ni in nis)

        def check_significant_strides(a: torch.Tensor, b: torch.Tensor) -> bool:
            # [FUNCTION DOC] HELPER: For a non-empty (symbolically resolved) tensor a, returns False
            # if any dimension with size greater than 1 has a stride differing from b; else True.
            if subst_symint(a.numel()) > 0:
                for idx in range(a.ndim):
                    if (
                        subst_symint(a.stride(idx)) != b.stride(idx)
                        and subst_symint(a.size(idx)) > 1
                    ):
                        return False
            return True

        def check(nv: torch.Tensor, rv: torch.Tensor, desc: Callable[[], str]) -> None:
            # [FUNCTION DOC] HELPER: Asserts the recorded tensor nv and real tensor rv agree in
            # dtype, size and significant strides; desc is a callable giving a message prefix.
            # Raises AssertionError if desc is not callable or on any mismatch.
            if not callable(desc):
                raise AssertionError(f"expected desc to be callable, got {type(desc)}")
            if nv.dtype != rv.dtype:
                raise AssertionError(f"{desc()}: {nv.dtype} != {rv.dtype}")
            if subst_symint_tuple(nv.size()) != rv.size():
                raise AssertionError(
                    f"{desc()}: {nv.size()} aka {subst_symint_tuple(nv.size())} != {rv.size()}"
                )
            same_strides = check_significant_strides(nv, rv)
            if not same_strides:
                raise AssertionError(
                    f"{desc()}: {nv.stride()} aka {subst_symint_tuple(nv.stride())} != {rv.stride()}"
                )

        r = super().run_node(n)
        if "val" in n.meta:
            n_vals, _n_spec = pytree.tree_flatten(n.meta["val"])
            r_vals, _r_spec = pytree.tree_flatten(r)
            # TODO: There is some sort of problem where we record that an
            # operator returned a tuple/list, and then later it turns out the
            # real version of the operator returned a list/tuple. Need to
            # figure out what's actually going on here, the error itself is
            # harmless enough as we only getitem out the outputs.
            # assert n_spec == r_spec, f"{n_spec} != {r_spec}"
            if len(n_vals) != len(r_vals):
                raise AssertionError(f"{len(n_vals)} != {len(r_vals)}")
            for i, nv, rv in zip(range(len(n_vals)), n_vals, r_vals):
                if not isinstance(rv, torch.Tensor):
                    continue
                check(nv, rv, lambda: f"output {i} where {self.symbol_mapping}")
        return r


@make_boxed_compiler
def debug_nop(
    fx_g: fx.GraphModule, _: Any
) -> Callable[[DebugInterpreter, Any, dict[Node, Any] | None, bool], Any]:
    # [FUNCTION DOC] Boxed compiler returning the run method of a DebugInterpreter over fx_g,
    # i.e. a slow callable that validates dtypes, sizes and strides while executing.
    # KNOWN ISSUE: the annotated return type lists self as a parameter although a bound method is returned.
    """
    Returns a (slow) interpreter over the FX graph module that also checks
    various debugging properties (e.g., that tracing strides matched real
    strides.)
    """
    return DebugInterpreter(fx_g).run


@make_boxed_compiler
def simple_ts_compile(fx_g: fx.GraphModule, _: Any) -> torch.jit.ScriptModule:
    # [FUNCTION DOC] Boxed compiler: strips overloads in place, scripts the graph with TorchScript,
    # and returns the frozen eval-mode module. Does less than ts_compile (no inference passes).
    strip_overloads(fx_g)
    f = torch.jit.script(fx_g)
    f = torch.jit.freeze(f.eval())
    return f


def nnc_jit(f: Callable[..., Any]) -> Callable[..., Any]:
    # [FUNCTION DOC] Wraps f with aot_function using simple_ts_compile as the compiler and
    # returns the compiled callable.
    return aot_function(f, simple_ts_compile)


aten = torch.ops.aten
default_decompositions = {
    aten.detach,
    aten.gelu_backward,
    aten.leaky_relu_backward,
    aten.sigmoid_backward,
    aten.threshold_backward,
    aten.hardtanh_backward,
    aten.hardsigmoid_backward,
    aten.hardswish_backward,
    aten.tanh_backward,
    aten.silu_backward,
    aten.elu_backward,
    aten.cudnn_batch_norm,
    aten.cudnn_batch_norm_backward,
    aten.masked_fill.Scalar,
    aten.masked_fill.Tensor,
    aten.elu,
    aten.leaky_relu,
    aten.hardtanh,
    aten.hardswish,
    aten.hardsigmoid,
    aten.conj_physical,
    aten.is_same_size,
}

# pyrefly: ignore[bad-argument-type]
default_decompositions = get_decompositions(default_decompositions)


@make_boxed_compiler
def print_compile(fx_g: fx.GraphModule, _: Any) -> fx.GraphModule:
    # [FUNCTION DOC] Boxed compiler: prints the graph's generated code and returns the graph unchanged.
    print(fx_g.code)
    return fx_g


def memory_efficient_fusion(
    fn: Callable[_P, _R] | nn.Module,
    **kwargs: Any,
) -> Callable[_P, _R] | nn.Module:
    # [FUNCTION DOC] Builds a config with ts_compile for forward and backward, the min-cut
    # rematerialization partitioner, and default_decompositions; kwargs override entries. Returns
    # aot_module(fn) if fn is an nn.Module, else aot_function(fn).
    # KNOWN ISSUE: docstring mentions NVFuser, but the code uses the TorchScript compiler.
    """
    Wrapper function over :func:`aot_function` and :func:`aot_module` to perform
    memory efficient fusion. It uses the
    :func:`min_cut_rematerialization_partition` partitioner to perform efficient
    recomputation. It uses NVFuser to compile the generated forward and backward
    graphs.

    .. warning::
        This API is experimental and likely to change.

    Args:
        fn (Union[Callable, nn.Module]): A Python function or a ``nn.Module``
            that takes one or more arguments. Must return one or more Tensors.
        **kwargs: Any other overrides you want to make to the settings

    Returns:
        Returns a ``Callable``  or ``nn.Module`` that retains the eager behavior
        of the original :attr:`fn`, but whose forward and backward graphs have
        gone through recomputation optimizations, and the graphs have been
        compiled with nvfuser.

    """
    config = {
        "fw_compiler": ts_compile,
        "bw_compiler": ts_compile,
        "partition_fn": min_cut_rematerialization_partition,
        "decompositions": default_decompositions,
    }
    config.update(kwargs)
    if isinstance(fn, torch.nn.Module):
        return aot_module(fn, **config)  # pyrefly: ignore[bad-argument-type]
    else:
        return aot_function(fn, **config)  # pyrefly: ignore[bad-argument-type]


def debug_compile(
    fx_g: fx.GraphModule, inps: Sequence[torch.Tensor]
) -> torch.jit.ScriptModule:
    # [FUNCTION DOC] HELPER: Debugging compiler. Writes the graph as a module into a folder named
    # foo, prints repro instructions for the minifier (using input shapes and dtypes), imports
    # that module, runs it once on CUDA with inps, then returns ts_compile(fx_g, inps).
    # Side effects: file output in the current directory, stdout, CUDA execution. Raises if
    # CUDA is unavailable or inps are not usable tensors.
    # KNOWN ISSUE: inps are passed to the CUDA module unchanged, so CPU inputs would fail.
    fx_g.to_folder("foo")
    print(
        f"""
##############################################################
# To minimize FX graph, copy and paste the below and run it  #
##############################################################

import torch
import torch.fx as fx
from functorch.compile import minifier, check_nvfuser_subprocess, check_nvfuser_correctness_subprocess

inps = {[(i.shape, i.dtype) for i in inps]}
inps = [torch.ones(shape, dtype=dtype, device='cuda') for (shape, dtype) in inps]
from foo import FxModule
mod = FxModule().cuda()

with torch.jit.fuser("fuser2"):
  # check_nvfuser_subprocess can be replaced with check_nvfuser_correctness_subprocess
  minifier(fx.symbolic_trace(mod), inps, check_nvfuser_subprocess)
"""
    )
    # pyrefly: ignore[missing-import, missing-module-attribute]
    from foo import FxModule

    FxModule().cuda()(*inps)

    return ts_compile(fx_g, inps)


graph_index: int = 0


def get_inputs(input_data_path: str) -> list[torch.Tensor]:
    # [FUNCTION DOC] Loads a pickled list of input metadata (as written by _save_fx_default) and
    # builds random inputs: a one-element entry becomes a random float converted with the
    # recorded type; otherwise a tensor of the recorded shape, dtype and device, filled with
    # zeros for integer/bool-like dtypes (randint in [0,1)) or uniform random values. Returns
    # the list. Raises on file or unpickling errors. Strides in the metadata are ignored.
    # KNOWN ISSUE: unpickling executes arbitrary code if the file is untrusted.
    """
    Return a random input for the given inputs meta generated from _save_fx_default.
    """
    inputs: list[torch.Tensor] = []
    with open(input_data_path, "rb") as f:
        inputs_meta = pickle.load(f)
        inputs = []
        for meta in inputs_meta:
            if len(meta) == 1:
                type = meta
                input_ = type(random.random())
            else:
                type, shape, _stride, dtype, device = meta
                if dtype in {
                    torch.int,
                    torch.int32,
                    torch.int64,
                    torch.bool,
                    torch.int,
                    torch.uint8,
                    int,
                    float,
                }:
                    input_ = torch.randint(0, 1, shape, dtype=dtype, device=device)
                else:
                    input_ = torch.rand(shape, dtype=dtype, device=device)
            inputs.append(input_)
    return inputs


def _save_fx_default(
    current_name: str,
    folder_name: str,
    dump_example_input: bool,
    gm: torch.fx.GraphModule,
    example_inputs: list[torch.Tensor],
) -> nn.Module:
    # [FUNCTION DOC] HELPER: Runs aot_module_simplified on gm with compilers that save graphs to
    # disk: forward and backward graphs (and the joint graph, partitioned by default_partition)
    # are written under folder_name/current_name along with pickled input metadata, optionally
    # the example inputs. Uses and increments the global graph_index (once per backward). Returns
    # the compiled module. Side effects: filesystem writes, warning log for empty graphs.
    """
    The forward, backward, and joint computation graph will be stored in
    {folder_name}/{current_name}/{current_name}_forward_{graph_index},
    {folder_name}/{current_name}/{current_name}_backward_{graph_index}, and
    {folder_name}/{current_name}/{current_name}_joint_{graph_index} respectively.
    The input shape of the graphs will be stored in the .input files.
    These files can be loaded with pickle,
    and is a list of format (type, shape, stride, dtype, device).
    In the case of type = int or float, it is just (type,).
    For joint graph input, it is a nested list [[],[]]
    where the two inner lists have the same format.
    If dump_example_input is True, example_inputs will be stored in .pt file.
    Since each function might produce multiple graphs,
    the graph_index is used to distinguish different graphs
    """
    from functorch.compile import aot_module_simplified

    def get_input_meta(args: Any) -> list[Any]:
        # [FUNCTION DOC] HELPER: Builds a list of per-argument metadata: a one-element entry with the
        # type for plain int/float, else (type, shape, stride, dtype, device). If the first arg is
        # a tuple (joint input), concatenates the metadata of the first two args recursively.
        input_meta = []
        if len(args) > 0 and isinstance(args[0], tuple):  # joint input
            input_meta += get_input_meta(args[0])
            input_meta += get_input_meta(args[1])
            return input_meta
        for arg in args:
            if type(arg) is int or type(arg) is float:
                input_meta.append((type(arg),))
            else:
                input_meta.append(
                    (type(arg), arg.shape, arg.stride(), arg.dtype, arg.device)
                )
        return input_meta

    def graph_saver_helper(
        gm_to_save: fx.GraphModule, args: Any, type_name: str
    ) -> None:
        # [FUNCTION DOC] HELPER: Saves gm_to_save (a deep copy with default codegen) as a folder module
        # named current_name_type_name_graph_index plus a pickled .input metadata file, and
        # optionally the args as a .pt file. Empty graphs only log a warning and are skipped.
        global graph_index
        if len(gm_to_save.graph.nodes) == 0:
            log.log(
                logging.WARNING,
                "No nodes in graph {%s}_{%s}_{%s}.",
                current_name,
                type_name,
                graph_index,
            )
            return

        gm = copy.deepcopy(gm_to_save)
        gm.graph.set_codegen(torch.fx.graph.CodeGen())  # remove codegen
        gm.recompile()

        input_meta = get_input_meta(args)

        os.makedirs(f"{folder_name}/{current_name}", exist_ok=True)
        gm.to_folder(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}"
        )
        with open(
            f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.input",
            "wb",
        ) as f:
            pickle.dump(input_meta, f)
        if dump_example_input:
            torch.save(
                args,
                f"{folder_name}/{current_name}/{current_name}_{type_name}_{graph_index}/{current_name}_{type_name}_{graph_index}.pt",
            )

    def graph_saver_forward(
        gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> fx.GraphModule:
        # [FUNCTION DOC] HELPER: Forward compiler hook: saves the graph as "forward" and returns it.
        graph_saver_helper(gm, example_inputs, "forward")
        return gm

    def graph_saver_backward(
        gm: fx.GraphModule, example_inputs: list[torch.Tensor]
    ) -> fx.GraphModule:
        # [FUNCTION DOC] HELPER: Backward compiler hook: saves the graph as "backward", increments the
        # global graph_index, and returns the graph unchanged.
        graph_saver_helper(gm, example_inputs, "backward")
        global graph_index
        graph_index += 1
        return gm

    def graph_saver_joint(
        gm: fx.GraphModule, joint_args: list[torch.Tensor]
    ) -> tuple[fx.GraphModule, fx.GraphModule]:
        # [FUNCTION DOC] HELPER: Partition hook: saves the joint graph as "joint", then returns the
        # forward/backward split from default_partition.
        graph_saver_helper(gm, joint_args, "joint")
        return default_partition(gm, joint_args)  # pyrefly: ignore[missing-argument]

    # pyrefly: ignore[bad-return]
    return aot_module_simplified(
        gm,
        example_inputs,
        fw_compiler=graph_saver_forward,  # pyrefly: ignore[bad-argument-type]
        bw_compiler=graph_saver_backward,  # pyrefly: ignore[bad-argument-type]
        partition_fn=graph_saver_joint,
        decompositions=default_decompositions,  # pyrefly: ignore[bad-argument-type]
    )


# WARNING: This isn't tested anywhere!!
def graph_dumper_aot(
    current_name: str, folder_name: str, dump_example_input: bool = False
) -> Callable[[bool, nn.Module], Any]:
    # [FUNCTION DOC] Resets the global graph_index to 0 and returns a partial of _save_fx_default
    # bound to current_name, folder_name and dump_example_input, suitable as a backend that
    # dumps forward/backward/joint graphs when called with a module and example inputs.
    # KNOWN ISSUE: the annotated return type (bool, module) does not match the actual remaining
    # parameters (graph module, example inputs); also marked in source as untested.
    """
    Dump the forward, backward, and joint computation graph.
    Example Usage:
    save_fx_func = graph_dumper_aot(current_name, folder_name, dump_example_input = False)
    optimize_ctx = torchdynamo.optimize(
        save_fx_func
    )
    with torch.enable_grad():
        with optimize_ctx:
            result = forward_and_backward_pass(model, example_inputs)
    """
    global graph_index
    graph_index = 0
    return partial(_save_fx_default, current_name, folder_name, dump_example_input)
