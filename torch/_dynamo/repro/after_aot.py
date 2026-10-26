# [MODULE DOC] Repro generation, minification and accuracy analysis for graphs
# after AOT Autograd (the Inductor-facing stage). Main pieces: a compiler
# wrapper (wrap_compiler_debug) that dumps repro/minifier scripts on failure or
# bad accuracy; generators that serialize an FX graph, its inputs, config, env
# vars and user Triton kernels into a standalone script; minifier predicates
# (inductor_fails, inductor_accuracy_fails); and the command-line driver
# run_repro (run / minify / analyze / get_args / minifier-query) embedded in
# generated scripts. Import-time side effects: imports Triton if present (else
# defines placeholder classes), loads the inductor config module, and sets the
# module flag use_buck from the fbcode check.
"""
Utilities for reproducing and debugging issues in PyTorch's Dynamo AOT compilation.

This module provides tools and infrastructure for:
1. Generating minimal reproducible test cases ("repros") from failing compilations
2. Analyzing accuracy issues between eager and compiled execution
3. Minifying large models/inputs to isolate problematic patterns
4. Debugging compiler errors and accuracy divergences

The main components include:
- Repro generation: Creates standalone Python files that reproduce compiler issues
- Minification: Reduces large graphs to minimal failing examples
- Accuracy analysis: Compares compiled vs eager execution, with fp64 reference
- Debug tools: Dumps graph state, tracks intermediates, analyzes divergences

This is primarily used by PyTorch developers and researchers to debug issues in
the Dynamo AOT compilation pipeline, particularly for the Inductor backend.
"""

from __future__ import annotations

import argparse
import ast
import copy
import functools
import io
import logging
import os
import shutil
import subprocess
import sys
import textwrap
import types
import typing
import uuid
from importlib import import_module
from tempfile import TemporaryFile
from typing import Any, IO, TYPE_CHECKING, TypedDict
from typing_extensions import Unpack

import sympy


try:
    import triton.language as tl
    from triton.runtime.autotuner import Autotuner, Heuristics
    from triton.runtime.jit import JITFunction

    TritonConstexpr = tl.constexpr
except ImportError:

    class Autotuner:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Empty placeholder used when Triton is not installed
        # so isinstance checks against Triton autotuner kernels are always false.
        pass

    class JITFunction:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Empty placeholder used when Triton is not installed
        # so isinstance checks against Triton JIT kernels are always false.
        pass

    class Heuristics:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Empty placeholder used when Triton is not installed
        # so isinstance checks against Triton heuristics wrappers are always false.
        pass

    class TritonConstexpr:  # type: ignore[no-redef]
        # [CLASS DOC] HELPER: Empty placeholder used when Triton is not installed
        # so isinstance checks against Triton constexpr values are always false.
        pass


import torch
import torch.fx as fx
import torch.nn as nn
from torch._dynamo.debug_utils import (
    _cuda_system_info_comment,
    AccuracyError,
    backend_accuracy_fails,
    BuckTargetWriter,
    cast_to_fp64,
    extra_deps,
    extra_imports,
    generate_config_string,
    generate_env_vars_string,
    helper_for_dump_minify,
    InputReader,
    InputWriter,
    MAX_CONSTANT_NUMEL_INLINE,
    minifier_dir,
    NNModuleToString,
    NopInputReader,
    same_two_models,
)
from torch._dynamo.utils import clone_inputs, counters, same
from torch._environment import is_fbcode
from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table
from torch._inductor.cpp_builder import normalize_path_separator
from torch._library.fake_class_registry import FakeScriptObject
from torch._ops import OpOverload
from torch.fx.experimental.proxy_tensor import make_fx
from torch.fx.experimental.symbolic_shapes import (
    fx_placeholder_targets,
    has_free_symbols,
)
from torch.hub import tqdm

from .. import config


def _find_repeat_interleave_constraints(
    gm: torch.fx.GraphModule,
) -> list[tuple[str, str]]:
    # [FUNCTION DOC] HELPER: Scans the graph for repeat_interleave calls whose
    # first argument and output_size keyword are both graph placeholders. Returns
    # a list of (repeats placeholder name, output_size placeholder name) pairs.
    # Pure; no side effects. Nodes missing either placeholder form are skipped.
    """
    Find repeat_interleave operations with output_size constraints.

    Returns list of (repeats_placeholder_name, output_size_placeholder_name) pairs.
    These represent constraints where sum(repeats) must equal output_size.
    """
    constraints = []
    for node in gm.graph.nodes:
        if (
            node.op != "call_function"
            or "repeat_interleave" not in str(node.target)
            or not node.args
        ):
            continue

        output_size_node = node.kwargs.get("output_size")
        repeats_node = node.args[0]

        # Both must be FX nodes (not constants) and direct placeholders
        if (
            isinstance(repeats_node, torch.fx.Node)
            and isinstance(output_size_node, torch.fx.Node)
            and repeats_node.op == "placeholder"
            and output_size_node.op == "placeholder"
        ):
            constraints.append((str(repeats_node.target), str(output_size_node.target)))

    return constraints


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch._inductor.compile_fx import _CompileFxCallable, _CompileFxKwargs
    from torch._inductor.output_code import OutputCode
    from torch._inductor.utils import InputType

    from . import ReproOptions


log = logging.getLogger(__name__)


inductor_config = import_module("torch._inductor.config")


class GroupInfo(TypedDict):
    # [CLASS DOC] HELPER: Typed mapping describing one process group: its
    # world size and the rank of the current process within it.
    size: int
    rank: int


def _extract_distributed_info(
    gm: torch.fx.GraphModule,
) -> dict[str, GroupInfo]:
    # [FUNCTION DOC] HELPER: Walks graph nodes that call c10d functional
    # collective op overloads, normalizes their arguments to keywords, and for each
    # distinct string group_name resolves the live process group to record its size
    # and this process's rank. Returns a mapping from group name to GroupInfo.
    # Nodes that are not such ops, cannot be normalized, or lack a string group name
    # are skipped. Requires the named groups to exist in the current process.
    """
    Extract process group information from distributed ops in the graph.

    Returns a dict mapping group names to dicts with 'size' and 'rank' keys.
    Example: {'tp': {'size': 4, 'rank': 0}, 'dp': {'size': 2, 'rank': 0}}
    """
    from torch.distributed import GroupName
    from torch.fx.operator_schemas import normalize_function

    group_info: dict[str, GroupInfo] = {}

    for node in gm.graph.nodes:
        if node.op != "call_function":
            continue
        if not isinstance(node.target, OpOverload):
            continue
        if node.target.namespace not in {"_c10d_functional", "c10d_functional"}:
            continue

        opt_args_kwargs = normalize_function(
            node.target,
            args=node.args,
            kwargs=node.kwargs,
            normalize_to_only_use_kwargs=True,
        )
        if opt_args_kwargs is None:
            continue
        _, kwargs = opt_args_kwargs

        group_name_ = kwargs.get("group_name")
        if not isinstance(group_name_, str):
            continue
        group_name = typing.cast(GroupName, group_name_)

        if group_name in group_info:
            continue

        from torch.distributed.distributed_c10d import (
            _get_group_size_by_name,
            _resolve_process_group,
        )

        group_size = _get_group_size_by_name(group_name)
        pg = _resolve_process_group(group_name)
        rank = pg.rank()
        group_info[group_name] = {"size": group_size, "rank": rank}

    return group_info


def setup_fake_process_groups(
    group_info: dict[str, GroupInfo],
) -> None:
    # [FUNCTION DOC] Recreates fake (single-process) distributed state for a repro
    # run. Does nothing for an empty mapping. Otherwise initializes a default fake
    # process group with the largest group size as world size (rank taken from the
    # first group of that size), clears all registered process groups, then
    # registers the default group under every full-size group name and creates and
    # registers new groups (ranks 0..size-1) for smaller ones. Side effects: mutates
    # global distributed state.
    """
    Set up fake process groups for repro execution.

    Args:
        group_info: dict mapping group_name -> {'size': group_size, 'rank': rank}
    """
    import torch.distributed as dist
    from torch.testing._internal.distributed.fake_pg import FakeStore

    if not group_info:
        return

    world_size = max(info["size"] for info in group_info.values())

    global_rank = 0
    for info in group_info.values():
        if info["size"] == world_size:
            global_rank = info["rank"]
            break

    store = FakeStore()
    dist.init_process_group(
        backend="fake",
        rank=global_rank,
        world_size=world_size,
        store=store,
    )

    default_pg = dist.distributed_c10d._get_default_group()
    torch._C._distributed_c10d._unregister_all_process_groups()

    for group_name, info in group_info.items():
        group_size = info["size"]
        if group_size == world_size:
            # pyrefly: ignore[bad-argument-type]
            torch._C._distributed_c10d._register_process_group(group_name, default_pg)
        else:
            ranks = list(range(group_size))
            new_pg = dist.new_group(ranks)
            # pyrefly: ignore[bad-argument-type]
            torch._C._distributed_c10d._register_process_group(group_name, new_pg)


def generate_standalone_repro(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    *,
    save_path: str | None = None,
) -> str:
    # [FUNCTION DOC] Builds the text of a standalone inductor repro script for the
    # graph and example args by writing it to an in-memory buffer (no save dir, so
    # inputs are not stored on disk). If save_path is given, also writes the script
    # to that file and logs it. Returns the script text.
    """
    Generate a self-contained repro script from an FX graph.
    """
    buf = io.StringIO()
    save_graph_repro(buf, gm, args, "inductor", save_dir=None)
    repro = buf.getvalue()

    if save_path is not None:
        with open(save_path, "w") as f:
            f.write(repro)
        log.info("Saved standalone repro to %s", save_path)

    return repro


use_buck = is_fbcode()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           MAIN ENTRY POINT
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def wrap_compiler_debug(
    unconfigured_compiler_fn: _CompileFxCallable,
    compiler_name: str,
) -> _CompileFxCallable:
    # [FUNCTION DOC] Wraps a compile_fx-style compiler so failures or accuracy
    # problems are turned into repro/minifier scripts, according to the
    # repro_after and repro_level config. Returns the wrapper (debug_wrapper), which
    # preserves the wrapped function's metadata. compiler_name is used in dump
    # names and must be inductor for accuracy mode.
    """
    Minifier for Fx Graph modules after Aot Autograd has finished. We wrap both
    forward and backward call separately with the backend compiler_fn - like
    inductor or nvfuser. Intercepting after Aot Autograd presents neat
    abstraction, where all the params are lifted as graph inputs, making it easy
    to save the graph as a string.
    """

    @functools.wraps(unconfigured_compiler_fn)
    def debug_wrapper(
        gm: torch.fx.GraphModule,
        example_inputs: Sequence[InputType],
        compile_region_name: str | None = None,
        **kwargs: Unpack[_CompileFxKwargs],
    ) -> OutputCode:
        # [FUNCTION DOC] HELPER: Compiles the graph with the wrapped compiler
        # (extra kwargs and region name bound). Raises if repro_after is not
        # dynamo, aot or unset. If compilation raises and repro_after is aot, dumps
        # graph state (level 1) or a minifier launcher (level 2) from a pre-compile
        # copy of the graph, logs, and re-raises. When repro_after is aot returns a
        # boxed callable that defers to inner_debug_fn on real inputs; otherwise
        # returns the compiled function unchanged.
        from torch._subclasses import FakeTensorMode

        compiler_fn = functools.partial(
            unconfigured_compiler_fn,
            compile_region_name=compile_region_name,
            **kwargs,
        )

        from torch._functorch.aot_autograd import get_aot_graph_name

        graph_name = get_aot_graph_name()

        # TODO: why do we need to deepcopy the original graph?
        orig_graph = copy.deepcopy(gm.graph)
        if config.repro_after not in ("dynamo", "aot", None):
            raise AssertionError(
                f"repro_after must be 'dynamo', 'aot', or None, got {config.repro_after!r}"
            )

        try:
            # Call the compiler_fn - which is either aot_autograd or inductor
            # with fake inputs
            inner_compiled_fn = compiler_fn(gm, example_inputs)
        except Exception:
            # TODO: Failures here are troublesome because no real inputs,
            # need a different serialization strategy
            if config.repro_after == "aot":
                if config.repro_level == 1:
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                elif config.repro_level == 2:
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        example_inputs,
                        compiler_name,
                    )
                log.error("CompilerError")
            raise

        # We may run regular PyTorch compute that may trigger Dynamo, do NOT
        # recursively attempt to accuracy minify in that case!
        def deferred_for_real_inputs(
            real_inputs: Sequence[InputType], **_kwargs: object
        ) -> Any:
            # [FUNCTION DOC] HELPER: Boxed-call entry that runs the compiled
            # function on real inputs. Raises if unexpected kwargs are passed. If
            # repro_after is not aot, calls the compiled function directly (error if
            # it is a string); otherwise runs inner_debug_fn with repro_after
            # temporarily disabled to avoid recursive minification.
            # This is a bit obscure: if we recursively try to accuracy minify
            # the SAME function, this would trigger.  But most of the time
            # we should never hit this branch
            if _kwargs:
                raise AssertionError(f"Unexpected kwargs: {_kwargs}")
            if config.repro_after != "aot":
                if isinstance(inner_compiled_fn, str):
                    raise AssertionError("inner_compiled_fn should not be a string")
                return inner_compiled_fn(real_inputs)
            with config.patch(repro_after=None):
                return inner_debug_fn(real_inputs)

        def inner_debug_fn(real_inputs: Sequence[InputType]) -> Any:
            # [FUNCTION DOC] HELPER: Runs the compiled function on real inputs with
            # debugging by repro_level. Level 3 always dumps a minifier launcher
            # first. Level 4 (inductor only, else NotImplementedError) compares
            # against eager for accuracy; on failure dumps graph state and minifier
            # script and raises AccuracyError. Otherwise runs the compiled function,
            # synchronizes the accelerator for non-CPU inputs to surface faults, and
            # on exception dumps state (level 1) or minifier script (level 2) from
            # fake copies of the inputs before re-raising. Returns the outputs.
            """
            Aot Autograd fw_compiler and bw_compiler can have fake tensors. So,
            example_inputs can be fake tensors. We can call compiler_fn (which is
            inductor or nvfuser) with fake tensors but the actually compiled_fn
            should be called with real tensors. Therefore, the actual invocation
            is deferred.
            """
            # Copy the tensor attrs like shape, stride etc by converting to Fake Tensor
            # because inductor clears the tensor list in its codegen. And example_inputs
            # are available only for the first invocation.
            fake_mode = FakeTensorMode()
            copy_tensor_attrs = [
                fake_mode.from_tensor(x) if isinstance(x, torch.Tensor) else x
                for x in real_inputs
            ]
            if config.repro_level == 3:
                # Always dump the original module in case we have segfaults
                dump_to_minify(
                    fx.GraphModule(gm, orig_graph), real_inputs, compiler_name
                )

            if config.repro_level == 4:
                if compiler_name != "inductor":
                    raise NotImplementedError(
                        "Accuracy minification is supported for inductor only"
                    )
                failed = not same_two_models(
                    gm,
                    inner_compiled_fn,  # type: ignore[arg-type]
                    real_inputs,
                    only_fwd=True,
                    ignore_non_fp=config.repro_ignore_non_fp,
                )

                if failed:
                    log.warning(
                        "Accuracy failed for the AOT Autograd graph %s", graph_name
                    )
                    dump_compiler_graph_state(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    dump_to_minify(
                        fx.GraphModule(gm, orig_graph),
                        real_inputs,
                        f"{compiler_name}_accuracy",
                    )
                    raise AccuracyError("Bad accuracy detected")
                else:
                    # Call the compiled function with real inputs
                    return inner_compiled_fn(real_inputs)  # type: ignore[operator]
            else:
                try:
                    # Call the compiled function with real inputs
                    out = inner_compiled_fn(real_inputs)  # type: ignore[operator]
                    # sync cuda kernels to ensure IMA detection
                    if (
                        any(
                            isinstance(arg, torch.Tensor) and arg.device.type != "cpu"
                            for arg in example_inputs
                        )
                        and torch.accelerator.is_available()
                    ):
                        torch.accelerator.synchronize()
                    return out
                except Exception:
                    if config.repro_level == 1:
                        dump_compiler_graph_state(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    elif config.repro_level == 2:
                        dump_to_minify(
                            fx.GraphModule(gm, orig_graph),
                            copy_tensor_attrs,
                            compiler_name,
                        )
                    raise

        if config.repro_after == "aot":
            compiled_fn = deferred_for_real_inputs
            compiled_fn._boxed_call = True  # type: ignore[attr-defined]
            return compiled_fn  # type: ignore[return-value]
        else:
            return inner_compiled_fn

    return debug_wrapper


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP REPROS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def maybe_fbcode_instructions() -> str:
    # [FUNCTION DOC] HELPER: Returns a quoted instruction block (how to run the
    # repro under buck, including a TARGETS template with extra deps) when running
    # in fbcode, otherwise an empty string.
    if is_fbcode():
        extra_deps_formatted = "\n".join([f'        "{dep}",' for dep in extra_deps])
        if len(extra_deps_formatted) > 0:
            extra_deps_formatted = "\n" + extra_deps_formatted
        return f"""\
\"\"\"
To run this script in fbcode:
- Create a directory (//scripts/{{your_unixname}}/repro)
- Put this file in scripts/{{your_unixname}}/repro/fx_graph_runnable.py
- Add a TARGETS file that looks like the following
- `buck2 run //scripts/{{your_unixname}}/repro:repro`

NOTE: you may need additional deps to actually be able to run the script.
```
# Contents of TARGETS file
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

python_binary(
    name = "repro",
    main_src = "fx_graph_runnable.py",
    deps = [
        "//caffe2:torch",{extra_deps_formatted}
    ],
)
```
\"\"\"
"""
    else:
        return ""


def generate_custom_triton_kernel(kernel: Any) -> str:
    # [FUNCTION DOC] Returns source text recreating a user Triton kernel. For an
    # autotuner, emits an autotune decorator listing each config (kwargs, warps,
    # stages) with an empty key; a heuristics-wrapped kernel yields only an error
    # message string. Then appends a jit decorator and the kernel source (taken from
    # the kernel or its wrapped function). Pure.
    res = ""
    if isinstance(kernel, Autotuner):
        # pyrefly: ignore [missing-attribute]
        if isinstance(kernel.fn, Heuristics):
            res += "ERROR: Repro will not work as intended, "
            res += "triton.runtime.autotuner.Heuristics is not currently supported\n"
            return res

        config_strs = []
        # pyrefly: ignore [missing-attribute]
        for kernel_config in kernel.configs:
            config_strs.append(f"""triton.Config(
                    {str(kernel_config.kwargs)},
                    num_warps={kernel_config.num_warps},
                    num_stages={kernel_config.num_stages},
                )""")

        config_str = ",".join(config_strs)
        res += textwrap.dedent(f"""
        @triton.autotune(
            configs=[
                {config_str}
            ],
            key=[]
        )
        """).strip()

    # pyrefly: ignore [missing-attribute]
    src_code = kernel.src if isinstance(kernel, JITFunction) else kernel.fn.src
    res += "\n@triton.jit\n"
    res += src_code
    res += "\n"

    return res


def generate_compiler_repro_string(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    *,
    stable_output: bool = False,
    save_dir: str | None = None,
    stable_hash: bool = False,
    has_distributed_ops: bool = False,
) -> str:
    # [FUNCTION DOC] Builds the full text of a repro script body (everything before
    # the main block): env var and config setup, imports (distributed and Triton
    # ones as needed), optional version/system comments, re-registration of user
    # Triton kernels (with referenced constants, imports and nested kernels), the
    # module source for the graph, input-loading code (tensors, symints, None,
    # opaque objects, generators, unsupported), symbol hint definitions prepended at
    # the top, repeat_interleave fixup lines, and creation of the Repro module.
    # Reads the global kernel side table. Returns the script string. Kernels that
    # fail with an attribute error produce an ERROR line in the output instead.
    if save_dir is not None:
        save_dir = normalize_path_separator(save_dir)
    # Add distributed imports if needed
    distributed_imports = ""
    if has_distributed_ops:
        distributed_imports = textwrap.dedent(
            """
import torch.distributed as dist
from torch.testing._internal.distributed.fake_pg import FakeStore
        """
        ).strip()

    triton_imports = ""

    if len(kernel_side_table.id_to_kernel) > 0:
        triton_imports = textwrap.dedent(
            """
import triton
import triton.language as tl
        """
        ).strip()

    model_str = textwrap.dedent(
        f"""
{generate_env_vars_string(stable_output=stable_output)}
import torch
from torch import tensor, device
import torch.fx as fx
from torch._dynamo.testing import rand_strided
import math
from math import inf
import torch._inductor.inductor_prims
{distributed_imports}
{triton_imports}

{generate_config_string(stable_output=stable_output)}

isolate_fails_code_str = None

{extra_imports}

{maybe_fbcode_instructions()}
     """
    )
    model_str += textwrap.dedent(
        """
if "__compile_source__" in globals():
    import inspect as __after_aot_inspect
    import linecache as __after_aot_linecache
    __after_aot_filename = __after_aot_inspect.currentframe().f_code.co_filename
    __after_aot_linecache.cache[__after_aot_filename] = (
        len(__compile_source__),
        None,
        __compile_source__.splitlines(True),
        __after_aot_filename,
    )
"""
    )
    if not stable_output:
        model_str += f"# torch version: {torch.version.__version__}\n"
        if hasattr(torch.version, "cuda"):
            model_str += f"# torch cuda version: {torch.version.cuda}\n"
        if hasattr(torch.version, "git_version"):
            model_str += f"# torch git version: {torch.version.git_version}\n\n\n"
        model_str += _cuda_system_info_comment()

    kernel_side_table_prefix = (
        "torch._higher_order_ops.triton_kernel_wrap.kernel_side_table"
    )

    def get_fn_name(kernel: Any) -> str:
        # [FUNCTION DOC] HELPER: Returns the unqualified function name of a Triton
        # kernel, unwrapping autotuner wrappers.
        fn: Any = kernel if isinstance(kernel, JITFunction) else kernel.fn
        return fn.__name__.split(".")[-1]

    def get_triton_import_line(name: str, val: Any) -> str | None:
        # [FUNCTION DOC] HELPER: Given a global name and value referenced by kernel
        # source, returns an import statement line recreating it if it is a Triton
        # module or object (module alias or from-import, with rename when needed);
        # returns None for the names triton and tl or non-Triton values.
        # User-defined Triton kernels are serialized from their source, not from
        # their original Python module.  If the source references a global
        # imported from Triton, such as `from triton.language.extra import
        # libdevice`, the standalone repro must recreate that import.
        if name in ("triton", "tl"):
            return None

        if isinstance(val, types.ModuleType):
            module_name = val.__name__
            if module_name == "triton" or module_name.startswith("triton."):
                return f"import {module_name} as {name}"
            return None

        module_name = getattr(val, "__module__", None)
        object_name = getattr(val, "__name__", None)
        if (
            isinstance(module_name, str)
            and (module_name == "triton" or module_name.startswith("triton."))
            and isinstance(object_name, str)
        ):
            if name == object_name:
                return f"from {module_name} import {object_name}"
            return f"from {module_name} import {object_name} as {name}"

        return None

    def write_kernel_dependencies(
        kernel: Any,
        written_constexpr_vars: set[str],
        written_nested_kernels: set[str],
        written_triton_imports: set[str],
    ) -> str:
        # [FUNCTION DOC] HELPER: Parses the kernel source to find referenced and
        # called names, then returns text defining needed Triton import lines,
        # simple constants (constexpr, int, float, str, bool globals) and nested
        # JIT kernels (recursively, with dependencies first). Mutates the three
        # passed sets to avoid duplicates and cycles. Returns empty text if the
        # kernel lacks function or source.
        """Write out global triton imports, tl.constexpr vars, and nested kernels."""
        result = ""
        jit_fn = kernel if isinstance(kernel, JITFunction) else kernel.fn
        if not getattr(jit_fn, "fn", None) or not getattr(jit_fn, "src", None):
            return result

        fn_globals = getattr(jit_fn.fn, "__globals__", {})
        src = jit_fn.src
        full_src = src if src.strip().startswith("def ") else "def " + src

        referenced_names: set[str] = set()
        called_names: set[str] = set()
        for node in ast.walk(ast.parse(full_src)):
            if isinstance(node, ast.Name):
                referenced_names.add(node.id)
            if isinstance(node, ast.Call) and isinstance(node.func, ast.Name):
                called_names.add(node.func.id)

        # Only recreate globals that are safe and useful for a standalone
        # Triton repro: imports anchored in the Triton package and simple
        # constants.  Other user globals remain unsupported here.
        for name in sorted(referenced_names):
            if name in written_constexpr_vars:
                continue
            val = fn_globals.get(name)

            import_line = get_triton_import_line(name, val)
            if import_line is not None:
                if import_line not in written_triton_imports:
                    result += import_line + "\n"
                    written_triton_imports.add(import_line)
                continue

            if isinstance(val, TritonConstexpr) and getattr(val, "value", None):
                result += f"{name} = tl.constexpr({val.value})\n"
            elif isinstance(val, (int, float, str, bool)):
                result += f"{name} = {val!r}\n"
            else:
                continue
            written_constexpr_vars.add(name)

        # Write out nested kernel dependencies
        for name in called_names:
            val = fn_globals.get(name)
            if not isinstance(val, JITFunction) or val is jit_fn:
                continue
            nested_fn_name = get_fn_name(val)
            if nested_fn_name in written_nested_kernels:
                continue
            # Mark as written before recursing to prevent cycles
            written_nested_kernels.add(nested_fn_name)
            result += write_kernel_dependencies(
                val,
                written_constexpr_vars,
                written_nested_kernels,
                written_triton_imports,
            )
            result += generate_custom_triton_kernel(val)

        return result

    written_nested_kernels: set[str] = set()
    written_constexpr_vars: set[str] = set()
    written_triton_imports: set[str] = {
        "import triton",
        "import triton.language as tl",
    }

    model_str += f"{kernel_side_table_prefix}.reset_table()\n"

    for idx in kernel_side_table.id_to_kernel:
        kernel = kernel_side_table.get_kernel(idx)

        try:
            model_str += write_kernel_dependencies(
                kernel,
                written_constexpr_vars,
                written_nested_kernels,
                written_triton_imports,
            )
            fn_name = get_fn_name(kernel)

            unique_name = f"{fn_name}_{idx}"

            kernel_code = generate_custom_triton_kernel(kernel)
            kernel_code = kernel_code.replace(
                f"def {fn_name}(", f"def {unique_name}(", 1
            )
            model_str += kernel_code
            model_str += f"{kernel_side_table_prefix}.add_kernel({unique_name})\n"
        except AttributeError as e:
            model_str += "ERROR: Repro will not work as intended, "
            model_str += f"User defined triton kernel exception: {e}\n"

    if len(kernel_side_table.constant_args) > 0:
        model_str += f"{kernel_side_table_prefix}.constant_args={kernel_side_table.constant_args}\n"

    # This string is also emitted as a best-effort trace artifact during
    # normal compilation, so unsupported modules must not mask compile errors.
    model_str += NNModuleToString.convert(gm, allow_unsafe_repr=True)

    writer = InputWriter(save_dir, stable_hash=stable_hash)
    # pyrefly: ignore [implicit-any]
    used_syms = {}

    # Extract from graph placeholders and their corresponding arguments
    placeholder_targets = fx_placeholder_targets(gm)
    for placeholder, arg in zip(placeholder_targets, args):
        if isinstance(arg, (int, torch.SymInt)):
            writer.symint(placeholder, arg)
        elif isinstance(arg, torch.Tensor):
            # TODO: improve these names with FQN
            writer.tensor(placeholder, arg)
        elif arg is None:
            writer.const(placeholder)
        elif isinstance(arg, FakeScriptObject):
            writer.opaque(placeholder, arg.script_class_name)
        elif isinstance(arg, torch._C.Generator):
            writer.generator(placeholder, arg)
        else:
            writer.unsupported(placeholder, arg)

        # Extract symbolic variables from the same arguments

        if (
            isinstance(arg, torch.SymInt)
            # By checking sympy.Symbol, we are excluding any symbolic expressions.
            # TODO: we may need to solve expressions to extract symbol definitions.
            and isinstance(arg.node.expr, sympy.Symbol)
            and arg.node.hint is not None
        ):
            used_syms[str(arg.node)] = arg.node.hint
        elif isinstance(arg, torch.Tensor):
            # Extract symbolic variables from tensor shapes and strides
            for dim in arg.shape:
                if (
                    isinstance(dim, torch.SymInt)
                    and isinstance(dim.node.expr, sympy.Symbol)
                    and dim.node.hint is not None
                ):
                    used_syms[str(dim.node)] = dim.node.hint
            for stride in arg.stride():
                if (
                    isinstance(stride, torch.SymInt)
                    and isinstance(stride.node.expr, sympy.Symbol)
                    and stride.node.hint is not None
                ):
                    used_syms[str(stride.node)] = stride.node.hint
            # Extract symbols from storage nbytes (can be a symbolic expression)
            storage = arg.untyped_storage()
            nbytes = storage.nbytes()
            if isinstance(nbytes, torch.SymInt):
                expr = nbytes.node.expr
                shape_env = nbytes.node.shape_env
                for sym in expr.free_symbols:
                    sym_name = str(sym)
                    if sym_name not in used_syms and shape_env is not None:
                        hint = shape_env.backed_var_to_val.get(sym)
                        if hint is not None:
                            used_syms[sym_name] = int(hint)
    # Add symbolic variable definitions to the top of the generated code
    if used_syms:
        hint_lines = "\n".join(
            f"{name} = {hint}" for name, hint in sorted(used_syms.items())
        )
        model_str = f"{hint_lines}\n\n{model_str}"

    # Add fixup code for repeat_interleave constraints
    # When inputs are regenerated randomly, sum(repeats) != output_size
    # This fixup adjusts the repeats tensor to satisfy the constraint
    constraints = _find_repeat_interleave_constraints(gm)
    if constraints:
        placeholder_to_idx = {name: idx for idx, name in enumerate(placeholder_targets)}
        for repeats_name, output_size_name in constraints:
            repeats_idx = placeholder_to_idx.get(repeats_name)
            output_size_idx = placeholder_to_idx.get(output_size_name)
            if repeats_idx is not None and output_size_idx is not None:
                # Guard with hasattr since NopInputReader doesn't have args
                writer._lines.append(
                    "# Fixup: ensure sum(repeats) == output_size for repeat_interleave"
                )
                writer._lines.append("if hasattr(reader, 'args'):")
                writer._lines.append(f"    _repeats = reader.args[{repeats_idx}]")
                writer._lines.append(
                    f"    _output_size = reader.args[{output_size_idx}]"
                )
                writer._lines.append(
                    "    if isinstance(_repeats, torch.Tensor) and _repeats.dtype == torch.int64:"
                )
                writer._lines.append("        _n = _repeats.numel()")
                writer._lines.append("        _repeats.fill_(_output_size // _n)")
                writer._lines.append("        _repeats[:_output_size % _n] += 1")

    load_args_lines = writer.lines()
    load_args_code = "\n".join(load_args_lines)
    model_str += load_args_code + "\n"

    model_str += "mod = Repro()\n"
    return model_str


def save_graph_repro(
    fd: IO[Any],
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    stable_output: bool = False,
    save_dir: str | None = None,
    command: str = "run",
    accuracy: str | bool | None = None,
    tracing_mode: str | None = None,
    check_str: str | None = None,
    stable_hash: bool = False,
) -> None:
    if any(
        isinstance(arg, torch.fx.experimental._backward_state.BackwardState)
        for arg in args
    ):
        fd.write(
            "Repro is not generated due to existence of BackwardState in graph input"
        )
        return

    if save_dir is not None:
        save_dir = normalize_path_separator(save_dir)

    # Extract distributed info from the graph
    distributed_info = _extract_distributed_info(gm)
    has_distributed_ops = len(distributed_info) > 0

    fd.write(
        generate_compiler_repro_string(
            gm,
            args,
            stable_output=stable_output,
            save_dir=save_dir,
            stable_hash=stable_hash,
            has_distributed_ops=has_distributed_ops,
        )
    )
    if accuracy is None:
        accuracy = "_accuracy" in compiler_name
    if tracing_mode is None:
        tracing_mode = "real"
        if any(
            has_free_symbols(a) for a in args if not isinstance(a, FakeScriptObject)
        ):
            tracing_mode = "symbolic"
    fd.write("if __name__ == '__main__':\n")
    fd.write("    from torch._dynamo.repro.after_aot import run_repro\n")

    # Add distributed initialization before run_repro if needed
    if has_distributed_ops:
        fd.write(
            "    from torch._dynamo.repro.after_aot import setup_fake_process_groups\n"
        )
        fd.write(f"    setup_fake_process_groups({distributed_info!r})\n")

    fd.write(
        f"    with torch.no_grad():\n"
        f"        run_repro(mod, load_args, accuracy={accuracy!r}, command={command!r}, "
        f"save_dir={save_dir!r}, tracing_mode={tracing_mode!r}, check_str={check_str!r})\n"
        f"        # To run it separately, do \n"
        f"        # mod, args = run_repro(mod, load_args, accuracy={accuracy!r}, command='get_args', "
        f"save_dir={save_dir!r}, tracing_mode={tracing_mode!r}, check_str={check_str!r})\n"
        f"        # mod(*args)"
    )

    # Add distributed cleanup after run_repro
    if has_distributed_ops:
        fd.write("\n    dist.destroy_process_group()\n")


def dump_compiler_graph_state(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    *,
    accuracy: str | bool | None = None,
) -> None:
    # [FUNCTION DOC] Saves a repro checkpoint named by node count into the
    # minifier checkpoints directory (created if needed), logs it, and copies it to
    # repro.py in the current directory for convenience (plus a buck target in
    # fbcode). A failed copy due to permissions is only logged as a warning.
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{len(gm.graph.nodes)}.py")
    log.warning(
        "Writing checkpoint with %s nodes to %s", len(gm.graph.nodes), file_name
    )
    with open(file_name, "w") as fd:
        save_graph_repro(
            fd, gm, args, compiler_name, save_dir=subdir, accuracy=accuracy
        )
    curdir = os.getcwd()
    repro_path = os.path.join(curdir, "repro.py")
    try:
        shutil.copyfile(file_name, repro_path)
        log.warning("Copying repro file for convenience to %s", repro_path)
        if use_buck:
            BuckTargetWriter(file_name).write()
    except OSError:
        log.warning("No write permissions for %s", repro_path)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           DUMP MINIFIER
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def dump_to_minify(
    gm: torch.fx.GraphModule, args: Sequence[Any], compiler_name: str
) -> None:
    # [FUNCTION DOC] Renders a repro script with the minify command into memory
    # (inputs saved in the checkpoints directory, created if needed) and hands it
    # to the shared minifier-launcher helper, which writes the launcher file.
    # KNOWN ISSUE: annotated as returning None but returns the helper's result.
    out = io.StringIO()
    # TODO: factor this out
    subdir = os.path.join(minifier_dir(), "checkpoints")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    save_graph_repro(out, gm, args, compiler_name, save_dir=subdir, command="minify")
    return helper_for_dump_minify(out.getvalue())


def isolate_fails(
    fx_g: torch.fx.GraphModule,
    args: Sequence[Any],
    compiler_name: str,
    env: dict[str, Any] | None = None,
    save_dir: str | None = None,
    accuracy: bool | str | None = None,
    tracing_mode: str | None = None,
    check_str: str | None = None,
) -> bool:
    # [FUNCTION DOC] Minifier predicate that tests a graph in a separate process.
    # Writes a minifier-query repro script to a random file in an isolate
    # subdirectory of the current directory, runs it with the current environment
    # overlaid by env (via buck in fbcode, else the current interpreter), echoes its
    # indented stdout and stderr, and returns True when the process exits nonzero.
    if env is None:
        env = {}
    subdir = os.path.join(os.getcwd(), "isolate")
    if not os.path.exists(subdir):
        os.makedirs(subdir, exist_ok=True)
    file_name = os.path.join(subdir, f"{str(uuid.uuid4())[:5]}.py")
    with open(file_name, "w") as fd:
        save_graph_repro(
            fd,
            fx_g,
            args,
            compiler_name,
            save_dir=save_dir,
            command="minifier-query",
            accuracy=accuracy,
            tracing_mode=tracing_mode,
            check_str=check_str,
        )
    # with open(file_name, "r") as fd:
    #     print(fd.read())
    new_env = os.environ.copy()
    new_env = {**new_env, **env}
    if use_buck:
        cmd = BuckTargetWriter(file_name).write(print_msg=False)
    else:
        cmd = [sys.executable, file_name]
    with (
        TemporaryFile() as stdout,
        TemporaryFile() as stderr,
        subprocess.Popen(
            cmd,
            cwd=subdir,
            stdout=stdout,
            stderr=stderr,
            env=new_env,
        ) as p,
    ):
        p.wait()

        stdout.seek(0)
        stderr.seek(0)
        print(
            textwrap.indent(stdout.read().decode("utf-8"), prefix=">>  "),
            file=sys.stdout,
        )
        print(
            textwrap.indent(stderr.read().decode("utf-8"), prefix=">>  "),
            file=sys.stderr,
        )
        # print(f"Isolated test failed - {file_name}")
        return p.returncode != 0


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       MINIFIER TOOLS
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def inductor_fails(
    fx_g: torch.fx.GraphModule, args: Sequence[Any], check_str: str | None = None
) -> bool:
    # [FUNCTION DOC] Minifier predicate: True if compiling and running the graph
    # with Inductor raises an error (matching check_str when given, and printing
    # it). First runs the graph eagerly; if eager fails or returns something other
    # than a flat tuple/list, returns False (not a valid candidate). Syncs the
    # accelerator when inputs are on a non-CPU device so faults surface.
    has_gpu = any(
        isinstance(arg, torch.Tensor) and arg.device.type != "cpu" for arg in args
    )

    def sync() -> None:
        # [FUNCTION DOC] HELPER: Synchronizes the accelerator if inputs were on a
        # non-CPU device and an accelerator is available.
        if has_gpu and torch.accelerator.is_available():
            # Ensures that segfaults are surfaced
            torch.accelerator.synchronize()

    from torch._inductor.compile_fx import compile_fx_inner

    try:
        result = fx_g(*args)
        if not isinstance(result, (tuple, list)):
            raise AssertionError(
                f"Expected result to be a tuple or list, got {type(result)}"
            )
        if any(isinstance(x, (tuple, list)) for x in result):
            raise AssertionError("Result should not contain nested tuples or lists")
    except Exception:
        return False

    sync()

    try:
        compile_args = _get_compile_args(fx_g, args)
        compile_mod = compile_fx_inner(fx_g, compile_args)
        if isinstance(compile_mod, str):
            raise AssertionError("compile_fx_inner should not return a string")
        compile_mod(args)
        sync()
    except Exception as e:
        if check_str is not None and check_str not in repr(e):
            return False
        print(repr(e))
        return True
    return False


def inductor_accuracy_fails(
    fx_g: torch.fx.GraphModule,
    args: Sequence[Any],
    check_str: str | None = None,
    *,
    require_fp64: bool = False,
    ignore_non_fp: bool = False,
) -> bool:
    # [FUNCTION DOC] Minifier predicate for accuracy: returns True if the Inductor
    # compiled graph diverges from eager (versus fp64 reference when require_fp64),
    # optionally ignoring non-floating outputs. Delegates to the forward-only
    # accuracy checker with a compile function that derives compile args from the
    # graph. The check_str argument is accepted but unused.
    from torch._inductor.compile_fx import compile_fx_inner

    def _compile_with_symbolic_args(
        gm: torch.fx.GraphModule, inputs: list[Any]
    ) -> torch.fx.GraphModule:
        # [FUNCTION DOC] HELPER: Compiles gm with Inductor using compile args
        # extracted from placeholder metadata when traced symbolically.
        return compile_fx_inner(gm, _get_compile_args(gm, inputs))  # type: ignore[return-value]

    return backend_aot_accuracy_fails(
        fx_g,
        args,  # type: ignore[arg-type]
        _compile_with_symbolic_args,  # type: ignore[arg-type]
        require_fp64=require_fp64,
        ignore_non_fp=ignore_non_fp,
    )


backend_aot_accuracy_fails = functools.partial(backend_accuracy_fails, only_fwd=True)


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                           REPRO MAIN
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


def _build_symbolic_wrapper(
    mod: nn.Module,
    args: list[Any],
    symint_exprs: dict[int, str],
) -> tuple[nn.Module, list[Any]] | None:
    # [FUNCTION DOC] HELPER: Builds a wrapper module and argument list preserving
    # symbolic relationships among integer args. Splits symint expression strings
    # (by arg index) into free symbols (names like s followed by digits) and derived
    # expressions. Returns None when there are no free symbols. Otherwise returns a
    # wrapper that takes carrier int8 tensors (one per free symbol, sized by the
    # hint, on the first CUDA tensor's device else CPU) followed by non-symint args,
    # reads sizes from carriers, evaluates derived expressions, and calls the inner
    # module. Currently unused by repro_common.
    """Build a wrapper module that preserves symbolic relationships.

    Returns (wrapper_module, wrapper_args) where wrapper_args has carrier
    tensors in place of free-symbol symints, and the wrapper's forward()
    extracts SymInts from carrier.size(0), computes derived expressions,
    and delegates to the inner module.

    Returns None if the expressions don't contain free/derived structure
    (nothing to do).
    """
    import re

    free_sym_re = re.compile(r"^s\d+$")
    free_positions: dict[int, str] = {}
    derived_positions: dict[int, str] = {}

    for idx, expr_str in symint_exprs.items():
        if free_sym_re.fullmatch(expr_str):
            free_positions[idx] = expr_str
        else:
            derived_positions[idx] = expr_str

    if not free_positions:
        return None

    # Order: carrier tensors first, then all non-symint args in original order
    free_order = sorted(free_positions.keys())
    free_sym_names = [free_positions[i] for i in free_order]

    # Precompile derived expressions
    derived_compiled = {
        idx: compile(expr_str, f"<derived arg {idx}>", "eval")
        for idx, expr_str in derived_positions.items()
    }

    n_args = len(args)

    class _SymIntWrapper(nn.Module):
        # [CLASS DOC] HELPER: Module that rebuilds the original integer arguments
        # from carrier tensor sizes and derived expressions, then calls the wrapped
        # inner module. State: inner (the wrapped module).
        def __init__(self, inner: nn.Module) -> None:
            # [FUNCTION DOC] HELPER: Stores the wrapped module.
            super().__init__()
            self.inner = inner

        def forward(self, *flat_args: Any) -> Any:
            # [FUNCTION DOC] HELPER: Takes carrier tensors then other args; maps
            # each free symbol name to its carrier's first-dimension size,
            # evaluates derived expressions (with no builtins) over those symbols,
            # reassembles the original argument order, and returns the inner
            # module's result.
            # KNOWN ISSUE: a statement that indexes derived_positions has no
            # effect (leftover "keep reference" line).
            carriers = flat_args[: len(free_order)]
            other = flat_args[len(free_order) :]

            syms: dict[str, Any] = {}
            for i, sym_name in enumerate(free_sym_names):
                syms[sym_name] = carriers[i].size(0)

            for idx, code in sorted(derived_compiled.items()):
                derived_positions[idx]  # keep reference
                syms[f"_derived_{idx}"] = eval(code, {"__builtins__": {}}, syms)

            other_iter = iter(other)
            rebuilt = []
            for i in range(n_args):
                if i in free_positions:
                    rebuilt.append(syms[free_positions[i]])
                elif i in derived_positions:
                    rebuilt.append(syms[f"_derived_{i}"])
                else:
                    rebuilt.append(next(other_iter))

            return self.inner(*rebuilt)

    # Build wrapper args: carriers + non-symint args
    wrapper_args: list[Any] = []
    for i in free_order:
        hint = int(args[i]) if not isinstance(args[i], int) else args[i]
        device = "cpu"
        for a in args:
            if isinstance(a, torch.Tensor) and a.is_cuda:
                device = a.device
                break
        wrapper_args.append(torch.empty(hint, dtype=torch.int8, device=device))

    for i, arg in enumerate(args):
        if i not in free_positions and i not in derived_positions:
            wrapper_args.append(arg)

    return _SymIntWrapper(mod), wrapper_args


def repro_common(
    options: ReproOptions, mod: nn.Module, load_args: Any
) -> tuple[torch.fx.GraphModule, list[Any]]:
    # [FUNCTION DOC] Shared setup for repro commands. Verifies the module has no
    # parameters (raises AssertionError otherwise), warns about large unserialized
    # buffers and about a missing or newer load_args version, runs load_args once
    # with a no-op reader to count inputs and again with a real reader (progress
    # bar) to materialize args, traces the module to a GraphModule with the
    # requested tracing mode, and turns on the inductor intermediate-hooks config
    # (global side effect). Returns the graph module and args.
    # Invariant for graphs we generate with the repro script
    if any(mod.named_parameters()):
        raise AssertionError("Repro graph should not have any parameters")
    for n, b in mod.named_buffers():
        if b.numel() > MAX_CONSTANT_NUMEL_INLINE:
            log.warning(
                "Constant %s was not serialized, generated random data instead. "
                "If you think this is affecting you, please comment on "
                "https://github.com/pytorch/pytorch/issues/100468",
                n,
            )

    if not hasattr(load_args, "_version"):
        log.warning(
            "load_args does not have a _version attribute, please file a bug to PyTorch "
            "and describe how you generate this repro script"
        )
    else:
        if load_args._version > 0:
            log.warning(
                "load_args is version %s, but this version of PyTorch only supports "
                "version 0.  We will try to run it anyway but there may be an incompatibility; "
                "if so, try upgrading your version of PyTorch.",
                load_args._version,
            )

    nop_reader = NopInputReader()
    load_args(nop_reader)

    with tqdm(desc="Loading inputs", total=nop_reader.total) as pbar:
        input_reader = InputReader(save_dir=options.save_dir, pbar=pbar)
        load_args(input_reader)
        args = input_reader.args

    # Turn mod into a GraphModule the slow way
    # TODO: speed this up
    # NOTE: symint_exprs (from reader.symint(val, expr=...)) are preserved
    # in the repro script for documentation. A future improvement could use
    # _build_symbolic_wrapper to reconstruct algebraic relationships between
    # free and derived symints, but the arg reordering is fragile across
    # different graph structures so we skip it for now.
    mod = make_fx(mod, tracing_mode=options.tracing_mode)(*args)  # type: ignore[arg-type]

    # pyrefly: ignore [bad-assignment]
    torch._inductor.config.generate_intermediate_hooks = True

    return mod, args


def _get_compile_args(mod: torch.fx.GraphModule, args: Sequence[Any]) -> Sequence[Any]:
    # [FUNCTION DOC] HELPER: Chooses what to pass to the Inductor compile entry.
    # Returns the given args unchanged unless the graph has placeholders carrying
    # symbolic integer metadata (symbolic tracing); in that case returns per
    # placeholder the fake metadata value, falling back to the matching arg.
    """Extract FakeTensor/SymInt args from the traced graph for compilation.

    When repro_common traces with tracing_mode='symbolic', the resulting
    GraphModule's placeholder nodes carry FakeTensor/SymInt metadata.
    compile_fx_inner needs these (not the concrete args) so that Inductor
    generates proper symbolic-size bindings in the output code.

    For tracing_mode='real', concrete args are fine — we must NOT extract
    FakeTensor metadata because different nodes may have FakeTensors from
    different FakeTensorModes, causing a FakeTensorMode mismatch assertion
    in Inductor.  We detect symbolic tracing by checking for SymInt values,
    which only exist when tracing_mode='symbolic'.
    """
    placeholders = [n for n in mod.graph.nodes if n.op == "placeholder"]
    if not placeholders:
        return args
    # Only extract metadata if the graph was traced with symbolic mode.
    # SymInt values in placeholder metadata are the reliable indicator —
    # FakeTensors appear in both real and symbolic modes, but only symbolic
    # tracing creates SymInts for integer inputs.
    has_symint = any(isinstance(n.meta.get("val"), torch.SymInt) for n in placeholders)
    if not has_symint:
        return args
    return [n.meta.get("val", a) for n, a in zip(placeholders, args)]


ACCURACY_FAILS: dict[str, Callable[[torch.fx.GraphModule, Any], bool]] = {
    "": inductor_fails,
    # This might look inverted but it's not.  strict_accuracy means "we will
    # minify any time we see anything that diverges", whereas accuracy is more
    # conservative, and will only minify if there is a meaningful fp64
    # divergence
    "accuracy": functools.partial(
        inductor_accuracy_fails, require_fp64=True, ignore_non_fp=True
    ),
    "strict_accuracy": inductor_accuracy_fails,
}


def repro_minifier_query(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    # [FUNCTION DOC] Loads the repro, picks the failure predicate by the accuracy
    # option, and exits the process with status 1 if the predicate says the graph
    # fails, else status 0. Used by isolated minifier subprocesses.
    mod, args = repro_common(options, mod, load_args)
    fail_fn = functools.partial(
        ACCURACY_FAILS[options.accuracy],
        check_str=options.check_str,  # type: ignore[call-arg]
    )
    if fail_fn(mod, args):
        sys.exit(1)
    else:
        sys.exit(0)


def repro_minify(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    # [FUNCTION DOC] Runs the delta-debugging minifier on the loaded repro graph.
    # The failure predicate is either an isolated-subprocess check (pinned to GPU 1
    # if two or more GPUs, else 0, via an environment variable) or the in-process
    # predicate chosen by accuracy mode. Runs with repro_after disabled; state dumps
    # go to the checkpoint writer. Side effects: files on disk, subprocesses.
    from functorch.compile import minifier

    mod, args = repro_common(options, mod, load_args)
    compiler_name = "inductor_accuracy" if options.accuracy != "" else "inductor"

    favored_device = 1 if torch.cuda.device_count() >= 2 else 0
    env_variables = {"CUDA_VISIBLE_DEVICES": str(favored_device)}

    module_fails: Any
    if options.isolate:
        module_fails = functools.partial(
            isolate_fails,
            env=env_variables,
            compiler_name=compiler_name,
            save_dir=options.save_dir,
            accuracy=options.accuracy,
            tracing_mode=options.tracing_mode,
        )
    else:
        module_fails = ACCURACY_FAILS[options.accuracy]

    with config.patch(repro_after=None):
        minifier(
            mod,
            args,
            module_fails=functools.partial(module_fails, check_str=options.check_str),
            dump_state=functools.partial(
                dump_compiler_graph_state, compiler_name=compiler_name
            ),
            save_dir=options.save_dir,
            offload_to_disk=options.offload_to_disk,
            skip_offload=options.skip_saving_eager_intermediates,
            skip_sanity=options.skip_sanity,
            max_granularity=options.max_granularity,
        )


def repro_analyze(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    # [FUNCTION DOC] Accuracy analyzer. Compiles the graph with Inductor, runs it
    # while saving every intermediate tensor to a content store under save_dir
    # (raises RuntimeError if save_dir is unset), optionally re-runs to check
    # Inductor determinism, runs an fp64 copy through an interpreter to save and
    # check fp64 intermediates, and finally interprets the eager graph comparing
    # each saved node against Inductor and fp64 values using the repro tolerance,
    # printing NONDETERMINISTIC or DIVERGED messages. Various skip options disable
    # steps. Raises AssertionError if arg lists are not consumed or a divergence
    # is not logged.
    from torch._inductor.compile_fx import compile_fx_inner
    from torch._inductor.hooks import intermediate_hook

    mod, args = repro_common(options, mod, load_args)

    # TODO: The logic for cloning inputs/models here is intentionally
    # modeled off of run_fwd_maybe_bwd, but arguably it is better not to
    # clone inputs (as you are doubling your effective GPU memory usage).
    # It is certainly faster though!  It probably makes sense to let the
    # user specify the offload strategy.

    compile_mod = copy.deepcopy(mod)
    compile_args = _get_compile_args(compile_mod, args)
    with tqdm(desc="Compiling"):
        compiled = compile_fx_inner(compile_mod, compile_args)
    total = counters["inductor"]["intermediate_hooks"]

    known_names = set()

    def save_hook(name: str, val: Any) -> None:
        # [FUNCTION DOC] HELPER: Intermediate hook that records the node name and,
        # unless skipped, writes the tensor to the content store under the inductor
        # subpath; advances the progress bar.
        known_names.add(name)
        if not options.skip_saving_inductor_intermediates:
            writer.write_tensor(os.path.join("inductor", name), val)
        pbar.update(1)  # type: ignore[has-type]

    # analyze stores and re-reads intermediates on disk, so a save_dir is
    # required; --no-save-dir (save_dir=None) is invalid for this subcommand.
    save_dir = options.save_dir
    if save_dir is None:
        raise RuntimeError("analyze requires a save_dir; do not pass --no-save-dir")
    writer = torch.utils._content_store.ContentStoreWriter(
        save_dir, stable_hash=options.stable_hash
    )
    reader = torch.utils._content_store.ContentStoreReader(save_dir)

    new_args = clone_inputs(args)
    with (
        intermediate_hook(save_hook),
        tqdm(desc="Saving inductor intermediates", total=total) as pbar,
    ):
        if isinstance(compiled, str):
            raise AssertionError("compile_fx_inner should not return a string")
        compiled(new_args)  # type: ignore[arg-type]
        if new_args:
            raise AssertionError("new_args should be empty after compiled() call")

    def compare_tuples(tuple1: tuple[Any], tuple2: tuple[Any]) -> str | None:
        # [FUNCTION DOC] HELPER: Compares two metadata tuples elementwise; returns
        # None if equal, else a text listing each differing pair joined with and.
        diff_indices = [i for i in range(len(tuple1)) if tuple1[i] != tuple2[i]]
        diff_values = [(tuple1[i], tuple2[i]) for i in diff_indices]

        if not diff_values:
            return None
        else:
            return " and ".join(f"{a} != {b}" for a, b in diff_values)

    def check_hook(name: str, val: Any) -> None:
        # [FUNCTION DOC] HELPER: Intermediate hook that compares the tensor's
        # metadata to the one saved earlier under the inductor subpath and prints a
        # nondeterminism message on mismatch; advances the progress bar.
        meta = writer.compute_tensor_metadata(val)
        meta2 = reader.read_tensor_metadata(os.path.join("inductor", name))
        reason = compare_tuples(meta, meta2)
        if reason is not None:
            pbar.write(f"NONDETERMINISTIC INDUCTOR at {name} ({reason})")
        pbar.update(1)

    if not options.skip_check_deterministic:
        new_args = clone_inputs(args)
        with (
            intermediate_hook(check_hook),
            tqdm(desc="Checking inductor determinism", total=total) as pbar,
        ):
            compiled(new_args)  # type: ignore[arg-type]
            if new_args:
                raise AssertionError("new_args should be empty after compiled() call")

    class WriterInterp(fx.Interpreter):
        # [CLASS DOC] HELPER: Graph interpreter that saves the result of every node
        # whose name was seen during the Inductor run into the content store under
        # a given subdirectory. State: subdir.
        def __init__(self, mod: torch.nn.Module, subdir: str) -> None:
            # [FUNCTION DOC] HELPER: Initializes the interpreter and stores subdir.
            super().__init__(mod)
            self.subdir = subdir

        def run_node(self, n: torch.fx.Node) -> Any:
            # [FUNCTION DOC] HELPER: Runs the node; if its name is a known
            # intermediate, writes the result to the store and advances progress.
            # Returns the result.
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                pbar.update(1)
                writer.write_tensor(os.path.join(self.subdir, name), r)
            return r

    # NB: the module cast doesn't actually do anything, since there are no
    # parameters/buffers on the module
    if not options.skip_saving_float64_intermediates:
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))  # type: ignore[arg-type]
        with tqdm(desc="Saving float64 intermediates", total=total) as pbar:
            WriterInterp(new_mod, "float64").boxed_run(new_args)
        if new_args:
            raise AssertionError("new_args should be empty after boxed_run() call")

    class ExactReaderInterp(fx.Interpreter):
        # [CLASS DOC] HELPER: Graph interpreter that checks fp64 determinism by
        # comparing each known node's result metadata to the saved fp64 metadata.
        def run_node(self, n: torch.fx.Node) -> Any:
            # [FUNCTION DOC] HELPER: Runs the node and, for known intermediates,
            # prints a nondeterminism message when metadata differs from the saved
            # fp64 value; returns the result.
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                meta = writer.compute_tensor_metadata(r)
                meta2 = reader.read_tensor_metadata(os.path.join("float64", name))
                reason = compare_tuples(meta, meta2)
                if reason is not None:
                    pbar.write(f"NONDETERMINISTIC FLOAT64 at {name} ({reason})")
                pbar.update(1)
            return r

    # TODO: check eager determinism

    if not options.skip_check_deterministic:
        new_mod, new_args = cast_to_fp64(copy.deepcopy(mod), clone_inputs(args))  # type: ignore[arg-type]
        with tqdm(desc="Checking float64 determinism", total=total) as pbar:
            ExactReaderInterp(new_mod).boxed_run(new_args)
            if new_args:
                raise AssertionError("new_args should be empty after boxed_run() call")

    # Now that we've saved everything, interp through the eager graph
    # and do comparisons
    class ReaderInterp(fx.Interpreter):
        # [CLASS DOC] HELPER: Graph interpreter that runs the eager graph and, for
        # each known node, compares the eager value to the saved Inductor and fp64
        # values, printing DIVERGED messages.
        def run_node(self, n: torch.fx.Node) -> Any:
            # [FUNCTION DOC] HELPER: Runs the node; for known intermediates reads
            # the saved Inductor and fp64 tensors and checks closeness within the
            # configured tolerance (NaNs equal). Raises AssertionError if the check
            # fails without having logged a reason. Returns the eager result.
            r = super().run_node(n)
            name = n.name
            if name in known_names:
                inductor = reader.read_tensor(os.path.join("inductor", name))
                float64 = reader.read_tensor(os.path.join("float64", name))
                logged = False

                def log_error(msg: str, *args: Any) -> None:
                    # [FUNCTION DOC] HELPER: Error callback that marks a divergence
                    # as logged and prints a formatted DIVERGED message.
                    nonlocal logged
                    logged = True
                    pbar.write(f"DIVERGED at {name}: {msg % args}")

                if not same(
                    r,
                    inductor,
                    float64,
                    tol=torch._dynamo.config.repro_tolerance,
                    equal_nan=True,
                    log_error=log_error,
                ):
                    if not logged:
                        raise AssertionError("Divergence detected but not logged")
                pbar.update(1)
            return r

    with tqdm(desc="Checking divergence", total=total) as pbar:
        ReaderInterp(mod).boxed_run(args)
    if args:
        raise AssertionError("args should be empty after boxed_run() call")


def repro_get_args(
    options: ReproOptions, mod: nn.Module, load_args: Any
) -> tuple[torch.fx.GraphModule, list[Any]]:
    # [FUNCTION DOC] Loads the repro and returns the traced graph module and its
    # input args without running anything.
    mod, args = repro_common(options, mod, load_args)
    return mod, args


def repro_run(options: ReproOptions, mod: nn.Module, load_args: Any) -> None:
    # [FUNCTION DOC] Compiles a copy of the loaded graph with Inductor and runs it.
    # With an accuracy mode set, compares against eager and raises AccuracyError
    # on divergence. Otherwise just executes it and synchronizes the accelerator
    # for non-CPU inputs. Raises AssertionError if compilation returns a string.
    from torch._inductor.compile_fx import compile_fx_inner

    mod, args = repro_common(options, mod, load_args)

    compile_mod = copy.deepcopy(mod)
    compile_args = _get_compile_args(compile_mod, args)
    compiled = compile_fx_inner(compile_mod, compile_args)
    if isinstance(compiled, str):
        raise AssertionError("compile_fx_inner should not return a string")

    if options.accuracy != "":
        # We don't really respect --accuracy vs --strict-accuracy here, it
        # seems counterintuitive
        if not same_two_models(
            mod,
            compiled,  # type: ignore[arg-type]
            args,
            only_fwd=True,
            ignore_non_fp=config.repro_ignore_non_fp,
        ):
            raise AccuracyError("Bad accuracy detected")
    else:
        compiled(list(args))
        if (
            any(
                isinstance(arg, torch.Tensor) and arg.device.type != "cpu"
                for arg in args
            )
            and torch.accelerator.is_available()
        ):
            torch.accelerator.synchronize()  # ensure segfaults are surfaced


# TODO: lazily load the inputs or something, rather than cloning them
def run_repro(
    mod: nn.Module,
    load_args: Any,
    *,
    command: str = "run",
    accuracy: bool | str = "",
    save_dir: str | None = None,
    tracing_mode: str | None = None,
    patch_code: str | None = None,
    check_str: str | None = None,
    **kwargs: Any,
) -> Any:
    # [FUNCTION DOC] Command-line entry point embedded in generated repro scripts.
    # Warns about unknown keyword args, normalizes boolean accuracy to the mode
    # strings, builds an argument parser with run, minify, get_args, analyze and
    # minifier-query subcommands (shared accuracy/save-dir/tracing-mode flags plus
    # per-command flags), defaults to the given command when no CLI args exist,
    # and dispatches to the matching repro_* function, returning its result.
    # The patch_code argument is accepted but ignored with a warning.
    for k in kwargs:
        log.warning(
            "Unrecognized kwarg %s; perhaps this repro was made on a newer version of PyTorch",
            k,
        )

    if accuracy is True:
        accuracy = "accuracy"
    elif accuracy is False:
        accuracy = ""

    if patch_code is not None:
        log.warning(
            "patch_code no longer works on this version of PyTorch, silently ignoring"
        )

    parser = argparse.ArgumentParser(
        description=f"""\
An after_aot repro script, typically triggering a bug in PyTorch Inductor.
When run with no arguments, this script defaults to running '{command}'.
Extra flags may be available; to find out more, try '{command} --help'.
There are also alternate subcommands available, see below.

default settings on this script:
  {accuracy=}
  {tracing_mode=}
  {save_dir=}
  {check_str=}
""",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    def common_flags(parser: argparse.ArgumentParser) -> None:
        # [FUNCTION DOC] HELPER: Adds the flags shared by all subcommands to a
        # parser: mutually exclusive no-accuracy, accuracy, strict-accuracy; save
        # directory and its negation; and tracing mode, with defaults taken from
        # the enclosing run_repro arguments.
        accuracy_group = parser.add_mutually_exclusive_group()
        accuracy_group.add_argument(
            "--no-accuracy",
            dest="accuracy",
            action="store_const",
            const="",
            default=accuracy,
            help="do not test accuracy, just run the module and see if it errors",
        )
        accuracy_group.add_argument(
            "--accuracy",
            action="store_const",
            const="accuracy",
            default=accuracy,
            help="""\
test if the RMSE between the compiled module and the fp64 reference is greater
than eager and the fp64 reference. This is usually more reliable than the
standard allclose test, as we expect numeric differences from compiling, often
improving accuracy over eager.  RMSE test allows for compiled module to
diverge greatly from eager, as long as this divergence moves it closer to the
'true' mathematical value of the network.  Caveats: (1) double precision can
still suffer from rounding error, so it is not a perfect reference (see for
example 'Herbie: Automatically Improving Floating Point Accuracy') for
approaches that detect the necessary working precision and compute it in
arbitrary precision floating point; unfortunately, this is not practical for
tensor computation; (2) if there are not enough samples in the output being
compared, we may get unlucky and have an unlucky greater RMSE than eager; this
could be overcome by applying a more rigorous statistical test at some
p-value, which we leave for future work.
""",
        )
        accuracy_group.add_argument(
            "--strict-accuracy",
            dest="accuracy",
            action="store_const",
            const="strict_accuracy",
            default=accuracy,
            help="""\
by default, when doing accuracy minification we will reject reductions which
change the divergence from a floating point divergence to an integral/boolean
divergence.  This is because some operations like ReLU involve temporarily
sharp boundaries that smooth out again afterwards; without requiring
divergence on floating point, the minifier will often fixate on divergent
boolean tensor even though this is not the true source of the divergence.
However, rejecting these reductions makes it more difficult for the minifier
to make progress.  Using this option will let the minifier progress for ALL
divergences--you just might not end up with a useful repro in the end.""",
        )

        parser.add_argument(
            "--save-dir",
            type=str,
            default=save_dir,
            metavar="DIR",
            help="directory where saved inputs live",
        )
        parser.add_argument(
            "--no-save-dir",
            dest="save_dir",
            action="store_const",
            const=None,
            help="don't use any directory for saved inputs",
        )
        parser.add_argument(
            "--tracing-mode",
            type=str,
            metavar="{real,fake,symbolic}",
            default=tracing_mode,
            help="how to trace the repro module into a GraphModule with metadata",
        )

    subparsers = parser.add_subparsers(
        dest="command", metavar="{run,minify,analyze}", required=True
    )

    parser_run = subparsers.add_parser(
        "run",
        help="just run the repro",
    )
    common_flags(parser_run)

    parser_minify = subparsers.add_parser(
        "minify", help="run the minifier on the repro"
    )
    common_flags(parser_minify)
    parser_get_args = subparsers.add_parser("get_args", help="get the args")
    common_flags(parser_get_args)
    parser_minify_isolate = parser_minify.add_mutually_exclusive_group()
    parser_minify_isolate.add_argument(
        "--isolate",
        action="store_true",
        default=True,
        help="run in separate processes to avoid interference (default)",
    )
    parser_minify_isolate.add_argument(
        "--no-isolate",
        dest="isolate",
        action="store_false",
        help="speed up by running all compilation in same process",
    )
    parser_minify.add_argument(
        "--skip-saving-eager-intermediates",
        action="store_true",
        help="skip saving eager intermediates on --minify",
    )
    # TODO: make this an option for --analyze too
    parser_minify.add_argument(
        "--offload-to-disk",
        action="store_true",
        help="during minification, offload delta debugging intermediates to disk.  Use if you're OOMing",
    )
    parser_minify.add_argument(
        "--skip-sanity",
        action="store_true",
        help="skip sanity check at beginning of minification on original graph",
    )
    parser_minify.add_argument(
        "--max-granularity",
        type=int,
        default=None,
        help="start at this granularity and work down; must be power of 2",
    )
    parser_minify.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )

    parser_analyze = subparsers.add_parser(
        "analyze", help="run the accuracy analyzer on the repro"
    )
    common_flags(parser_analyze)
    parser_analyze.add_argument(
        "--skip-saving-inductor-intermediates",
        action="store_true",
        help="skip saving inductor intermediates on --analyze",
    )
    parser_analyze.add_argument(
        "--skip-saving-float64-intermediates",
        action="store_true",
        help="skip saving float64 intermediates",
    )
    parser_analyze.add_argument(
        "--skip-check-deterministic",
        action="store_true",
        help="skip checking that the network is deterministic",
    )
    parser_analyze.add_argument(
        "--stable-hash",
        action="store_true",
        help="use SHA-1 checksum instead of fast (but possibly unsound) hash",
    )

    # Run the repro in the context of minification, inverting exit code meaning
    parser_minifier_query = subparsers.add_parser(
        "minifier-query",
    )
    common_flags(parser_minifier_query)
    parser_minifier_query.add_argument(
        "--check-str",
        type=str,
        default=check_str,
        help="require minified program to fail with error containing this string",
    )

    args = None
    if len(sys.argv) <= 1:
        args = [command, *sys.argv[1:]]

    options = typing.cast("ReproOptions", parser.parse_args(args))
    COMMAND_FNS = {
        "minify": repro_minify,
        "analyze": repro_analyze,
        "minifier-query": repro_minifier_query,
        "run": repro_run,
        "get_args": repro_get_args,
    }
    return COMMAND_FNS[options.command](options, mod, load_args)
