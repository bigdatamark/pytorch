# mypy: allow-untyped-defs
# [PACKAGE DOC]
# Public entry point of TorchInductor. Exposes thin wrappers: compile (FX graph
# to compiled callable), aoti_compile_and_package / aoti_load_package / aot_compile
# (ahead-of-time compilation to .pt2 packages), standalone_compile and
# compile_to_python / load_from_python (re-exported from standalone_compile),
# list_mode_options / list_options, and cudagraph_mark_step_begin.
# Heavy submodules (compile_fx, package, debug, cudagraph_trees) are imported
# lazily inside functions. Import-time side effects: creates the module logger
# and defines lite_mode_options. Also defines the private _CudagraphAnnotation dataclass.
from __future__ import annotations

import dataclasses
import io
import logging
import os
from typing import Any, IO, Literal, Optional, TYPE_CHECKING, Union

import torch.fx

from .standalone_compile import (
    compile_to_python,
    CompiledArtifact,
    DynamicShapesType,
    load_from_python,
)


if TYPE_CHECKING:
    from torch._inductor.utils import InputType
    from torch._subclasses import FakeTensorMode
    from torch.export import ExportedProgram
    from torch.export.pt2_archive._package import AOTICompiledModel
    from torch.export.pt2_archive._package_weights import Weights
    from torch.types import FileLike

__all__ = [
    "compile",
    "compile_to_python",
    "load_from_python",
    "list_mode_options",
    "list_options",
    "cudagraph_mark_step_begin",
    "standalone_compile",
]


log = logging.getLogger(__name__)


def compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    options: dict[str, Any] | None = None,
):
    # [FUNCTION DOC]
    # Compiles an FX graph module with TorchInductor by delegating to compile_fx
    # (imported lazily). Args: gm, example_inputs (list of tensors), options
    # (optional mapping of inductor config overrides passed as config_patches).
    # Returns a callable with the same behavior as gm. Errors come from compile_fx.
    """
    Compile a given FX graph with TorchInductor.  This allows compiling
    FX graphs captured without using TorchDynamo.

    Args:
        gm: The FX graph to compile.
        example_inputs:  List of tensor inputs.
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Callable with same behavior as gm but faster.
    """
    from .compile_fx import compile_fx

    return compile_fx(gm, example_inputs, config_patches=options)


def aoti_compile_and_package(
    exported_program: ExportedProgram,
    _deprecated_unused_args=None,
    _deprecated_unused_kwargs=None,
    *,
    package_path: FileLike | None = None,
    inductor_configs: dict[str, Any] | None = None,
) -> str:
    # [FUNCTION DOC]
    # Compiles an ExportedProgram with AOTInductor and packages the result into a
    # .pt2 artifact. Validates: input must be an ExportedProgram (ValueError
    # otherwise); example_inputs must be set (RuntimeError); package_path must be
    # None, a writable+seekable buffer, or a path ending in .pt2 (AssertionError);
    # the aot_inductor.output_path config must not be set (RuntimeError). Warns if
    # the deprecated args/kwargs are passed. Forces the aot_inductor.package config
    # on (mutating the passed config mapping) and runs the inner compile through
    # the minifier wrapper. Returns the package path.
    """
    Compiles the exported program with AOTInductor, and packages it into a .pt2
    artifact specified by the input package_path. To load the package, you can
    call ``torch._inductor.aoti_load_package(package_path)``.

    An example usage is as follows:

    .. code-block:: python

        ep = torch.export.export(M(), ...)
        aoti_file = torch._inductor.aoti_compile_and_package(
            ep, package_path="my_package.pt2"
        )
        compiled_model = torch._inductor.aoti_load_package("my_package.pt2")

    To compile and save multiple models into a single ``.pt2`` artifact, you can do
    the following:

    .. code-block:: python

        ep1 = torch.export.export(M1(), ...)
        aoti_file1 = torch._inductor.aot_compile(
            ep1, ..., options={"aot_inductor.package": True}
        )
        ep2 = torch.export.export(M2(), ...)
        aoti_file2 = torch._inductor.aot_compile(
            ep2, ..., options={"aot_inductor.package": True}
        )

        from torch._inductor.package import package_aoti, load_package

        package_aoti("my_package.pt2", {"model1": aoti_file1, "model2": aoti_file2})

        compiled_model1 = load_package("my_package.pt2", "model1")
        compiled_model2 = load_package("my_package.pt2", "model2")

    Args:
        exported_program: An exported program created through a call from torch.export
        package_path: Optional specified path to the generated .pt2 artifact.
        inductor_configs: Optional dictionary of configs to control inductor.

    Returns:
        Path to the generated artifact
    """
    from torch.export import ExportedProgram

    from .debug import aot_inductor_minifier_wrapper

    if not isinstance(exported_program, ExportedProgram):
        raise ValueError("Only ExportedProgram is supported")

    if exported_program.example_inputs is None:
        raise RuntimeError(
            "exported_program.example_inputs is required to be set in order "
            "for AOTInductor compilation."
        )

    if _deprecated_unused_args is not None or _deprecated_unused_kwargs is not None:
        log.warning(
            "You no longer need to specify args/kwargs to aoti_compile_and_package "
            "as we can get this information from exported_program.example_inputs."
        )

    if not (
        package_path is None
        or (
            isinstance(package_path, (io.IOBase, IO))
            and package_path.writable()
            and package_path.seekable()
        )
        or (
            isinstance(package_path, (str, os.PathLike))
            and os.fspath(package_path).endswith(".pt2")
        )
    ):
        raise AssertionError(
            f"Expect package path to be a file ending in .pt2, is None, or is a buffer. Instead got {package_path}"
        )

    inductor_configs = inductor_configs or {}
    inductor_configs["aot_inductor.package"] = True

    if inductor_configs.get("aot_inductor.output_path"):
        raise RuntimeError(
            "Please pass in a package path to aot_inductor_compile() instead "
            "of setting the aot_inductor.output_path config."
        )

    # a wrapper around aoti_compile_and_package_inner.
    return aot_inductor_minifier_wrapper(
        _aoti_compile_and_package_inner,
        exported_program,
        package_path=package_path,
        inductor_configs=inductor_configs,
    )


def _aoti_compile_and_package_inner(
    gm: torch.nn.Module,
    # flat_example_inputs: List[Any],
    args: tuple[Any],
    kwargs: dict[str, Any] | None = None,
    *,
    load_and_run: bool = False,
    check_accuracy: str | None = None,
    package_path: str | io.BytesIO | None = None,
    inductor_configs: dict[str, Any] | None = None,
):
    # [FUNCTION DOC] HELPER: Inner worker for aoti_compile_and_package, also used by the
    # minifier. Requires gm to be an fx GraphModule (AssertionError otherwise) and kwargs
    # to be empty when check_accuracy is set. Runs aot_compile, then packages the files
    # into package_path (derived from the first .so, or .cpp, file name plus .pt2 when
    # not given). If load_and_run or check_accuracy is set, loads the package and either
    # runs it or compares it to gm, raising AccuracyError on mismatch. Returns the path.
    """
    See docstring for aoti_compile_and_package.

    If `load_and_run` is True, this function will load the compiled model and run it.
    This is for the minifier to check the correctness of the compiled model.

    If `check_accuracy` is set, this function will check the accuracy of the compiled
    model against gm. kwargs must be None if check_accuracy is set.
    "strict_accuracy" means "we will minify any time we see anything that
     diverges", whereas "accuracy" is more conservative, and will only minify if there
     is a meaningful fp64 divergence
    """

    if check_accuracy:
        if not (kwargs is None or len(kwargs) == 0):
            raise AssertionError(
                "when checking for accuracy, the inputs must have been flattened and kwargs is None"
            )

    from .package import package_aoti

    if not isinstance(gm, torch.fx.GraphModule):
        raise AssertionError(f"expected torch.fx.GraphModule, got {type(gm)}")

    kwargs = kwargs or {}

    aoti_files = aot_compile(gm, args, kwargs, options=inductor_configs)
    if not isinstance(aoti_files, list):
        raise AssertionError(
            f"expected aoti_files to be a list, got {type(aoti_files)}"
        )

    if package_path is None:
        path = [
            os.path.splitext(file)[0]
            for file in aoti_files
            if isinstance(file, str) and os.path.splitext(file)[1] == ".so"
        ]
        if len(path) == 0:
            path = [
                os.path.splitext(file)[0]
                for file in aoti_files
                if isinstance(file, str) and os.path.splitext(file)[1] == ".cpp"
            ]
        package_path = path[0] + ".pt2"

    res = package_aoti(package_path, aoti_files)
    if res != package_path:
        raise AssertionError(
            f"expected res == package_path, got {res} != {package_path}"
        )

    if load_and_run or check_accuracy:
        compiled_model = aoti_load_package(package_path)
        if check_accuracy:
            from torch._dynamo.debug_utils import AccuracyError, same_two_models

            # This might look inverted but it's not.  strict_accuracy means "we will
            # minify any time we see anything that diverges", whereas accuracy is more
            # conservative, and will only minify if there is a meaningful fp64
            # divergence
            not_strict_accuracy = check_accuracy == "accuracy"
            if not same_two_models(
                gm,
                compiled_model,  # type: ignore[arg-type]
                args,
                only_fwd=True,
                require_fp64=not_strict_accuracy,
                ignore_non_fp=not_strict_accuracy,
            ):
                raise AccuracyError("Bad accuracy detected")
        else:
            compiled_model(*args, **kwargs)

    return package_path


def aoti_load_package(
    path: FileLike, run_single_threaded: bool = False, device_index: int = -1
) -> AOTICompiledModel:
    # [FUNCTION DOC]
    # Loads the default model from a .pt2 package by delegating to
    # torch._inductor.package.load_package, forwarding run_single_threaded and
    # device_index (-1 means the default device). Returns the compiled model.
    """
    Loads the model from the PT2 package.

    If multiple models were packaged into the PT2, this will load the default
    model. To load a specific model, you can directly call the load API

    .. code-block:: python

        from torch._inductor.package import load_package

        compiled_model1 = load_package("my_package.pt2", "model1")
        compiled_model2 = load_package("my_package.pt2", "model2")

    Args:
        path: Path to the .pt2 package
        run_single_threaded (bool): Whether the model should be run without
            thread synchronization logic. This is useful to avoid conflicts with
            CUDAGraphs.
        device_index (int): The index of the device to which the PT2 package is
            to be loaded. By default, `device_index=-1` is used, which corresponds
            to the device `cuda` when using CUDA. Passing `device_index=1` would
            load the package to `cuda:1`, for example.
    """
    from torch._inductor.package import load_package

    return load_package(
        path, run_single_threaded=run_single_threaded, device_index=device_index
    )


def aot_compile(
    gm: torch.fx.GraphModule,
    args: tuple[Any, ...],
    kwargs: dict[str, Any] | None = None,
    *,
    options: dict[str, Any] | None = None,
) -> str | list[str | Weights] | torch.fx.GraphModule:
    # [FUNCTION DOC]
    # Ahead-of-time compiles an FX graph module with Inductor. Side effect: if gm
    # has a _guards_fn attribute, its call node is erased from the graph, the
    # attribute is deleted and gm is recompiled (mutating the caller's module).
    # Flattens args/kwargs into flat example inputs and options, then calls
    # compile_fx_aot under the compiling-state context. Returns a shared library
    # path, or a list of generated files when aot_inductor.package is enabled.
    """
    Ahead-of-time compile a given FX graph with TorchInductor into a shared library.

    Args:
        gm: The FX graph to compile.
        args:  Example arguments
        kwargs: Example keyword arguments
        options:  Optional dict of config options.  See `torch._inductor.config`.

    Returns:
        Path to the generated shared library, or a list of files generated by
        AOTI if aot_inductor.package=True.
        TODO: make it return a list by default
    """
    from .compile_fx import _aoti_flatten_inputs, compile_fx_aot

    if hasattr(gm, "_guards_fn"):
        # Do not compile the guards function, since it may contain checks
        # that are not currently supported by AOTI. In particular, non-Tensor
        # arguments are converted to None and will fail specialization checks.
        node = next(iter(gm.graph.find_nodes(op="call_module", target="_guards_fn")))
        gm.graph.erase_node(node)
        delattr(gm, "_guards_fn")
        gm.recompile()

    flat_example_inputs, options = _aoti_flatten_inputs(
        gm, args, kwargs, options=options
    )
    from torch._export.utils import _compiling_state_context

    with _compiling_state_context():
        return compile_fx_aot(
            gm,
            flat_example_inputs,  # type: ignore[arg-type]
            config_patches=options,
        )


lite_mode_options = {
    # Fallback by default unless users explicitly annotated with
    # regional inductor compile.
    "fallback_by_default": True,
    "selective_decompose": True,
    # Disable reorder optimizations
    "reorder_for_peak_memory": False,
    "reorder_for_compute_comm_overlap": False,
    "triton.reorder_for_reducing_graph_partitions": False,
    # Disable pre-, joint-, post-grad passes
    "use_pre_grad_passes": False,
    "use_joint_graph_passes": False,
    "use_post_grad_passes": False,
    # Disable dead code elimination (dce) and buffer reuse
    "use_dce": False,
    "allow_buffer_reuse": False,
}


def list_mode_options(
    mode: str | None = None, dynamic: bool | None = None
) -> dict[str, Any]:
    # [FUNCTION DOC]
    # Returns the config-override mapping for a torch.compile mode (default, lite,
    # reduce-overhead, max-autotune-no-cudagraphs, max-autotune), or the mapping of
    # all modes when mode is empty. Raises RuntimeError for an unknown mode.
    # The dynamic argument is accepted but unused.
    r"""Returns a dictionary describing the optimizations that each of the available
    modes passed to `torch.compile()` performs.

    Args:
        mode (str, optional): The mode to return the optimizations for.
        If None, returns optimizations for all modes
        dynamic (bool, optional): Whether dynamic shape is enabled.

    Example::
        >>> torch._inductor.list_mode_options()
    """

    mode_options: dict[str, dict[str, bool]] = {
        "default": {},
        # lite backend for opt-in optimizations
        "lite": lite_mode_options,
        # enable cudagraphs
        "reduce-overhead": {
            "triton.cudagraphs": True,
        },
        # enable max-autotune
        "max-autotune-no-cudagraphs": {
            "max_autotune": True,
            "coordinate_descent_tuning": True,
        },
        # enable max-autotune
        # enable cudagraphs
        "max-autotune": {
            "max_autotune": True,
            "triton.cudagraphs": True,
            "coordinate_descent_tuning": True,
        },
    }
    try:
        return mode_options[mode] if mode else mode_options
    except KeyError as e:
        raise RuntimeError(
            f"Unrecognized mode={mode}, should be one of: {', '.join(mode_options.keys())}"
        ) from e


def list_options() -> list[str]:
    # [FUNCTION DOC]
    # Returns the names of all inductor config options, taken from a copy of the
    # current inductor config.
    # KNOWN ISSUE: the docstring says it returns a dictionary, but it returns a list of names.
    r"""Returns a dictionary describing the optimizations and debug configurations
    that are available to `torch.compile()`.

    The options are documented in `torch._inductor.config`.

    Example::

        >>> torch._inductor.list_options()
    """

    from torch._inductor import config

    current_config: dict[str, Any] = config.get_config_copy()

    return list(current_config.keys())


def cudagraph_mark_step_begin():
    # [FUNCTION DOC]
    # Marks the start of a new training/inference iteration for CUDA graph trees
    # by calling cudagraph_trees.mark_step_begin (global cudagraph tree state change).
    "Indicates that a new iteration of inference or training is about to begin."
    from .cudagraph_trees import mark_step_begin

    mark_step_begin()


def standalone_compile(
    gm: torch.fx.GraphModule,
    example_inputs: list[InputType],
    *,
    dynamic_shapes: DynamicShapesType = "from_graph",
    options: dict[str, Any] | None = None,
    aot: bool = False,  # AOT mode, which uses BundledAOTAutogradCache
    donate_graph_module: bool = False,
    fake_mode: FakeTensorMode | None = None,
) -> CompiledArtifact:
    # [FUNCTION DOC]
    # Precompiles a graph module with Inductor into a CompiledArtifact that can be
    # saved and reloaded. Normalizes missing options to an empty mapping and
    # forwards everything (dynamic_shapes mode, aot flag, donate_graph_module,
    # optional fake_mode) to standalone_compile.standalone_compile.
    """
    Precompilation API for inductor.

    .. code-block:: python

        compiled_artifact = torch._inductor.standalone_compile(gm, args)
        compiled_artifact.save(path=path, format="binary")

        # Later on a new process
        loaded = torch._inductor.CompiledArtifact.load(path=path, format="binary")
        compiled_out = loaded(*args)

    Args:
        gm: Graph Module
        example_inputs: Inputs for the graph module
        dynamic_shapes: If "from_graph" (default), we will use the dynamic
            shapes in the passed-in graph module.
            If "from_tracing_context", we use the dynamic shape info in the
            ambient tracing context.
            If "from_example_inputs", we will specialize the graph on the
            example_inputs.
        options: Inductor compilation options
        donate_graph_module: If True, standalone_compile takes ownership of
            the graph module and may mutate it, avoiding an internal deepcopy.
            Defaults to False for backwards compatibility.
        fake_mode: Optional FakeTensorMode to use when
            dynamic_shapes="from_example_inputs". The mode must have a ShapeEnv.
            When omitted, a fresh FakeTensorMode is created as before.

    Returns:
        CompiledArtifact that can be saved to disk or invoked directly.
    """
    from .standalone_compile import standalone_compile

    options = options if options else {}
    return standalone_compile(
        gm,
        example_inputs,
        dynamic_shapes=dynamic_shapes,
        options=options,
        aot=aot,
        donate_graph_module=donate_graph_module,
        fake_mode=fake_mode,
    )


@dataclasses.dataclass
class _CudagraphAnnotation:
    # [CLASS DOC] HELPER: Dataclass holding optional per-region cudagraph flags:
    # fwd and bwd (True/False/None meaning forward/backward cudagraph choice, or unset).
    fwd: bool | None
    bwd: bool | None
