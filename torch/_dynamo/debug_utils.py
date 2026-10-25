# [MODULE DOC]
# Debugging support for Dynamo/Inductor repro and minifier tooling: serializes
# fx GraphModules into runnable source (NNModuleToString), records/replays graph
# inputs (InputWriter/InputReader/NopInputReader), checks accuracy of a compiled
# backend against eager (same_two_models, backend_accuracy_fails), emits env var
# and config prelude strings for repro scripts, writes the minifier launcher
# file, parses printed AOT graphs into runnable kwargs, and offers a cProfile
# decorator. Optional Buck helpers (BuckTargetWriter) are used only in fbcode.
# Import-time side effects: imports torch._inductor.config, detects fbcode
# (use_buck) and, if so, imports libfb build info and builds extra_deps,
# cur_target and extra_imports module globals.
"""
Debug utilities for TorchDynamo compilation and execution.

This module provides various debugging tools and utilities for TorchDynamo, including:

- Minification support for reducing test cases while preserving bugs
- Input/output handling via InputReader and InputWriter for reproducible testing
- Accuracy checking between original and compiled models
- Neural network module string conversion via NNModuleToString
- Profiling tools and system information collection
- Buck build system integration for Meta-internal testing

Key classes:
- InputReader/InputWriter: Handle serialization of model inputs/outputs
- NNModuleToString: Converts nn.Modules to string representations
- BuckTargetWriter: Manages Buck build system integration
"""

from __future__ import annotations

import atexit
import copy
import cProfile
import functools
import getpass
import inspect
import itertools
import logging
import os
import re
import subprocess
import sys
import tempfile
import textwrap
from collections import Counter
from importlib import import_module
from typing import Any, cast, TYPE_CHECKING, TypeVar

import torch
import torch._prims_common as utils
import torch._subclasses.meta_utils
from torch import Tensor
from torch._dynamo.testing import rand_strided
from torch._inductor.cpp_builder import normalize_path_separator
from torch._prims_common import is_float_dtype
from torch.multiprocessing.reductions import StorageWeakRef
from torch.utils._config_module import ConfigModule
from torch.utils._content_store import ContentStoreReader, ContentStoreWriter

from . import config
from .utils import clone_inputs, get_debug_dir, warn_once


if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch.hub import tqdm
    from torch.storage import UntypedStorage


log = logging.getLogger(__name__)

T = TypeVar("T")


inductor_config = import_module("torch._inductor.config")
use_buck = inductor_config.is_fbcode()

if use_buck:
    import libfb.py.build_info


# pyrefly: ignore [implicit-any]
extra_deps = []
extra_imports = ""
cur_target = ""
if use_buck:
    extra_deps = [
        "//caffe2/torch/fb/sparsenn:sparsenn_operators_gpu",
        "//caffe2/torch/fb/sparsenn:sparsenn_operators",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops_cpu",
        "//deeplearning/fbgemm/fbgemm_gpu:sparse_ops",
    ]
    cur_target = libfb.py.build_info.BuildInfo.get_build_rule().replace("fbcode:", "//")  # type: ignore[possibly-undefined]
    # Preload common fbcode custom-op libraries so repros that use those ops
    # work out of the box. Best-effort: a repro whose graph doesn't use these
    # ops (or that is run outside a buck target linking them) must not fail
    # just because the library isn't present.
    _extra_deps_list = "\n".join(f'    "{x}",' for x in extra_deps)
    extra_imports = (
        f"for _extra_dep in [\n{_extra_deps_list}\n]:\n"
        "    try:\n"
        "        torch.ops.load_library(_extra_dep)\n"
        "    except OSError:\n"
        "        pass\n"
    )


BUCK_CMD_PREFIX = ["buck2", "run", "@mode/dev-nosan"]


class BuckTargetWriter:
    # [CLASS DOC] Generates a Buck TARGETS python_binary rule for a single
    # repro/minifier script so it can be run with buck2. Attributes: subdir and
    # py_file (location of script), target (file name without extension), path
    # (fbcode-relative dotted main module), cmd_line_path (buck target label).
    # Only meaningful for scripts located under an fbcode checkout.
    def __init__(self, filename: str) -> None:
        # [FUNCTION DOC] Splits the absolute path of filename into directory and
        # file, and derives the target name, the dotted main_module path (text
        # after "fbcode.") and the buck command line target. Assumes the path
        # contains "fbcode/"; otherwise the derived paths are garbled (no error).
        self.subdir, self.py_file = os.path.split(os.path.abspath(filename))
        self.target = self.py_file.replace(".py", "")

        # Get main_module path from fbcode
        self.path = f"{self.subdir.replace('/', '.')}.{self.target}"
        self.path = self.path[self.path.find("fbcode.") :]
        self.path = self.path[7:]

        # Get cmd line path
        tmp = self.subdir
        tmp = tmp[tmp.find("fbcode/") :][7:]
        self.cmd_line_path = f"//{tmp}:{self.target}"

    def build(self) -> str:
        # [FUNCTION DOC] Returns the text of a TARGETS file defining a
        # python_binary for the script, with torch/functorch/triton deps, the
        # current build rule, and the global extra_deps as cpp_deps.
        extra_cpp_deps = "\n".join([f'        "{x}",' for x in extra_deps])
        return textwrap.dedent(
            f"""
load("@fbcode_macros//build_defs:python_binary.bzl", "python_binary")

python_binary(
    name="{self.target}",
    srcs = ["{self.py_file}"],
    compile = False,
    deps = [
        "//caffe2:torch",
        "//caffe2:libtorch",
        "//caffe2/functorch:functorch",
        "//triton:triton",
        "{cur_target}",
    ],
    cpp_deps = [
{extra_cpp_deps}
    ],
    main_module = "{self.path}",
    par_style = "xar",
)
"""
        )

    def write(self, print_msg: bool = True) -> list[str]:
        # [FUNCTION DOC] Writes the generated rule to a file named TARGETS in the
        # script directory (overwriting any existing file; OSError propagates).
        # If print_msg is true, logs a warning with the buck command to run.
        # Returns the buck command as a list of tokens.
        target_file = os.path.join(self.subdir, "TARGETS")
        with open(target_file, "w") as fd:
            fd.write(self.build())
        # log.warning("Wrote isolation TARGETS file at %s", target_file)
        cmd_split = BUCK_CMD_PREFIX + [self.cmd_line_path]
        if print_msg:
            log.warning(
                "Found an example that reproduces the error. Run this cmd to repro - %s",
                " ".join(cmd_split),
            )
        return cmd_split


def minifier_dir() -> str:
    # [FUNCTION DOC] Returns the directory used for minifier output (a
    # "minifier" subdirectory of the Dynamo debug dir), creating it if missing.
    # Side effect: creates directories on disk.
    # KNOWN ISSUE: os.path.join never returns None, so the temp-dir fallback
    # branch is dead code.
    path = os.path.join(get_debug_dir(), "minifier")
    if path is None:
        path = f"{tempfile.gettempdir()}/minifier_{getpass.getuser()}"
    if not os.path.exists(path):
        os.makedirs(path, exist_ok=True)
    return path


MAX_CONSTANT_NUMEL_INLINE = 4


class UnsupportedNNModuleError(AssertionError):
    # [CLASS DOC] Raised by NNModuleToString.convert when a child module cannot
    # be expressed as constructor source. Subclasses AssertionError; no state.
    pass


class NNModuleToString:
    # [CLASS DOC] Namespace of static helpers that turn an fx GraphModule into
    # the source text of a torch.nn.Module class named Repro (plus classes for
    # nested subgraph GraphModules). Class attributes: fake_quant_modules,
    # qat_conv_modules, qat_linear_modules (fully qualified type names handled
    # by special constructors), default_qat_qconfig_backends (backends tried when
    # matching a default QAT qconfig), safe_reprs (module types whose repr is a
    # valid constructor call). Entry points: can_convert_to_string and convert.
    fake_quant_modules = {
        "torch.ao.quantization.fake_quantize.FakeQuantize",
        "torch.ao.quantization.fake_quantize.FusedMovingAvgObsFakeQuantize",
    }

    qat_conv_modules = {
        "torch.ao.nn.qat.modules.conv.Conv1d",
        "torch.ao.nn.qat.modules.conv.Conv2d",
        "torch.ao.nn.qat.modules.conv.Conv3d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn1d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn2d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBn3d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU1d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU2d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvBnReLU3d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU1d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU2d",
        "torch.ao.nn.intrinsic.qat.modules.conv_fused.ConvReLU3d",
    }

    qat_linear_modules = {
        "torch.ao.nn.qat.modules.linear.Linear",
        "torch.ao.nn.intrinsic.qat.modules.linear_relu.LinearReLU",
    }

    # x86 currently aliases fbgemm in repr, so emit fbgemm as the canonical
    # constructor for both until those default qconfigs diverge.
    default_qat_qconfig_backends = ("fbgemm", "x86", "qnnpack", "onednn")

    safe_reprs = [
        torch.nn.Linear,
        torch.nn.Conv1d,
        torch.nn.Conv2d,
        torch.nn.Conv3d,
        torch.nn.BatchNorm1d,
        torch.nn.BatchNorm2d,
        torch.nn.BatchNorm3d,
        torch.nn.LayerNorm,
        torch.nn.Dropout,
        torch.nn.Softmax,
        torch.nn.ReLU,
        torch.nn.GELU,
        torch.nn.Identity,
        torch.nn.MaxPool2d,
        torch.nn.Embedding,
        torch.nn.Tanh,
        torch.nn.ConvTranspose1d,
        torch.nn.GLU,
        torch.nn.LSTM,
        torch.nn.Flatten,
        torch.nn.AdaptiveAvgPool2d,
    ]

    @staticmethod
    def can_convert_to_string(gm: torch.fx.GraphModule) -> bool:
        # [FUNCTION DOC] Returns true if every child module of gm is either a
        # nested GraphModule (serialized recursively) or has a safe constructor
        # string. Otherwise logs a warning listing the offending modules and
        # returns false. Does not mutate gm.
        cant_convert = set()
        for _, module in gm.named_children():
            # Nested GraphModules (e.g. torch.cond / torch.while_loop subgraphs)
            # are serialized recursively by convert().
            if isinstance(module, torch.fx.GraphModule):
                continue
            if NNModuleToString._module_constructor_string(module) is None:
                cant_convert.add(module)

        if len(cant_convert) > 0:
            log.warning(
                "Cannot safely convert some modules to strings - %s", cant_convert
            )
            return False
        return True

    @staticmethod
    def _module_type_name(module: torch.nn.Module) -> str:
        # [FUNCTION DOC] HELPER: Returns the fully qualified "module.ClassName"
        # string of the module's type.
        return f"{type(module).__module__}.{type(module).__name__}"

    @staticmethod
    def _type_name(type_: type[object]) -> str:
        # [FUNCTION DOC] HELPER: Returns the fully qualified "module.ClassName"
        # string of a type object.
        return f"{type_.__module__}.{type_.__name__}"

    @staticmethod
    def _fake_quant_constructor(module: torch.nn.Module) -> str | None:
        # [FUNCTION DOC] HELPER: For FakeQuantize-style modules, returns a
        # constructor call string using the observer class, quant_min/max (scaled
        # back up when the observer used reduce_range), dtype, qscheme,
        # is_dynamic and, when present, reduce_range and ch_axis. Returns None
        # for any other module type.
        module_type = NNModuleToString._module_type_name(module)
        if module_type not in NNModuleToString.fake_quant_modules:
            return None

        observer = module.activation_post_process  # type: ignore[attr-defined]
        quant_min = cast(int, module.quant_min)  # type: ignore[attr-defined]
        quant_max = cast(int, module.quant_max)  # type: ignore[attr-defined]
        if getattr(observer, "reduce_range", False):
            quant_min *= 2
            quant_max = quant_max * 2 + 1
        args = [
            f"observer={NNModuleToString._type_name(type(observer))}",
            f"quant_min={quant_min!r}",
            f"quant_max={quant_max!r}",
            f"dtype={module.dtype!r}",  # type: ignore[attr-defined]
            f"qscheme={module.qscheme!r}",  # type: ignore[attr-defined]
            f"is_dynamic={module.is_dynamic!r}",  # type: ignore[attr-defined]
        ]
        if hasattr(observer, "reduce_range"):
            args.append(f"reduce_range={observer.reduce_range!r}")
        if hasattr(observer, "ch_axis"):
            args.append(f"ch_axis={observer.ch_axis!r}")
        return f"{module_type}({', '.join(args)})"

    @staticmethod
    def _qat_qconfig_constructor(module: torch.nn.Module) -> str | None:
        # [FUNCTION DOC] HELPER: Compares the module's qconfig by repr against the
        # default QAT qconfig of each known backend (skipping backends that raise
        # AssertionError) and returns a keyword argument string selecting the
        # first match, or None if no default qconfig matches.
        qconfig = getattr(module, "qconfig", None)
        for backend in NNModuleToString.default_qat_qconfig_backends:
            try:
                default_qconfig = torch.ao.quantization.get_default_qat_qconfig(backend)
            except AssertionError:
                continue
            if repr(qconfig) == repr(default_qconfig):
                return (
                    "qconfig=torch.ao.quantization.get_default_qat_qconfig"
                    f"({backend!r})"
                )
        return None

    @staticmethod
    def _qat_conv_constructor(module: torch.nn.Module) -> str | None:
        # [FUNCTION DOC] HELPER: For known QAT conv (optionally fused with
        # batchnorm/relu) modules whose qconfig is a default one, returns a
        # constructor call string with channels, conv hyperparameters, bias,
        # padding mode, batchnorm eps/momentum/freeze_bn when a bn child exists,
        # and the qconfig. Returns None otherwise.
        module_type = NNModuleToString._module_type_name(module)
        if module_type not in NNModuleToString.qat_conv_modules:
            return None
        qconfig = NNModuleToString._qat_qconfig_constructor(module)
        if qconfig is None:
            return None

        args = [
            repr(module.in_channels),  # type: ignore[attr-defined]
            repr(module.out_channels),  # type: ignore[attr-defined]
            f"kernel_size={module.kernel_size!r}",  # type: ignore[attr-defined]
            f"stride={module.stride!r}",  # type: ignore[attr-defined]
            f"padding={module.padding!r}",  # type: ignore[attr-defined]
            f"dilation={module.dilation!r}",  # type: ignore[attr-defined]
            f"groups={module.groups!r}",  # type: ignore[attr-defined]
            f"bias={module.bias is not None}",  # type: ignore[attr-defined]
            f"padding_mode={module.padding_mode!r}",  # type: ignore[attr-defined]
        ]
        bn = getattr(module, "bn", None)
        if bn is not None:
            args.extend(
                [
                    f"eps={bn.eps!r}",
                    f"momentum={bn.momentum!r}",
                    f"freeze_bn={module.freeze_bn!r}",  # type: ignore[attr-defined]
                ]
            )
        args.append(qconfig)
        return f"{module_type}({', '.join(args)})"

    @staticmethod
    def _qat_linear_constructor(module: torch.nn.Module) -> str | None:
        # [FUNCTION DOC] HELPER: For known QAT linear modules with a default
        # qconfig, returns a constructor call string (features, bias, qconfig).
        # Returns None otherwise.
        module_type = NNModuleToString._module_type_name(module)
        if module_type not in NNModuleToString.qat_linear_modules:
            return None
        qconfig = NNModuleToString._qat_qconfig_constructor(module)
        if qconfig is None:
            return None

        args = [
            repr(module.in_features),  # type: ignore[attr-defined]
            repr(module.out_features),  # type: ignore[attr-defined]
            f"bias={module.bias is not None}",  # type: ignore[attr-defined]
            qconfig,
        ]
        return f"{module_type}({', '.join(args)})"

    @staticmethod
    def _can_emit_module_constructor(module: torch.nn.Module, module_str: str) -> bool:
        # [FUNCTION DOC] HELPER: True only if the module's exact type is in
        # safe_reprs and module_str compiles as a valid expression (no syntax
        # error); false otherwise.
        if type(module) not in NNModuleToString.safe_reprs:
            return False

        try:
            compile(f"mod = {module_str}", "<module repr>", "exec")
        except SyntaxError:
            return False
        return True

    @staticmethod
    def _module_constructor_string(
        module: torch.nn.Module, *, allow_unsafe_repr: bool = False
    ) -> str | None:
        # [FUNCTION DOC] HELPER: Returns source text that constructs an
        # equivalent module: the repr if the type is safe, else a fake-quant or
        # QAT special constructor, else (only when allow_unsafe_repr is true) the
        # raw repr. Returns None if none applies.
        module_str = repr(module)
        if NNModuleToString._can_emit_module_constructor(module, module_str):
            return module_str
        constructor_str = (
            NNModuleToString._fake_quant_constructor(module)
            or NNModuleToString._qat_conv_constructor(module)
            or NNModuleToString._qat_linear_constructor(module)
        )
        if constructor_str is not None:
            return constructor_str
        if allow_unsafe_repr:
            return module_str
        return None

    @staticmethod
    def convert(gm: torch.fx.GraphModule, *, allow_unsafe_repr: bool = False) -> str:
        # [FUNCTION DOC] Returns python source (starting with a star import of
        # torch.nn) defining class Repro for gm, preceded by classes for any
        # nested GraphModule subgraphs (deduplicated by identity). Children are
        # constructed from constructor strings (moved to their device if not
        # CPU); buffers become literal tensors when tiny, otherwise random
        # tensors of the same shape/dtype; parameters become random Parameters;
        # the forward code is gm.code. Raises UnsupportedNNModuleError for a
        # child with no constructor string, AssertionError if the tensor print
        # threshold is below the inline limit.
        from torch.nn.modules.module import _addindent

        tab = " " * 4

        # Higher-order ops (torch.cond, torch.while_loop, ...) attach their
        # subgraphs as nested GraphModule children. These have no
        # constructor-style repr, so serialize each nested GraphModule as its
        # own top-level class (defined before Repro so it is in scope) and
        # instantiate it in __init__. Leaf nn.Modules use the constructor-string
        # path below.
        subgraph_classes: dict[int, str] = {}
        subgraph_defs: list[str] = []

        def register_subgraph(name_hint: str, sub: torch.fx.GraphModule) -> str:
            # [FUNCTION DOC] HELPER: Returns the generated class name for a
            # nested GraphModule, emitting its class definition into
            # subgraph_defs the first time it is seen (keyed by object identity).
            # Reserve the name before recursing so nested/shared subgraphs are
            # deduplicated by identity and cannot recurse infinitely.
            if id(sub) in subgraph_classes:
                return subgraph_classes[id(sub)]
            class_name = f"{name_hint}_{len(subgraph_classes)}"
            subgraph_classes[id(sub)] = class_name
            body = emit_module_body(sub)
            subgraph_defs.append(f"class {class_name}(torch.nn.Module):\n{body}")
            return class_name

        def emit_module_body(module: torch.nn.Module) -> str:
            # [FUNCTION DOC] HELPER: Returns the indented source of an __init__
            # (children, buffers, parameters) followed by the module's generated
            # forward code. Raises UnsupportedNNModuleError for unconvertible
            # children.
            model_str = (
                f"{tab}def __init__(self) -> None:\n{tab * 2}super().__init__()\n"
            )

            for module_name, child in module.named_children():
                if isinstance(child, torch.fx.GraphModule):
                    class_name = register_subgraph(module_name, child)
                    model_str += f"{tab * 2}self.{module_name} = {class_name}()\n"
                    continue
                module_str = NNModuleToString._module_constructor_string(
                    child, allow_unsafe_repr=allow_unsafe_repr
                )
                if module_str is None:
                    raise UnsupportedNNModuleError(
                        f"Cannot convert module to string: {child!r}"
                    )
                # module should be a core torch.nn.Module, so all parameters
                # and buffers should be on the same device.
                example_param = next(child.parameters(), None)
                example_tensor = (
                    example_param
                    if example_param is not None
                    else next(child.buffers(), None)
                )
                if example_tensor is not None and example_tensor.device.type != "cpu":
                    module_str = f'{module_str}.to("{example_tensor.device}")'
                model_str += f"{tab * 2}self.{module_name} = {module_str}\n"

            for buffer_name, buffer in module._buffers.items():
                if buffer is None:
                    continue
                # Serialize full data for small buffers
                if buffer.numel() <= MAX_CONSTANT_NUMEL_INLINE:
                    from torch._tensor_str import PRINT_OPTS

                    if PRINT_OPTS.threshold < MAX_CONSTANT_NUMEL_INLINE:
                        raise AssertionError(
                            f"PRINT_OPTS.threshold ({PRINT_OPTS.threshold}) must be >= "
                            f"MAX_CONSTANT_NUMEL_INLINE ({MAX_CONSTANT_NUMEL_INLINE})"
                        )
                    tensor_str = repr(buffer)
                elif torch.is_floating_point(buffer):
                    tensor_str = (
                        f"torch.randn({list(buffer.shape)}, dtype={buffer.dtype})"
                    )
                else:
                    tensor_str = f"torch.randint(1, size={list(buffer.shape)}, dtype={buffer.dtype})"
                if buffer.device.type != "cpu":
                    tensor_str = f'{tensor_str}.to("{buffer.device}")'
                model_str += (
                    f"{tab * 2}self.register_buffer('{buffer_name}', {tensor_str})\n"
                )

            for param_name, param in module._parameters.items():
                if param is None:
                    continue
                maybe_device = ""
                if param.device.type != "cpu":
                    maybe_device = f', device="{param.device}"'
                tensor_str = f"torch.nn.Parameter(torch.randn({list(param.shape)}, dtype={param.dtype}{maybe_device}))"
                model_str += f"{tab * 2}self.{param_name} = {tensor_str}\n"

            model_str += f"{_addindent(module.code, len(tab))}\n"
            return model_str

        repro_class = f"class Repro(torch.nn.Module):\n{emit_module_body(gm)}"
        # Subgraph classes must be defined before Repro instantiates them.
        return "\nfrom torch.nn import *\n" + "\n".join(subgraph_defs + [repro_class])


@functools.cache  # subprocess is expensive
def _cuda_system_info_comment() -> str:
    # [FUNCTION DOC] HELPER: Cached. Returns comment-line text describing CUDA
    # toolkit (nvcc --version output, skipped on ROCm or when nvcc is missing)
    # and GPU names with counts; or a single comment if CUDA is unavailable.
    # Side effect: runs the nvcc subprocess once.
    if not torch.cuda.is_available():
        return "# torch.cuda.is_available()==False, no GPU info collected\n"

    model_str = "# CUDA Info: \n"
    try:
        if torch.version.hip is None:
            cuda_version_out = subprocess.check_output(["nvcc", "--version"])
            cuda_version_lines = cuda_version_out.decode().split("\n")
            comment = "".join([f"# {s} \n" for s in cuda_version_lines if s != ""])
            model_str += f"{comment}\n"
        else:
            model_str += "# Not searching for nvcc on ROCM setup\n"
    except (OSError, subprocess.CalledProcessError):
        model_str += "# nvcc not found\n"

    gpu_names = Counter(
        torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())
    )

    model_str += "# GPU Hardware Info: \n"
    for name, count in gpu_names.items():
        model_str += f"# {name} : {count} \n"
    model_str += "\n"
    return model_str


def generate_env_vars_string(*, stable_output: bool = False) -> str:
    # [FUNCTION DOC] Returns python source that sets the current process's
    # TORCH/DYNAMO/INDUCTOR/TRITON environment variables (excluding some Triton
    # path vars and the repro-trigger vars, which are popped instead so repros do
    # not recurse). Single quotes in values are replaced with double quotes.
    # With stable_output, returns a placeholder comment instead.
    """
    Generate a string configuration for environment variables related to Dynamo, Inductor, and Triton.
    """
    if stable_output:
        return "# env var omitted due to stable_output=True"

    allow_list = ["TORCH", "DYNAMO", "INDUCTOR", "TRITON"]
    skip_list = ["TRITON_LIBDEVICE_PATH", "TRITON_PTXAS_PATH", "TRITON_LIBCUDA_PATH"]
    repro_env_vars = ["TORCHDYNAMO_REPRO_AFTER", "TORCHDYNAMO_REPRO_LEVEL"]

    def filter(key: str) -> bool:
        # [FUNCTION DOC] HELPER: True if the variable name contains an allowed
        # keyword and is not in the skip or repro lists.
        return (
            any(string in key for string in allow_list)
            and key not in skip_list
            and key not in repro_env_vars
        )

    config_lines = [
        f"""os.environ['{key}'] = '{value.replace("'", '"')}'"""
        for key, value in os.environ.items()
        if filter(key)
    ]
    # Repro scripts should not recursively generate more repro scripts when run.
    config_lines.extend(f"os.environ.pop('{key}', None)" for key in repro_env_vars)
    config_string = "\n".join(config_lines)
    return normalize_path_separator(f"""\
import os
{config_string}
    """)


def generate_config_string(*, stable_output: bool = False) -> str:
    # [FUNCTION DOC] Returns python source that imports the dynamo, inductor,
    # functorch and fx experimental config modules and assigns every non-default
    # value (via each module's codegen_config), plus non-default settings of
    # registered third-party inductor backend config modules. With stable_output,
    # returns a placeholder comment instead.
    import torch._functorch.config
    import torch._inductor.config
    from torch._inductor.codegen import common

    if stable_output:
        return "# config omitted due to stable_output=True"

    # Third-party Inductor backends can register their own ConfigModule.
    # Repros need to replay non-default values from those modules, and
    # ConfigModule.codegen_config() emits assignments but not the module import.
    extra_codegen_configs = []
    for c in common.custom_backend_codegen_configs.values():
        if isinstance(c, ConfigModule):
            codegen_config = c.codegen_config()
            if codegen_config:
                extra_codegen_configs.append(f"import {c.__name__}\n{codegen_config}")
    extra_codegen_configs_str = "\n".join(extra_codegen_configs)

    experimental_config = torch.fx.experimental._config.codegen_config()  # type: ignore[attr-defined]

    return f"""\
import torch._dynamo.config
import torch._inductor.config
import torch._functorch.config
import torch.fx.experimental._config
{torch._dynamo.config.codegen_config()}
{torch._inductor.config.codegen_config()}
{torch._functorch.config.codegen_config()}
{experimental_config}
{extra_codegen_configs_str}
"""


def get_minifier_repro_path() -> str:
    # [FUNCTION DOC] Returns the path of minifier_launcher.py inside the minifier
    # directory (creating the directory as a side effect).
    return os.path.join(minifier_dir(), "minifier_launcher.py")


def helper_for_dump_minify(contents: str) -> None:
    # [FUNCTION DOC] HELPER: Writes contents to the minifier launcher path
    # (logging a warning with the path; also writes a Buck TARGETS file under
    # fbcode). Raises NotImplementedError chained from the OSError if the file
    # cannot be written.
    minified_repro_path = get_minifier_repro_path()
    log.warning("Writing minified repro to:\n%s", minified_repro_path)

    if use_buck:
        BuckTargetWriter(minified_repro_path).write()
    try:
        with open(minified_repro_path, "w") as fd:
            fd.write(contents)

    except OSError as e:
        log.exception("")
        raise NotImplementedError(f"Could not write to {minified_repro_path}") from e


class AccuracyError(Exception):
    # [CLASS DOC] Exception signalling that compiled output does not match the
    # reference within tolerance. No state.
    pass


def clone_inputs_retaining_gradness(example_inputs: Sequence[Any]) -> list[Any]:
    # [FUNCTION DOC] Clones the inputs and forces each cloned tensor's
    # requires_grad to match the original (without a leaf check). Returns the
    # list of clones; inputs are not modified.
    """
    This clone inputs is different from utils clone_input. In case of minifier,
    all the tensors are leaf tensors while creating a new graph. So, we set the
    requires_grad field w/o checking the leafness of the tensor.
    """
    cloned_inputs = clone_inputs(example_inputs)
    for idx in range(len(example_inputs)):
        if isinstance(cloned_inputs[idx], torch.Tensor):
            cloned_inputs[idx].requires_grad_(example_inputs[idx].requires_grad)
    return cloned_inputs  # type: ignore[return-value]


def run_fwd_maybe_bwd(
    gm: torch.fx.GraphModule,
    args: Sequence[Any],
    only_fwd: bool = False,
    disable_clone: bool = False,
) -> Any:
    # [FUNCTION DOC] Runs gm on args using a deep copy of gm and (unless
    # disable_clone) cloned args. Supports the boxed calling convention. Returns
    # the raw output if only_fwd; otherwise runs backward on a scalar loss when
    # the output needs it and returns the collected results (outputs, grads).
    # With disable_clone the caller's args may be destroyed.
    """
    Runs a forward and possibly backward iteration for a given mod and args.

    When disable_clone is True, we will use args as-is without cloning.
    This is higher fidelity but we may destroy the args in the process.
    """
    from .testing import collect_results, reduce_to_scalar_loss, requires_bwd_pass

    gm = copy.deepcopy(gm)
    if not disable_clone:
        args = clone_inputs_retaining_gradness(args)

    if hasattr(gm, "zero_grad"):
        gm.zero_grad(True)

    # TorchInductor returned callable expects lists. So, may need a boxed calling convention.
    out = gm(args) if getattr(gm, "_boxed_call", False) else gm(*args)

    if only_fwd:
        return out
    if requires_bwd_pass(out):
        loss = reduce_to_scalar_loss(out)
        loss.backward()
    return collect_results(gm, out, None, args)


def same_two_models(
    gm: torch.fx.GraphModule,
    opt_gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    only_fwd: bool = False,
    *,
    require_fp64: bool = False,
    ignore_non_fp: bool = False,
) -> bool:
    # [FUNCTION DOC] Returns whether opt_gm matches gm on example_inputs within
    # config.repro_tolerance, optionally using an fp64 reference. If the fp64 run
    # fails it raises RuntimeError when require_fp64 else only warns. If opt_gm
    # itself raises, logs and returns True (treated as an unrelated problem).
    """
    Check two models have same accuracy.

    require_fp64: if True, raise an error if we unable to calculate the fp64 reference
    ignore_non_fp: if True, do not compare outputs which are not floating point.  This
        is mostly useful for the minifier (which wants to avoid quantizing floating point
        error into integer/boolean error)
    """
    from .utils import same

    ref = run_fwd_maybe_bwd(gm, example_inputs, only_fwd)

    fp64_ref = None
    if config.same_two_models_use_fp64:
        try:
            fp64_model, fp64_examples = cast_to_fp64(
                copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs)
            )
            fp64_ref = run_fwd_maybe_bwd(fp64_model, fp64_examples, only_fwd)
        except Exception:
            if require_fp64:
                raise RuntimeError(  # noqa: B904
                    "Could not generate fp64 outputs, workaround with torch._dynamo.config.same_two_models_use_fp64 = False"
                )
            log.warning("Could not generate fp64 outputs")

    try:
        res = run_fwd_maybe_bwd(opt_gm, example_inputs, only_fwd)
    except Exception:
        # This means that the minified graph is bad/exposes a different problem.
        # As we are checking accuracy here, lets log the exception and return True.
        log.exception(
            "While minifying the program in accuracy minification mode, "
            "ran into a runtime exception which is likely an unrelated issue."
            " Skipping this graph."
        )
        return True

    passing = same(
        ref,
        res,
        fp64_ref,
        tol=config.repro_tolerance,
        equal_nan=True,
        ignore_non_fp=ignore_non_fp,
    )
    return passing


def cast_dtype_args_to_fp64(model: torch.fx.GraphModule) -> torch.fx.GraphModule:
    # [FUNCTION DOC] Rewrites, in place, floating dtype arguments in the graph
    # (convert_element_type targets and dtype keyword arguments of call_function
    # nodes) to float64, then lints and recompiles. Returns the same model.
    # Raises AssertionError if a convert_element_type node lacks exactly two args.
    for node in model.graph.nodes:
        if (
            node.op == "call_function"
            and node.target is torch.ops.prims.convert_element_type.default
        ):
            if len(node.args) != 2:
                raise AssertionError(
                    f"Expected node to have 2 args, got {len(node.args)}"
                )
            if is_float_dtype(node.args[1]) and node.args[1] != torch.float64:
                node.args = (node.args[0], torch.float64)
        if node.op == "call_function":
            dtype = node.kwargs.get("dtype")
            if dtype is not None and is_float_dtype(dtype):
                new_kwargs = dict(node.kwargs)
                new_kwargs["dtype"] = torch.float64
                node.kwargs = new_kwargs

    model.graph.lint()
    model.recompile()
    return model


def cast_to(
    dtype: torch.dtype, model: torch.fx.GraphModule, inputs: list[Any]
) -> tuple[torch.fx.GraphModule, list[Any]]:
    # [FUNCTION DOC] Converts the model to dtype (also rewriting embedded dtype
    # arguments when dtype is float64) and casts floating tensors in the inputs
    # pytree to dtype. Returns the model and new inputs.
    from torch.utils._pytree import tree_map

    model = model.to(dtype)
    if dtype == torch.float64:
        # If casting to fp64 for accuracy comparison, we need to
        # replace dtype arguments embedded in the graph with fp64
        model = cast_dtype_args_to_fp64(model)

    inputs = tree_map(
        lambda x: x.to(dtype)
        if isinstance(x, torch.Tensor) and x.is_floating_point()
        else x,
        inputs,
    )
    return model, inputs


def cast_to_fp64(
    model: torch.fx.GraphModule, inputs: list[Any]
) -> tuple[torch.fx.GraphModule, list[Any]]:
    # [FUNCTION DOC] Shortcut for casting model and inputs to float64.
    return cast_to(torch.float64, model, inputs)


def backend_accuracy_fails(
    gm: torch.fx.GraphModule,
    example_inputs: Sequence[Any],
    compiler_fn: Callable[[torch.fx.GraphModule, list[Any]], torch.fx.GraphModule],
    only_fwd: bool = False,
    *,
    require_fp64: bool = False,
    ignore_non_fp: bool = False,
) -> bool:
    # [FUNCTION DOC] Compiles a copy of gm with compiler_fn and returns true if
    # the result disagrees with gm in accuracy. Any exception during compile or
    # comparison is logged and treated as no accuracy failure (returns false).
    try:
        compiled_gm = compiler_fn(
            copy.deepcopy(gm), clone_inputs_retaining_gradness(example_inputs)
        )
        return not same_two_models(
            gm,
            compiled_gm,
            example_inputs,
            only_fwd,
            require_fp64=require_fp64,
            ignore_non_fp=ignore_non_fp,
        )
    except Exception:
        # This means that the minified graph is bad/exposes a different problem.
        # As we are checking accuracy here, lets log the exception and return False.
        log.exception(
            "While minifying the program in accuracy minification mode, "
            "ran into a runtime exception which is likely an unrelated issue."
            " Skipping this graph"
        )
        return False


# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #
#                       REPRO SUPPORT CODE
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ #


# Helper functions for computing what the default values of tensor
# values should be.  These all coincide with factory functions, e.g., torch.empty


def _stride_or_default(
    stride: torch._prims_common.StrideType | None,
    *,
    shape: torch._prims_common.ShapeType,
) -> torch._prims_common.StrideType:
    # [FUNCTION DOC] HELPER: Returns stride if given, else contiguous strides
    # for shape.
    return stride if stride is not None else utils.make_contiguous_strides_for(shape)


def _mk_defaulter(d: T) -> Callable[[T | None], T]:
    # [FUNCTION DOC] HELPER: Returns a function mapping None to the default d and
    # any other value to itself.
    return lambda x: x if x is not None else d


_dtype_or_default = _mk_defaulter(torch.float32)
_device_or_default = _mk_defaulter(torch.device("cpu"))
_storage_offset_or_default = _mk_defaulter(0)
_requires_grad_or_default = _mk_defaulter(False)
_is_leaf_or_default = _mk_defaulter(False)


class NopInputReader:
    # [CLASS DOC] HELPER: Reader that ignores all inputs, only counting storages
    # and opaque objects in total. Used to count how many items a load_args
    # function would read (e.g. for a progress bar).
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the total counter to zero.
        self.total = 0

    def storage(
        self,
        storage_hash: str | None,
        nbytes: int,
        *,
        device: torch._prims_common.DeviceLikeType | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> None:
        # [FUNCTION DOC] HELPER: Counts one storage; returns nothing.
        self.total += 1

    def tensor(self, *args: Any, **kwargs: Any) -> torch.Tensor | None:
        # [FUNCTION DOC] HELPER: Ignores the call and returns None.
        pass

    def symint(self, *args: Any, **kwargs: Any) -> int | None:
        # [FUNCTION DOC] HELPER: Ignores the call and returns None.
        pass

    def const(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Ignores the call.
        pass

    def unsupported(self, name: str) -> None:
        # [FUNCTION DOC] HELPER: Ignores the call.
        pass

    def generator(self, device_type: str, device_index: int) -> None:
        # [FUNCTION DOC] HELPER: Ignores the call.
        pass

    def opaque(self, script_class_name: str) -> None:
        # [FUNCTION DOC] HELPER: Counts one opaque object; returns nothing.
        self.total += 1


# TODO: Support bundling the entire repro into a zip file for ease of
# transferring around
class InputReader:
    # [CLASS DOC] Replays the load_args function emitted by InputWriter: each
    # reader method rebuilds one argument (storage, tensor, symint, constant,
    # generator, ...), appending it to self.args in order. State: store (content
    # store reader or None meaning use random data), args (rebuilt arguments),
    # pbar (optional progress bar), and a lazily created symint_exprs mapping from
    # argument position to symbolic expression text.
    def __init__(
        self, save_dir: str | None = None, *, pbar: tqdm | None = None
    ) -> None:
        # [FUNCTION DOC] Creates a reader backed by a content store in save_dir,
        # or, when save_dir is None, warns and falls back to random data.
        # pbar, if given, is advanced once per storage read.
        # If None, we will generate random data instead.  It's important
        # to natively support this use case as it will allow people to
        # share repros without including the real data, if the problem
        # reproduces even on random data.
        if save_dir is None:
            log.warning("no save_dir specified, will generate random data")
        self.store = ContentStoreReader(save_dir) if save_dir is not None else None
        self.args: list[Any] = []
        self.pbar = pbar

    def storage(
        self,
        storage_hash: str | None,
        nbytes: int,
        *,
        device: torch._prims_common.DeviceLikeType | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> UntypedStorage:
        # [FUNCTION DOC] Returns an untyped storage for storage_hash: loaded from
        # the store when available (warning if its device differs from the
        # requested one), otherwise a warn-once fallback to random strided data of
        # nbytes using dtype_hint (default float32) on device (default CPU).
        if self.pbar is not None:
            self.pbar.update(1)
        device = _device_or_default(device)  # type: ignore[arg-type]
        dtype_hint = _dtype_or_default(dtype_hint)
        if self.store is not None and storage_hash is not None:
            try:
                storage = self.store.read_storage(storage_hash)
            except FileNotFoundError:
                pass
            else:
                if device != storage.device:
                    log.warning("device mismatch: %s != %s", device, storage.device)
                    # TODO: transfer it to the right device?  But failing this
                    # way would be very mysterious!  Would have been better
                    # not to store device in the serialized format...
                return storage
        warn_once(f"could not load {storage_hash}, generating random data instead")
        shape = (nbytes // dtype_hint.itemsize,)
        stride = _stride_or_default(None, shape=shape)
        return rand_strided(shape, stride, dtype_hint, device).untyped_storage()

    def tensor(
        self,
        storage: UntypedStorage,
        shape: torch._prims_common.ShapeType,
        stride: torch._prims_common.StrideType | None = None,
        *,
        storage_offset: int | None = None,
        dtype: torch.dtype | None = None,
        requires_grad: bool | None = None,
        is_leaf: bool | None = None,
        **metadata: Any,
    ) -> torch.Tensor:
        # [FUNCTION DOC] Builds a tensor viewing storage with the given shape,
        # stride, offset and dtype (defaults fill None values), restoring
        # requires_grad, leaf-ness (non-leaves get fake autograd history via a
        # clone) and extra metadata. Appends it to args and returns it. Raises
        # AssertionError if the resulting leaf status differs from is_leaf.
        stride = _stride_or_default(stride, shape=shape)
        storage_offset = _storage_offset_or_default(storage_offset)
        dtype = _dtype_or_default(dtype)
        is_leaf = _is_leaf_or_default(is_leaf)
        requires_grad = _requires_grad_or_default(requires_grad)
        t = torch.tensor(
            [], dtype=dtype, device=storage.device, requires_grad=requires_grad
        )
        with torch.no_grad():
            t.set_(storage, storage_offset, shape, stride)
        if not is_leaf:
            # Fake up some autograd history in a very naughty way
            with torch.enable_grad():
                t = t.clone(memory_format=torch.preserve_format)
            with torch.no_grad():
                t.set_(storage, storage_offset, shape, stride)
        if torch._subclasses.meta_utils.safe_is_leaf(t) != is_leaf:
            raise AssertionError(
                f"Tensor leaf status mismatch: safe_is_leaf(t) = "
                f"{torch._subclasses.meta_utils.safe_is_leaf(t)}, expected {is_leaf}"
            )
        torch._utils.set_tensor_metadata(t, metadata)
        self.args.append(t)
        return t  # for BC

    def symint(self, val: Any, *, expr: str | None = None) -> Any:
        # [FUNCTION DOC] Appends val as the next argument and returns it; if expr
        # is given, records it in symint_exprs (created on first use) under the
        # argument's position.
        self.args.append(val)
        if expr is not None:
            if not hasattr(self, "symint_exprs"):
                self.symint_exprs: dict[int, str] = {}
            self.symint_exprs[len(self.args) - 1] = expr
        return val  # for BC

    def const(self, name: str) -> None:
        # [FUNCTION DOC] Appends None as a placeholder for a constant argument
        # that was filtered out during compilation.
        self.args.append(None)

    def unsupported(self, name: str) -> None:
        # [FUNCTION DOC] Appends None as a placeholder for an argument of an
        # unsupported type.
        self.args.append(None)

    def generator(self, device_type: str, device_index: int) -> Any:
        # [FUNCTION DOC] Creates a new torch Generator on the given device,
        # appends it to args and returns it (RNG state is not restored).
        device = torch.device(device_type, device_index)
        gen = torch.Generator(device=device)
        self.args.append(gen)
        return gen

    def opaque(self, script_class_name: str) -> None:
        # [FUNCTION DOC] Appends None as a placeholder for an opaque script-class
        # object (not reconstructed).
        self.args.append(None)


# Here is our writer strategy:
#  1. We will stream all of the inputs to disk
#  2. You can now deterministically randomize the inputs, or reload
#     the inputs from disk
#  3. You can YOLO run the script without the inputs, in which case
#     we'll fill the inputs with random data and pray.  This is the
#     legacy behavior, but it's also useful if you want to find out
#     if we're so broken even random inputs trigger it
#  4. We could offer an in process "check if the randomized thing
#     works too" but this is delicate so we don't do it


def _serialize_storage_nbytes(nbytes: int | torch.SymInt) -> str:
    # [FUNCTION DOC] HELPER: Returns source text for a storage byte size: the
    # printed symbolic expression for a SymInt, otherwise the repr of the int.
    if isinstance(nbytes, torch.SymInt):
        from torch.fx.experimental.symbolic_shapes import SymExprPrinter

        return SymExprPrinter().doprint(nbytes.node.expr)
    return repr(nbytes)


class InputWriter:
    # [CLASS DOC] Records graph inputs as source lines of a load_args(reader)
    # function that an InputReader can replay. State: _lines (emitted statements),
    # storage_counter (names buf0, buf1, ...), save_dir and store (optional
    # content store that receives real storage data), seen_storages (weak-ref to
    # buffer name map so shared storages are emitted once). Methods mirror the
    # reader: storage, tensor, symint, const, unsupported, generator, opaque.
    def __init__(self, save_dir: str | None, *, stable_hash: bool = False) -> None:
        # [FUNCTION DOC] Initializes empty state; creates a content store writer
        # in save_dir (with stable_hash option) unless save_dir is None, in which
        # case storage contents are not saved.
        self._lines: list[str] = []
        # TODO: consider ensuring tensor and storage counters line up?
        self.storage_counter = itertools.count()
        self.save_dir = save_dir
        self.store = (
            ContentStoreWriter(save_dir, stable_hash=stable_hash)
            if save_dir is not None
            else None
        )
        self.seen_storages: dict[StorageWeakRef, str] = {}

    def lines(self) -> list[str]:
        # [FUNCTION DOC] Returns the source lines of the load_args function: a
        # def line, the indented recorded statements, and a version attribute
        # assignment.
        r = [
            "def load_args(reader):",
        ]
        r.extend(f"    {l}" for l in self._lines)
        # In case we need to change the internal format of load_args
        # in an FC-breaking way
        r.append("load_args._version = 0")
        return r

    # Storages are untyped, but we need to initialize them with data if
    # we don't have the real data, so we give a hint saying what kind
    # of initialization may be appropriate
    #
    # If we had a FakeTensor, device_hint tells us what device should be
    def storage(
        self,
        untyped_storage: UntypedStorage,
        *,
        device_hint: torch._prims_common.DeviceLikeType | None = None,
        dtype_hint: torch.dtype | None = None,
    ) -> str:
        # [FUNCTION DOC] Emits a reader.storage statement for untyped_storage
        # (once per distinct storage) and returns the variable name bound to it.
        # Writes real data to the store unless the storage is on meta; meta
        # storages use device_hint as the device (AssertionError if absent).
        # Dtype and device arguments are emitted only when not the defaults.
        ws = StorageWeakRef(untyped_storage)
        v = self.seen_storages.get(ws)
        if v is not None:
            return v
        v = f"buf{next(self.storage_counter)}"
        maybe_dtype_hint = ""
        if _dtype_or_default(None) != _dtype_or_default(dtype_hint):
            maybe_dtype_hint = f", dtype_hint={dtype_hint!r}"
        # TODO: being optional on device is kind of pointless as the default
        # is CPU but most repros we care about are CUDA
        maybe_device = ""
        device = untyped_storage.device
        if device.type == "meta":
            if device_hint is None:
                raise AssertionError(
                    "device_hint must be provided when storage device is 'meta'"
                )
            device = device_hint  # type: ignore[assignment]
        if _device_or_default(None) != device:
            maybe_device = f", device={device!r}"
        nbytes = untyped_storage.nbytes()
        nbytes_source = _serialize_storage_nbytes(nbytes)
        storage_hash = None
        if self.store is not None and untyped_storage.device.type != "meta":
            storage_hash = self.store.write_storage(untyped_storage)
        self._lines.append(
            f"{v} = reader.storage({storage_hash!r}, {nbytes_source}{maybe_device}{maybe_dtype_hint})"
        )
        self.seen_storages[ws] = v
        return v

    def tensor(self, name: str, t: torch.Tensor) -> None:
        # [FUNCTION DOC] Records a storage plus a reader.tensor statement for t,
        # listing only non-default stride, dtype, storage offset, extra metadata,
        # requires_grad and leaf-ness; name appears as a trailing comment.
        from torch.fx.experimental.symbolic_shapes import statically_known_true, sym_eq

        storage = self.storage(
            t.untyped_storage(), dtype_hint=t.dtype, device_hint=t.device
        )
        args = []
        # NB: this is positional, must come first
        if not statically_known_true(
            sym_eq(_stride_or_default(None, shape=t.shape), t.stride())
        ):
            args.append(str(tuple(t.stride())))
        if _dtype_or_default(None) != t.dtype:
            args.append(f"dtype={t.dtype!r}")
        if not statically_known_true(
            _storage_offset_or_default(None) == t.storage_offset()
        ):
            args.append(f"storage_offset={t.storage_offset()!r}")
        tensor_metadata = torch._utils.get_tensor_metadata(t)
        if tensor_metadata:
            args.extend(f"{k}={v!r}" for k, v in tensor_metadata.items())
        if _requires_grad_or_default(None) != t.requires_grad:
            args.append(f"requires_grad={t.requires_grad!r}")
        is_leaf = torch._subclasses.meta_utils.safe_is_leaf(t)
        if _is_leaf_or_default(None) != is_leaf:
            args.append(f"is_leaf={is_leaf!r}")
        self._lines.append(
            "reader.tensor("
            + ", ".join([storage, str(tuple(t.shape)), *args])
            + f")  # {name}"
        )

    def unsupported(self, name: str, arg: Any) -> None:
        # [FUNCTION DOC] Records a reader.unsupported statement noting the arg's
        # type. For list or tuple args it also emits, inside a string literal
        # block, best-effort statements for each element (tensor, symint, or
        # recursively unsupported) so a user can repair the repro by hand.
        # NB: Try hard not to /print/ a tensor, that will be very slow
        self._lines.append(
            f"reader.unsupported({name!r})  # unsupported type for dumping: {type(arg)}"
        )
        # Best effort dump as much useful stuff we can lol, in case you want
        # to repair the repro
        if isinstance(arg, (list, tuple)):
            self._lines.append('"""')
            for i, a in enumerate(arg):
                name_i = f"{name}[{i}]"
                if isinstance(a, torch.Tensor):
                    self.tensor(name_i, a)
                elif isinstance(a, (int, torch.SymInt)):
                    self.symint(name_i, a)
                else:
                    self.unsupported(name_i, a)
            self._lines.append('"""')

    # write out that the arg was filtered out as it is constant
    def const(self, name: str) -> None:
        # [FUNCTION DOC] Records a reader.const statement for an argument that was
        # filtered out during compilation as a constant.
        self._lines.append(
            f"reader.const({name!r})  # {name}, filtered out during compilation"
        )

    # TODO: this doesn't actually symint atm
    def symint(self, name: str, val: Any) -> None:
        # [FUNCTION DOC] Records a reader.symint statement: for a SymInt, its
        # hint and symbolic expression text; for a plain int, just the value.
        if isinstance(val, torch.SymInt):
            expr_str = str(val.node.expr)
            hint = val.node.hint
            self._lines.append(f"reader.symint({hint!r}, expr={expr_str!r})  # {name}")
        else:
            self._lines.append(f"reader.symint({val!r})  # {name}")

    def generator(self, name: str, arg: torch._C.Generator) -> None:
        # [FUNCTION DOC] Records a reader.generator statement with the
        # generator's device type and index (RNG state is not saved).
        device = arg.device
        self._lines.append(
            f"reader.generator({device.type!r}, {device.index!r})  # {name}"
        )

    def opaque(self, name: str, script_class_name: str) -> None:
        # [FUNCTION DOC] Records a reader.opaque statement naming the script
        # class of an opaque object argument.
        self._lines.append(f"reader.opaque({script_class_name!r})  # {name}")


def aot_graph_input_parser(
    func: Callable[[list[Tensor]], list[Tensor]],
    device: str = "cuda",
    sym_shapes: dict[str, int] | None = None,
    default_sym_shape: int | None = None,
) -> dict[str, Any]:
    # [FUNCTION DOC] Builds keyword arguments for calling func, a function whose
    # source came from a printed fx graph. Parses parameter annotations for dtype
    # and shape (random floats or zeros on device) and symbolic-size annotations
    # (looked up in sym_shapes, else default_sym_shape; torch._check fails if
    # neither exists). Symbolic dims are marked dynamic. If func takes self, adds
    # a container with tensor constants parsed from the source text. Requires
    # retrievable source (inspect errors propagate).
    """
    Takes in a function which has been printed with print_readable() and constructs kwargs to run it.

    Handles Tensor inputs, Symints, and a graph module which might have tensor constants.

    Consider a function `forward` defined as follows:

    def forward(self, primals_1: "f32[1001, 6]", primals_2: "f32[s0]", primals_3: "Sym(s0)",):
        _tensor_constant0: "i64[4190]" = self._tensor_constant0
        # Further implementation

    kwargs = aot_graph_input_parser(forward)
    forward(**kwargs)
    """

    from torch.utils._dtype_abbrs import dtype_abbrs

    dtype_map: dict[str, torch.dtype] = {
        value: key for key, value in dtype_abbrs.items()
    }
    dtype_pattern: str = "|".join(dtype_abbrs.values())

    # Extracting the source code from the function
    source = inspect.getsource(func)

    # Regular expressions
    tensor_assignment_regex = rf"(_tensor_constant\d+): \"({dtype_pattern})\[\s*(.*?)\s*\]\" = self\.(_tensor_constant\d+)"
    tensor_regex = rf"({dtype_pattern})\[\s*(.*?)\s*\]"
    sym_shape_regex = r"Sym\((s\d+)\)"

    class TensorContainer:
        # [CLASS DOC] HELPER: Empty attribute holder standing in for self, with
        # generated tensor constants attached as attributes.
        "Container for tensors as attributes"

    # Dictionary for tensors from annotations
    kwargs: dict[str, Any] = {}

    sym_shapes_dict: dict[str, int] = sym_shapes or {}

    def get_sym_int(symint: str) -> int:
        # [FUNCTION DOC] HELPER: Returns the concrete size for a symbol name from
        # the provided mapping or the default; raises via torch._check if neither
        # is available.
        torch._check(
            symint in sym_shapes_dict or default_sym_shape is not None,
            lambda: f"{symint} not in symbolic_shapes and default sym shape not passed in",
        )
        return sym_shapes_dict.get(symint, default_sym_shape)  # type: ignore[return-value]

    def gen_tensor(shape: torch._prims_common.ShapeType, dtype: torch.dtype) -> Tensor:
        # [FUNCTION DOC] HELPER: Creates a tensor on the target device from
        # textual dims: symbolic dims (those containing the letter s) are
        # resolved via get_sym_int and marked dynamic, empty dims are dropped.
        # Floating dtypes get random normal data, others zeros.
        # Resolve symbolic shapes to concrete values
        resolved_shape = []
        dynamic_dims = []
        for i, dim in enumerate(shape):
            dim = dim.strip()  # type: ignore[attr-defined]
            if "s" in dim:
                s = get_sym_int(dim)
                resolved_shape.append(s)
                dynamic_dims.append(i)
            else:
                if dim:
                    resolved_shape.append(int(dim))

        constructor = torch.randn if dtype.is_floating_point else torch.zeros
        out = constructor(resolved_shape, dtype=dtype, device=device)  # type: ignore[call-arg]
        for d in dynamic_dims:
            torch._dynamo.mark_dynamic(out, d)
        return out

    # Parse function annotations for tensor generation
    annotations = func.__annotations__
    for param, annotation in annotations.items():
        # Skip 'return' annotation
        if param == "return":
            continue

        match = re.search(tensor_regex, annotation)
        if match:
            data_type, shape_str = match.groups()
            shape = tuple(shape_str.split(","))
            dtype = dtype_map[data_type]
            # pyrefly: ignore [bad-argument-type]
            kwargs[param] = gen_tensor(shape, dtype)

        match = re.search(sym_shape_regex, annotation)
        if match:
            kwargs[param] = get_sym_int(match.group(1))

    if "self" in inspect.signature(func).parameters:
        container = TensorContainer()
        kwargs["self"] = container
        for match in re.finditer(tensor_assignment_regex, source):
            attr_name, data_type, shape_str, _ = match.groups()
            shape = tuple(shape_str.split(","))
            dtype = dtype_map[data_type]
            # pyrefly: ignore [bad-argument-type]
            setattr(container, attr_name, gen_tensor(shape, dtype))

    return kwargs


def profile_to_file(filename: str) -> Callable[[T], T]:
    # [FUNCTION DOC] Returns a decorator that runs the wrapped function under a
    # shared cProfile profiler and, at interpreter exit (atexit hook registered
    # here), dumps stats to filename (expanded to an absolute path) and prints a
    # snakeviz hint to stderr.
    # KNOWN ISSUE: the declared return type says Callable[[T], T] but the
    # decorator accepts and returns arbitrary callables.
    """
    Decorator to cProfile a given function and save the result to disk on process exit.

    Args:
        filename: filename to save profile to
    """
    prof = cProfile.Profile()
    filename = os.path.abspath(os.path.expanduser(filename))

    def decorator(fn: Any) -> Any:
        # [FUNCTION DOC] HELPER: Wraps fn so each call is profiled.
        @functools.wraps(fn)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            # [FUNCTION DOC] HELPER: Enables the profiler, calls fn and always
            # disables the profiler afterwards; returns fn's result.
            prof.enable()
            try:
                return fn(*args, **kwargs)
            finally:
                prof.disable()

        return wrapper

    def save_it() -> None:
        # [FUNCTION DOC] HELPER: Writes accumulated profile stats to the file and
        # prints viewing instructions to stderr (runs at exit).
        prof.dump_stats(filename)
        sys.stderr.write(
            textwrap.dedent(
                f"""\
                Wrote profile to {filename}, view with:

                    snakeviz {filename}

                """
            )
        )

    atexit.register(save_it)
    return decorator
