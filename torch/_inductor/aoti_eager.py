# [MODULE DOC]
# Support for AOTI eager mode: compiles a single op to a kernel library with
# AOTInductor, caches it on disk per namespace and device together with a JSON file of
# per-argument metadata (shapes, dtypes, scalar values, ...), and loads that cache back.
# Contains a registry of per-device-type compile/load backends, the metadata extraction
# helpers for each argument kind, a file-lock helper, and the compile-with-cache
# and cache loading entry points. Import-time side effects: creates the module logger
# and the empty backend registry.
import json
import logging
import os
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from unittest import mock

import torch
import torch._export
from torch._inductor.utils import is_cpu_device

from .runtime.runtime_utils import cache_dir


log = logging.getLogger(__name__)


@dataclass
class AOTICompileBackend:
    # [CLASS DOC] Pair of callables for a device type: compile_fn compiles an op and returns the
    # kernel library path, load_fn loads cached metadata items for an op (namespace, op name, device type).
    compile_fn: Callable[..., str]
    load_fn: Callable[[str, str, str], list[dict[str, Any] | None]]


_aoti_compile_backends: dict[str, AOTICompileBackend] = {}


def register_aoti_compile_backend(
    device_type: str,
    compile_fn: Callable[..., str],
    load_fn: Callable[[str, str, str], list[dict[str, Any] | None]],
) -> None:
    # [FUNCTION DOC] Registers (or replaces) the compile and load callables used for a device type
    # in the module-level backend registry (global state mutation).
    _aoti_compile_backends[device_type] = AOTICompileBackend(
        compile_fn=compile_fn,
        load_fn=load_fn,
    )


def aoti_eager_cache_dir(namespace: str, device: str) -> Path:
    # [FUNCTION DOC] Returns the cache directory path for a namespace and device: the inductor
    # cache dir joined with aoti_eager, namespace and device. Does not create it.
    return Path(cache_dir()) / "aoti_eager" / namespace / device


def aoti_eager_op_conf_lock(op_func_name_with_overload: str) -> Any:
    # [FUNCTION DOC] Returns a file lock (not yet acquired; use as a context manager) guarding the
    # op's JSON config, located in the inductor lock directory with the op name plus a
    # lock suffix and the standard lock timeout. Imports lazily to avoid a circular import.
    # Avoid circular import
    from torch._inductor.codecache import get_lock_dir, LOCK_TIMEOUT
    from torch.utils._filelock import FileLock

    op_conf_lock_file = f"{op_func_name_with_overload}.lock"
    lock_dir = get_lock_dir()
    return FileLock(os.path.join(lock_dir, op_conf_lock_file), timeout=LOCK_TIMEOUT)


def load_aoti_eager_cache(
    ns: str, op_func_name_with_overload: str, device_type: str
) -> list[dict[str, Any] | None]:
    # [FUNCTION DOC]
    # Loads cached kernel metadata for an op. A registered backend for the device type
    # takes over completely. Otherwise reads the op's JSON config under the file lock,
    # makes each kernel path absolute, converts stored dtype/layout/memory-format name
    # strings into torch objects and sets device index -1 for cpu. Returns an empty list
    # if the config or any kernel library is missing, or on any error (including the
    # NotImplementedError for dynamic-shape entries), which is logged, not raised.
    backend = _aoti_compile_backends.get(device_type)
    if backend:
        return backend.load_fn(ns, op_func_name_with_overload, device_type)

    device_kernel_cache = aoti_eager_cache_dir(ns, device_type)
    op_conf = device_kernel_cache / f"{op_func_name_with_overload}.json"
    if not op_conf.exists():
        return []

    try:
        with aoti_eager_op_conf_lock(op_func_name_with_overload):
            with open(op_conf) as f:
                json_data = json.load(f)
                for item in json_data:
                    # Get absolute path for kernel library
                    kernel_lib_abs_path = device_kernel_cache / item["kernel_path"]
                    item["kernel_path"] = kernel_lib_abs_path.as_posix()

                    # Check if the kernel library exists
                    if not kernel_lib_abs_path.exists():
                        return []

                    for metadata in item["meta_info"]:
                        if metadata.get("is_dynamic"):
                            raise NotImplementedError(
                                "Only support static shape for now"
                            )
                        if (
                            "device_type" in metadata
                            and metadata["device_type"] == "cpu"
                        ):
                            metadata["device_index"] = -1
                        for dtype_key in ["dtype", "dtype_value"]:
                            if dtype_key in metadata:
                                metadata[dtype_key] = getattr(
                                    torch, metadata[dtype_key].split(".")[-1]
                                )
                        if "layout_value" in metadata:
                            metadata["layout_value"] = getattr(
                                torch, metadata["layout_value"].split(".")[-1]
                            )
                        if "memory_format_value" in metadata:
                            metadata["memory_format_value"] = getattr(
                                torch, metadata["memory_format_value"].split(".")[-1]
                            )

                return json_data
    except Exception as e:
        err_msg = f"Failed to load aoti eager cache: {e}"
        log.exception(err_msg)
        return []


def supported_builtin_dtype_torch_dtype() -> dict[type, torch.dtype]:
    # [FUNCTION DOC] Returns the mapping from supported Python scalar types to torch dtypes (int to int32, float to float32, bool to bool).
    return {int: torch.int32, float: torch.float, bool: torch.bool}


def supported_scalar_types() -> tuple[type, ...]:
    # [FUNCTION DOC] Returns the tuple of Python scalar types accepted as op arguments (int, float, bool).
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    return tuple(type_to_torch_dtype.keys())


def extract_tensor_metadata(dynamic: bool, input: torch.Tensor) -> dict[str, Any]:
    # [FUNCTION DOC] Builds the cache metadata mapping for a tensor: dynamic flag, device type and
    # index (-1 for cpu), dtype string, sizes, strides, requires_grad and raw dispatch key
    # set. Raises AssertionError if input is not a tensor.
    metadata: dict[str, Any] = {}
    metadata["is_dynamic"] = dynamic

    if not isinstance(input, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(input)}")
    metadata["device_type"] = f"{input.device.type}"
    if is_cpu_device([input]):
        metadata["device_index"] = -1
    else:
        metadata["device_index"] = input.device.index
    metadata["dtype"] = f"{input.dtype}"
    metadata["sizes"] = list(input.size())
    metadata["strides"] = list(input.stride())
    metadata["requires_grad"] = input.requires_grad
    metadata["dispatch_key_set"] = torch._C._dispatch_keys(input).raw_repr()
    return metadata


def extract_tensor_list_metadata(
    dynamic: bool,
    input: list[torch.Tensor],
) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata for a tensor list: a mapping with key tensor_list holding the
    # per-tensor metadata. Raises AssertionError if any item is not a tensor.
    metadata_list = []
    for item in input:
        if not isinstance(item, torch.Tensor):
            raise AssertionError(f"expected torch.Tensor, got {type(item)}")
        metadata_list.append(extract_tensor_metadata(dynamic, item))

    metadata: dict[str, Any] = {}
    metadata["tensor_list"] = metadata_list
    return metadata


def extract_scalar_metadata(device_type: str, input: Any) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata for a Python int/float/bool: static, given device type (index
    # -1 for cpu else 0), mapped dtype string and the scalar value. Raises AssertionError
    # for unsupported types.
    if not isinstance(input, supported_scalar_types()):
        raise AssertionError(f"expected a supported scalar type, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["is_dynamic"] = False
    # Scalar tensor
    metadata["device_type"] = device_type
    metadata["device_index"] = -1 if device_type == "cpu" else 0
    type_to_torch_dtype = supported_builtin_dtype_torch_dtype()
    metadata["dtype"] = f"{type_to_torch_dtype[type(input)]}"
    metadata["scalar_value"] = input
    return metadata


def extract_string_metadata(input: str) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata with the string value; AssertionError if input is not a str.
    if not isinstance(input, str):
        raise AssertionError(f"expected str, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["string_value"] = input
    return metadata


def extract_dtype_metadata(input: torch.dtype) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata with the dtype stored as a string; AssertionError if not a dtype.
    if not isinstance(input, torch.dtype):
        raise AssertionError(f"expected torch.dtype, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["dtype_value"] = f"{input}"
    return metadata


def extract_device_metadata(input: torch.device) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata with the device type string and device index; AssertionError if not a device.
    if not isinstance(input, torch.device):
        raise AssertionError(f"expected torch.device, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["device_type_value"] = f"{input.type}"
    metadata["device_index_value"] = input.index
    return metadata


def extract_layout_metadata(input: torch.layout) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata with the layout stored as a string; AssertionError if not a layout.
    if not isinstance(input, torch.layout):
        raise AssertionError(f"expected torch.layout, got {type(input)}")
    metadata: dict[str, Any] = {}
    metadata["layout_value"] = f"{input}"
    return metadata


def extract_int_list_metadata(input: list[int]) -> dict[str, Any]:
    # [FUNCTION DOC] Builds metadata with the list of ints copied under int_list_value. Input must be
    # a list or tuple of non-bool ints (an empty one is allowed), else AssertionError.
    if not (
        isinstance(input, (list, tuple))
        and all(isinstance(item, int) and not isinstance(item, bool) for item in input)
    ):
        raise AssertionError(f"expected a list/tuple of int, got {input!r}")
    metadata: dict[str, Any] = {}
    metadata["int_list_value"] = list(input)
    return metadata


def aoti_compile_with_persistent_cache(
    ns: str,
    op_func_name_with_overload: str,
    device_type: str,
    dynamic: bool,
    f: Callable[..., Any],
    args: tuple[Any],
    kwargs: dict[str, Any],
    *,
    dynamic_shapes: dict[str, Any] | None = None,
    options: dict[str, Any] | None = None,
    remove_runtime_assertions: bool = False,
    disable_constraint_solver: bool = False,
) -> str:
    # [FUNCTION DOC]
    # Compiles an op function with AOTInductor and records it in the on-disk eager cache.
    # A registered backend for the device type handles everything. Otherwise: dynamic shapes
    # raise AssertionError; unsupported argument types (or lists of non-tensors) raise
    # NotImplementedError. Creates the cache and lib directories, compiles with the inductor
    # cache dir redirected to the lib directory via a temporary environment patch, builds
    # per-argument metadata (None args skipped; each tagged with its arg order), and under the
    # file lock appends a new entry (kernel path relative to the cache plus metadata) to the
    # op JSON file unless identical metadata exists. Returns the kernel library path.
    # Any failure while compiling or writing is logged and yields an empty string.
    # KNOWN ISSUE: the None skip and the compile-failure empty-string result mean callers must
    # check for an empty path; None args are rejected by the earlier type check anyway.
    """
    Compile the given function with persistent cache for AOTI eager mode.
    """
    backend = _aoti_compile_backends.get(device_type)
    if backend:
        return backend.compile_fn(
            ns,
            op_func_name_with_overload,
            device_type,
            dynamic,
            f,
            args,
            kwargs,
            dynamic_shapes=dynamic_shapes,
            options=options,
            remove_runtime_assertions=remove_runtime_assertions,
            disable_constraint_solver=disable_constraint_solver,
        )

    if dynamic:
        raise AssertionError("Only support static shape for now")
    flattened_inputs = list(args) + list(kwargs.values())
    if not all(
        isinstance(
            input,
            (
                supported_scalar_types(),
                torch.Tensor,
                list,
                str,
                torch.dtype,
                torch.device,
                torch.layout,
            ),
        )
        for input in flattened_inputs
    ):
        err_msg = f"Unsupported input types: {flattened_inputs}"
        log.exception(err_msg)
        raise NotImplementedError(err_msg)

    for input in flattened_inputs:
        if isinstance(input, list) and not all(
            isinstance(item, torch.Tensor) for item in input
        ):
            err_msg = f"_impl_with_aoti_compile encounters unsupported input types: {flattened_inputs}"
            log.exception(err_msg)
            raise NotImplementedError(err_msg)

    persistent_cache = aoti_eager_cache_dir(ns, device_type)
    if not persistent_cache.exists():
        persistent_cache.mkdir(parents=True)

    persistent_cache_lib = persistent_cache / "lib"
    if not persistent_cache_lib.exists():
        persistent_cache_lib.mkdir()

    with mock.patch.dict(
        os.environ,
        {"TORCHINDUCTOR_CACHE_DIR": persistent_cache_lib.absolute().as_posix()},
    ):
        try:
            kernel_lib_path = torch._export.aot_compile(
                f,
                args,
                kwargs,
                dynamic_shapes=dynamic_shapes,
                remove_runtime_assertions=remove_runtime_assertions,
                disable_constraint_solver=disable_constraint_solver,
                # Some operations may have non-Tensor parameters like int, float, bool. These
                # non-Tensor parameters will not be the input of the graph. Therefore, we do
                # need to keep the same signature.
                same_signature=False,
            )
            if not isinstance(kernel_lib_path, str):
                raise AssertionError(
                    f"expected kernel_lib_path to be str, got {type(kernel_lib_path)}"
                )

            kernel_metadata_items = []

            for idx, input in enumerate(flattened_inputs):
                # Skip explicit-None args (e.g. an optional `int[1]? dim`) so the
                # written metadata vector matches the cache reader, which also
                # drops None entries.
                if input is None:
                    continue
                if isinstance(input, torch.Tensor):
                    metadata = extract_tensor_metadata(dynamic, input)
                elif isinstance(input, (list, tuple)) and all(
                    isinstance(item, int) and not isinstance(item, bool)
                    for item in input
                ):
                    # `int[]` / `SymInt[]` params. Checked before the Tensor-list
                    # branch so an empty list/tuple is tagged INT_LIST, matching
                    # the cache reader and C++ unpack_input_parameters for an
                    # empty `int[]`. An empty `Tensor[]` is the converse case:
                    # C++ tags it TENSOR_LIST (isTensorList() is checked before
                    # isIntList()) while this tags it INT_LIST -- only a cache
                    # miss/recompile (per-op cache, fixed schema type per slot ->
                    # no aliasing), and it does not arise for real ops
                    # (`cat([])`/`stack([])` raise in eager).
                    metadata = extract_int_list_metadata(list(input))
                elif isinstance(input, (list, tuple)) and all(
                    isinstance(item, torch.Tensor) for item in input
                ):
                    metadata = extract_tensor_list_metadata(dynamic, list(input))
                elif isinstance(input, supported_scalar_types()):
                    metadata = extract_scalar_metadata(device_type, input)
                elif isinstance(input, str):
                    metadata = extract_string_metadata(input)
                elif isinstance(input, torch.dtype):
                    metadata = extract_dtype_metadata(input)
                elif isinstance(input, torch.device):
                    metadata = extract_device_metadata(input)
                elif isinstance(input, torch.layout):
                    metadata = extract_layout_metadata(input)
                else:
                    raise NotImplementedError(f"Unsupported input type: {type(input)}")

                metadata["arg_order"] = idx
                kernel_metadata_items.append(metadata)

            kernel_meta_info: dict[str, Any] = {}
            kernel_meta_info["meta_info"] = kernel_metadata_items
            kernel_meta_info["kernel_path"] = (
                Path(kernel_lib_path).relative_to(persistent_cache).as_posix()
            )

            json_data = []
            update_json = True
            op_conf = persistent_cache / f"{op_func_name_with_overload}.json"
            mode = "r" if op_conf.exists() else "w"
            with aoti_eager_op_conf_lock(op_func_name_with_overload):
                with open(op_conf, mode) as op_conf_file:
                    try:
                        json_data = json.load(op_conf_file)
                    except Exception:
                        json_data = []

                    if not isinstance(json_data, list):
                        raise AssertionError(
                            f"expected json_data to be list, got {type(json_data)}"
                        )
                    for item in json_data:
                        if not isinstance(item, dict):
                            raise AssertionError(
                                f"expected item to be dict, got {type(item)}"
                            )
                        # Same kernel meta info already exists in the json file
                        if item["meta_info"] == kernel_metadata_items:
                            update_json = False
                            break

                if update_json:
                    json_data.append(kernel_meta_info)
                    with open(op_conf, "w") as op_conf_file:
                        json.dump(json_data, op_conf_file, indent=4)

            return kernel_lib_path
        except Exception as e:
            err_msg = f"Failed to compile {op_func_name_with_overload}: {e}"
            log.exception(err_msg)
            return ""
