# mypy: allow-untyped-defs
# [MODULE DOC] Grab-bag of internal utilities: legacy tensor/storage type conversion and device
# transfer, unpickling rebuild functions for tensors (dense, sparse, quantized, nested, meta,
# wrapper subclass, parameters), flatten/unflatten and chunking of tensor lists, exception wrapping
# across threads, device-index and accelerator discovery, class properties, lazy seed tracking,
# callback registry, pickle compatibility name maps for the weights-only unpickler, FX-aware memory
# snapshot augmentation, and c10-synchronized environment variable access with an os.environ hook.
# Import-time side effects: defines module-level state (pending sparse-tensor list, logger,
# name mapping tables extended with Python 2 exception names); installs no hooks by itself.
import _compat_pickle
import copyreg
import functools
import importlib
import logging
import math
import os
import pickle
import re
import sys
import traceback
import warnings
from collections import defaultdict
from collections.abc import Callable
from types import ModuleType
from typing import Any, cast, Generic, TYPE_CHECKING, TypedDict
from typing_extensions import deprecated, NotRequired, ParamSpec

import torch


def _type(self, dtype=None, non_blocking=False, **kwargs):
    # [FUNCTION DOC] HELPER: Implementation of the legacy storage/tensor type method. With no dtype returns the dotted class name string of self. With a dtype (class or dotted-name string, imported on demand) returns self if it is already that type; otherwise converts: sparse to sparse converts values and indices separately, mixing sparse and dense raises RuntimeError; dense uses a new object of the target type copied from self, optionally non-blocking. The deprecated async keyword is accepted via the helper; other keywords raise TypeError.
    """Returns the type if `dtype` is not provided, else casts this object to
    the specified type.

    If this is already of the correct type, no copy is performed and the
    original object is returned.

    Args:
        dtype (type or string): The desired type
        non_blocking (bool): If ``True``, and the source is in pinned memory
            and destination is on the GPU or vice versa, the copy is performed
            asynchronously with respect to the host. Otherwise, the argument
            has no effect.
        **kwargs: For compatibility, may contain the key ``async`` in place of
            the ``non_blocking`` argument. The ``async`` arg is deprecated.
    """
    non_blocking = _get_async_or_non_blocking("type", non_blocking, kwargs)
    if dtype is None:
        return self.__module__ + "." + self.__class__.__name__

    if isinstance(dtype, str):
        dtype = _import_dotted_name(dtype)
    if dtype is type(self):
        return self
    if self.is_sparse:
        if not dtype.is_sparse:
            raise RuntimeError("Cannot cast sparse tensor to dense tensor")
        new_module_name = dtype.__module__.replace(".sparse", "")
        new_values_type_name = new_module_name + "." + dtype.__name__
        new_values = torch.Tensor._values(self).type(new_values_type_name, non_blocking)
        new_indices_type_name = new_module_name + ".LongTensor"
        new_indices = torch.Tensor._indices(self).type(
            new_indices_type_name, non_blocking
        )
        return dtype(new_indices, new_values, self.size())
    if dtype.is_sparse:
        raise RuntimeError("Cannot cast dense tensor to sparse tensor")
    return dtype(self.size()).copy_(self, non_blocking)


def _to(self, device, non_blocking=False):
    # [FUNCTION DOC] HELPER: Returns self if already on the device. For a CPU target returns an untyped storage of the same byte count (pinned when non_blocking and the source is on an accelerator device) copied from self. For other devices requires the device module to exist (else AssertionError); copies sparse data via indices and values when the device module has a sparse namespace, raises AssertionError for sparse otherwise, and for dense creates an untyped storage on the device and copies into it.
    """Returns a copy of this object in device memory.

    If this object is already on the correct device, then no copy is performed
    and the original object is returned.

    Args:
        device (int): The destination device.
        non_blocking (bool): If ``True`` and the source is in pinned memory,
            the copy will be asynchronous with respect to the host. Otherwise,
            the argument has no effect.
    """
    if self.device == device:
        return self

    if device.type == "cpu":
        pin_memory = non_blocking and self.device.type in (
            "cuda",
            "xpu",
            "mtia",
            torch._C._get_privateuse1_backend_name(),
        )
        untyped_storage = torch.empty(
            self.nbytes(), dtype=torch.uint8, device=device, pin_memory=pin_memory
        ).untyped_storage()
        untyped_storage.copy_(self, non_blocking)
        return untyped_storage

    device_module = getattr(torch, device.type, None)
    if device_module is None:
        raise AssertionError(f"{device.type.upper()} device module is not loaded")
    with device_module.device(device):
        if self.is_sparse and hasattr(device_module, "sparse"):
            new_type = getattr(device_module.sparse, self.__class__.__name__)
            indices = getattr(torch.Tensor._indices(self), device.type)(
                device, non_blocking
            )
            values = getattr(torch.Tensor._values(self), device.type)(
                device, non_blocking
            )
            return new_type(indices, values, self.size())
        else:
            if self.is_sparse:
                raise AssertionError(
                    f"sparse storage is not supported for {device.type.upper()} tensors"
                )
            untyped_storage = torch.UntypedStorage(self.size(), device=device)
            untyped_storage.copy_(self, non_blocking)
            return untyped_storage


def _get_async_or_non_blocking(function_name, non_blocking, kwargs):
    # [FUNCTION DOC] HELPER: Resolves the non_blocking flag from legacy keyword arguments. Returns the default when no extra kwargs; if the only kwarg is async, warns about deprecation and returns its value; any other kwargs raise TypeError naming the function and the offending argument.
    """Return the non-blocking flag given the function name and kwargs.

    Args:
        function_name (str): the name of the function being used.
        non_blocking (bool): the default value.
        **kwargs (dict): the kwargs passed to the function.
    """
    if not kwargs:
        return non_blocking
    if len(kwargs) != 1 or "async" not in kwargs:
        message = "{}() got an unexpected keyword argument '{}'"
        argument = list(kwargs.keys()).pop()
        raise TypeError(message.format(function_name, argument))
    warnings.warn("'async' is deprecated; use 'non_blocking'", stacklevel=2)
    return kwargs["async"]


def _get_restore_location(device):
    # [FUNCTION DOC] HELPER: Applies the thread-local serialization map_location to a device. No map_location returns the device; a mapping returns the mapped value (or the device if missing); a string or device is returned directly; a callable map_location raises RuntimeError (unsupported for these rebuild paths).
    """Return the map_location location.

    Used for rebuild functions where the tensor device is distinct from the storage
    """

    map_location = torch.serialization._serialization_tls.map_location
    if map_location is None:
        return device
    else:
        if isinstance(map_location, dict):
            return map_location.get(device, device)
        elif isinstance(map_location, (str, torch.device)):
            return map_location
        else:
            if not callable(map_location):
                raise AssertionError(
                    f"expected callable map_location, got {type(map_location).__name__}"
                )
            raise RuntimeError(
                "Callable map_location not supported with _rebuild_wrapper_subclass "
                "or _rebuild_device_tensor_from_numpy"
            )


# Note [Don't serialize hooks]
# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~
# Since time immemorial, we have serialized the backward hooks associated with
# variables.  This kind of half-worked--Python can pickle global functions
# (but not closures!)--but there were problems.
#
#   - It's fragile.  If you serialize a backward hook into a saved
#     model, and then you rename the function associated with the hook,
#     now your saved model is broken and you can't load it anymore.
#
#   - It's not actually used.  The standard recommendation is to
#     serialize the *state_dict* of a model, not the model itself
#     (since this is more stable to code changes affecting the model
#     serialization), and the state dict saves "data" only, thus
#     stripping the backward hooks.  In some cases, hooks are
#     essential to the well-functioning of a model (e.g., DDP),
#     but DDP already manages re-adding the hooks!
#
#   - We didn't serialize them in many cases.  Prior to #10220, we
#     were dropping backward hooks in ForkingPickler.  We "fixed" this
#     to be convenient with other serialization sites, but lack of
#     serializing backward hooks wasn't actually the root cause of
#     the bug.
#
# With these cases in mind, we have decided that a better strategy
# is to just NOT serialize hooks at all.
#
# Since this is a BC-breaking change, we should warn when we previously
# serialized a hook, but no longer do so. This will be done by adding a special
# sentinel property to hooks will be used to suppress this warning. If a hook
# has the property _torch_serialize_ignore, we will not emit a warning if we
# attempt to serialize a Tensor with this hook attached to it.
#
# By the way, when _backward_hooks is skipped, we must give an EMPTY
# OrderedDict(), if you pass a None you'll run afoul #12219.


# TODO: Once we decide to break serialization FC, `storage` no longer needs to
# be a TypedStorage
def _rebuild_tensor(storage, storage_offset, size, stride):
    # [FUNCTION DOC] HELPER: Builds a tensor sharing the given storage: makes an empty tensor with the storage dtype and device, then points it at the underlying untyped storage with the given offset, size and stride. Returns the tensor.
    # first construct a tensor with the correct dtype/device
    t = torch.empty((0,), dtype=storage.dtype, device=storage._untyped_storage.device)
    return t.set_(storage._untyped_storage, storage_offset, size, stride)


def get_tensor_metadata(tensor):
    # [FUNCTION DOC] Returns the serializable metadata mapping of a tensor (flags such as conj and neg bits). Raises AssertionError if the argument is not a tensor.
    # Tensor's Metadata for serializing.
    # Currently, this only returns a dict[string, bool] specifying whether
    # `conj` or `neg` bit is set.
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(tensor).__name__}")
    return torch._C._get_tensor_metadata(tensor)  # type: ignore[attr-defined]


def set_tensor_metadata(tensor, metadata):
    # [FUNCTION DOC] Applies a metadata mapping (as produced by get_tensor_metadata) onto a tensor in place. Raises AssertionError if metadata is not a dict or tensor is not a tensor.
    # See `get_tensor_metadata` above
    if not isinstance(metadata, dict):
        raise AssertionError(f"expected dict, got {type(metadata).__name__}")
    if not isinstance(tensor, torch.Tensor):
        raise AssertionError(f"expected torch.Tensor, got {type(tensor).__name__}")
    torch._C._set_tensor_metadata(tensor, metadata)  # type: ignore[attr-defined]


def _restore_device_fake_mode(tensor):
    # [FUNCTION DOC] HELPER: When a fake tensor mode is active and the tensor's storage records a fake device, sets the tensor's fake device to that device after applying the map_location mapping. Returns the tensor.
    if torch._guards.detect_fake_mode(None) is not None:
        if tensor.untyped_storage()._fake_device is not None:
            device = _get_restore_location(tensor.untyped_storage()._fake_device)
            if not isinstance(device, torch.device):
                device = torch.device(device)
            tensor.fake_device = torch.device(device)
    return tensor


def _rebuild_tensor_v2(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    metadata=None,
):
    # [FUNCTION DOC] HELPER: Unpickling entry point for tensors: rebuilds from storage, offset, size, stride; sets requires_grad; applies optional metadata; assigns the (expected empty) backward hooks for backward compatibility; restores fake device. Returns the tensor.
    tensor = _rebuild_tensor(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    if metadata:
        set_tensor_metadata(tensor, metadata)

    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks

    tensor = _restore_device_fake_mode(tensor)
    return tensor


def _rebuild_tensor_v3(
    storage,
    storage_offset,
    size,
    stride,
    requires_grad,
    backward_hooks,
    dtype,
    metadata=None,
):
    # [FUNCTION DOC] HELPER: Like the v2 rebuild but takes an explicit dtype (allowing a dtype different from the storage's typed view) and creates the empty tensor with that dtype and requires_grad directly. Applies metadata, hooks and fake-device restore. Returns the tensor.
    t = torch.empty(
        (0,),
        dtype=dtype,
        device=storage._untyped_storage.device,
        requires_grad=requires_grad,
    )
    t.set_(storage._untyped_storage, storage_offset, size, stride)
    if metadata:
        set_tensor_metadata(t, metadata)
    t._backward_hooks = backward_hooks
    t = _restore_device_fake_mode(t)
    return t


_sparse_tensors_to_validate: list["torch.Tensor"] = []


# In _legacy_load() in serialization.py we unpickle storages after the sparse
# tensors have been already unpickled. Those storages contain data necessary for
# validating sparse tensors: indices and values. That's why sparse tensors are
# first unpickled without any validation, and then this function is called just
# before _legacy_load() returns, so that all the sparse tensors can be validated
# in bulk.
#
# The same procedure must be followed by _load() in serialization.py because due
# to Pickler semantics, we have to use the same (non-validating) function for
# unpickling sparse tensors, regardless of the caller.
def _validate_loaded_sparse_tensors(weights_only=False):
    # [FUNCTION DOC] HELPER: Validates, in bulk, the sparse tensors collected in the module-level pending list during unpickling. Returns immediately if the list is empty. With weights_only it always validates (and warns about cost); otherwise only when the global sparse invariant checking is enabled. Dispatches validation by layout (COO or the four compressed layouts, pinning check off); other layouts raise NotImplementedError. The pending list is always cleared afterwards.
    # In weights_only mode we always validate: malformed sparse indices can
    # cause out-of-bounds reads later (e.g. in to_dense()), and an untrusted
    # checkpoint must not be able to slip those through. Otherwise we fall back
    # to the global check_sparse_tensor_invariants setting.
    try:
        if not _sparse_tensors_to_validate:
            return
        if weights_only:
            warnings.warn(
                "Validating sparse tensor invariants because weights_only=True; "
                "this is an O(nnz) scan per sparse tensor and may be slow for "
                "large checkpoints.",
                stacklevel=2,
            )
        elif not torch.sparse.check_sparse_tensor_invariants().is_enabled():
            return
        # We disable pinning check (see check_pinning=False below) to
        # avoid gh-153143. In fact, pinning check is unnecessary
        # anyway when loading sparse data from external sources.
        for t in _sparse_tensors_to_validate:
            if t.layout is torch.sparse_coo:
                torch._validate_sparse_coo_tensor_args(
                    t._indices(),
                    t._values(),
                    t.size(),
                    t.is_coalesced(),
                    check_pinning=False,
                )
            elif t.layout in {
                torch.sparse_csr,
                torch.sparse_csc,
                torch.sparse_bsr,
                torch.sparse_bsc,
            }:
                # TODO: Validation currently involves an expensive traversal
                # on CPU, which may include a device transfer.
                if t.layout in {torch.sparse_csr, torch.sparse_bsr}:
                    compressed_indices, plain_indices = (
                        t.crow_indices(),
                        t.col_indices(),
                    )
                else:
                    compressed_indices, plain_indices = (
                        t.ccol_indices(),
                        t.row_indices(),
                    )
                torch._validate_sparse_compressed_tensor_args(
                    compressed_indices,
                    plain_indices,
                    t.values(),
                    t.size(),
                    t.layout,
                    check_pinning=False,
                )
            else:
                raise NotImplementedError(
                    f"_validate_loaded_sparse_tensors for layout `{t.layout}`"
                )

    finally:
        _sparse_tensors_to_validate.clear()


def _rebuild_sparse_tensor(layout, data):
    # [FUNCTION DOC] HELPER: Unpickling entry point for sparse tensors. COO data is (indices, values, size[, coalesced flag]) with the old 3-tuple form accepted; compressed layouts take (compressed indices, plain indices, values, size). Creates the tensor without invariant checks, appends it to the pending-validation list, and returns it. Unknown layouts raise NotImplementedError.
    """
    Rebuilds a sparse tensor from its sparse storage representation.

    Args:
        layout (str): The sparse storage layout of the tensor.
        data (tuple): The tensor's sparse storage representation.
    """
    if layout == torch.sparse_coo:
        if len(data) == 3:
            # For BC:
            indices, values, size = data
            is_coalesced = None
        else:
            indices, values, size, is_coalesced = data
        result = torch.sparse_coo_tensor(
            indices, values, size, check_invariants=False, is_coalesced=is_coalesced
        )
        _sparse_tensors_to_validate.append(result)
        return result

    elif layout in {
        torch.sparse_csr,
        torch.sparse_csc,
        torch.sparse_bsr,
        torch.sparse_bsc,
    }:
        compressed_indices, plain_indices, values, size = data
        result = torch.sparse_compressed_tensor(
            compressed_indices,
            plain_indices,
            values,
            size,
            layout=layout,
            check_invariants=False,
        )
        _sparse_tensors_to_validate.append(result)
        return result

    raise NotImplementedError(f"rebuilding sparse tensor for layout {layout}")


def _rebuild_nested_tensor(buffer, sizes, strides, storage_offsets):
    # [FUNCTION DOC] HELPER: Rebuilds a nested tensor as a view over a buffer using the given sizes, strides and storage offsets.
    return torch._nested_view_from_buffer(buffer, sizes, strides, storage_offsets)


def _rebuild_device_tensor_from_cpu_tensor(data, dtype, device, requires_grad):
    # [FUNCTION DOC] HELPER: Maps the device through map_location, moves the CPU data to the given dtype and device, sets requires_grad, and returns it.
    device = _get_restore_location(device)
    tensor = data.to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


def _rebuild_device_tensor_from_numpy(data, dtype, device, requires_grad):
    # [FUNCTION DOC] HELPER: Maps the device through map_location, converts the numpy array to a tensor on the given dtype and device, sets requires_grad, and returns it.
    device = _get_restore_location(device)
    tensor = torch.from_numpy(data).to(dtype=dtype, device=device)
    tensor.requires_grad = requires_grad
    return tensor


# Should not be used, only here to be able to load Tensors serialized with older versions of pytorch
_rebuild_xla_tensor = _rebuild_device_tensor_from_numpy


def _rebuild_meta_tensor_no_storage(dtype, size, stride, requires_grad):
    # [FUNCTION DOC] HELPER: Creates a storage-less tensor on the meta device with the given dtype, size, stride and requires_grad.
    return torch.empty_strided(
        size, stride, dtype=dtype, device="meta", requires_grad=requires_grad
    )


def _rebuild_wrapper_subclass(
    cls,
    dtype,
    size,
    stride,
    storage_offset,
    layout,
    device,
    requires_grad,
):
    # [FUNCTION DOC] HELPER: Maps the device through map_location and creates a wrapper-subclass tensor of class cls with the given size, stride, dtype, offset, layout and requires_grad (no real storage).
    device = _get_restore_location(device)
    return torch.Tensor._make_wrapper_subclass(
        cls,
        size,
        strides=stride,
        dtype=dtype,
        storage_offset=storage_offset,
        layout=layout,
        device=device,
        requires_grad=requires_grad,
    )


# TODO: Once we decide to break serialization FC, `storage` no longer needs to
# be a TypedStorage
def _rebuild_qtensor(
    storage,
    storage_offset,
    size,
    stride,
    quantizer_params,
    requires_grad,
    backward_hooks,
):
    # [FUNCTION DOC] HELPER: Rebuilds a quantized tensor from storage and quantizer parameters. Per-tensor affine uses scale and zero point. Per-channel schemes validate the axis range and that scales and zero points both have length equal to the size along the axis (ValueError otherwise, to guard against untrusted checkpoints), converting list inputs to tensors of the appropriate dtype. Unknown schemes raise RuntimeError. Then points the tensor at the storage, sets requires_grad and the legacy hooks, and returns it.
    qscheme = quantizer_params[0]
    if qscheme == torch.per_tensor_affine:
        _, scale, zero_point = quantizer_params
        tensor = torch._empty_affine_quantized(
            size,
            scale=scale,
            zero_point=zero_point,
            dtype=storage.dtype,
            device=storage.device,
        )
    elif qscheme in (torch.per_channel_affine, torch.per_channel_affine_float_qparams):
        _, scales, zero_points, axis = quantizer_params
        # The C++ constructor does not bound-check axis or the
        # scales/zero_points length against size; values arriving from an
        # untrusted weights_only checkpoint can otherwise be arbitrary.
        if not 0 <= axis < len(size):
            raise ValueError(
                f"_rebuild_qtensor: per_channel axis {axis} out of range for size {tuple(size)}"
            )
        expected_len = int(size[axis])
        scales_len = len(scales) if isinstance(scales, list) else scales.numel()
        zero_points_len = (
            len(zero_points) if isinstance(zero_points, list) else zero_points.numel()
        )
        if scales_len != expected_len or zero_points_len != expected_len:
            raise ValueError(
                "_rebuild_qtensor: per_channel scales/zero_points length must equal "
                f"size[axis]={expected_len}, got scales={scales_len}, "
                f"zero_points={zero_points_len}"
            )
        if type(scales) is list and type(zero_points) is list:
            if qscheme == torch.per_channel_affine:
                scales = torch.tensor(scales, dtype=torch.double, device=storage.device)
                zero_points = torch.tensor(
                    zero_points, dtype=torch.long, device=storage.device
                )
            else:
                scales = torch.tensor(scales, dtype=torch.float, device=storage.device)
                zero_points = torch.tensor(
                    zero_points, dtype=torch.float, device=storage.device
                )
        tensor = torch._empty_per_channel_affine_quantized(
            size,
            scales=scales,
            zero_points=zero_points,
            axis=axis,
            dtype=storage.dtype,
            device=storage.device,
        )
    else:
        raise RuntimeError(f"Can't deserialize quantized tensor with qscheme {qscheme}")
    tensor.set_(storage, storage_offset, size, stride)
    tensor.requires_grad = requires_grad
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    tensor._backward_hooks = backward_hooks
    return tensor


def _rebuild_parameter(data, requires_grad, backward_hooks):
    # [FUNCTION DOC] HELPER: Creates a Parameter from data and requires_grad, assigns the legacy backward hooks, and returns it.
    param = torch.nn.Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    return param


def _rebuild_parameter_with_state(data, requires_grad, backward_hooks, state):
    # [FUNCTION DOC] HELPER: Like the parameter rebuild but also restores saved Python object state (attributes and slots) onto the parameter. Returns it.
    param = torch.nn.Parameter(data, requires_grad)
    # NB: This line exists only for backwards compatibility; the
    # general expectation is that backward_hooks is an empty
    # OrderedDict.  See Note [Don't serialize hooks]
    param._backward_hooks = backward_hooks

    # Restore state on Parameter like python attr.
    param = _set_obj_state(param, state)
    return param


def _get_obj_state(obj):
    # [FUNCTION DOC] HELPER: Returns the picklable state of a Python object: the result of its state hook if it has one; otherwise a pair of attribute mapping and slot-value mapping when it has slots, else its attribute mapping.
    # Get the state of the python subclass
    # This loosely mimics the function on the object class but since Tensor do not inherit
    # from it, we cannot call that function directly
    # https://github.com/python/cpython/blob/c83919bd635f4433f1c6ae8504996a9fe3c215e5/Objects/typeobject.c#L4891
    # Note that starting with Python 3.11, this `__getstate__` is always defined and thus
    # the else branch will never be taken.
    getstate_fn = getattr(obj, "__getstate__", None)
    if getstate_fn:
        state = getstate_fn()
    else:
        slots_to_save = copyreg._slotnames(obj.__class__)  # type: ignore[attr-defined]
        if slots_to_save:
            state = (
                obj.__dict__,
                {
                    name: getattr(obj, name)
                    for name in slots_to_save
                    if hasattr(obj, name)
                },
            )
        else:
            state = obj.__dict__

    return state


def _set_obj_state(obj, state):
    # [FUNCTION DOC] HELPER: Restores state produced by the state getter onto obj. A tuple state must have exactly two items (attribute mapping, slots mapping) else RuntimeError; sets each non-empty mapping entry as an attribute. Returns obj.
    if isinstance(state, tuple):
        if not len(state) == 2:
            raise RuntimeError(f"Invalid serialized state: {state}")
        dict_state = state[0]
        slots_state = state[1]
    else:
        dict_state = state
        slots_state = None

    # Starting with Python 3.11, the __dict__ attribute is lazily created
    # and is serialized as None when not needed.
    if dict_state:
        for k, v in dict_state.items():
            setattr(obj, k, v)

    if slots_state:
        for k, v in slots_state.items():
            setattr(obj, k, v)
    return obj


def _import_dotted_name(name):
    # [FUNCTION DOC] HELPER: Imports the first component of a dotted name and resolves the remaining components as attributes; returns the final object. Errors from import or attribute lookup propagate.
    components = name.split(".")
    obj = __import__(components[0])
    for component in components[1:]:
        obj = getattr(obj, component)
    return obj


def _flatten_dense_tensors(tensors):
    # [FUNCTION DOC] HELPER: Concatenates dense tensors of the same type into one contiguous 1D buffer via the native helper; returns the buffer.
    """Flatten dense tensors into a contiguous 1D buffer. Assume tensors are of
    same dense type.

    Since inputs are dense, the resulting tensor will be a concatenated 1D
    buffer. Element-wise operation on this buffer will be equivalent to
    operating individually.

    Args:
        tensors (Iterable[Tensor]): dense tensors to flatten.

    Returns:
        A contiguous 1D buffer containing input tensors.
    """
    return torch._C._nn.flatten_dense_tensors(tensors)


def _flatten_sparse_tensors(tensors):
    # [FUNCTION DOC] HELPER: Flattens the indices of all sparse tensors into one 1D buffer and their values into another; returns the (indices, values) pair.
    """Flatten sparse tensors into two contiguous 1D buffers, one of indices and
    one of values. Assume tensors are of same sparse type.

    Args:
        tensors (Iterable[Tensor]): sparse tensors to flatten.

    Returns:
        A tuple of two contiguous 1D buffers, one containing input tensors'
        indices and the other containing the values.
    """
    flat_indices = torch._C._nn.flatten_dense_tensors(
        [torch.Tensor._indices(t) for t in tensors]
    )
    flat_values = torch._C._nn.flatten_dense_tensors(
        [torch.Tensor._values(t) for t in tensors]
    )
    return flat_indices, flat_values


def _unflatten_dense_tensors(flat, tensors):
    # [FUNCTION DOC] HELPER: Returns views of the flat buffer shaped like each of the given tensors (inverse of the dense flatten).
    """View a flat buffer using the sizes of tensors. Assume that tensors are of
    same dense type, and that flat is given by _flatten_dense_tensors.

    Args:
        flat (Tensor): flattened dense tensors to unflatten.
        tensors (Iterable[Tensor]): dense tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened dense tensors with sizes same as tensors and values from
        flat.
    """
    return torch._C._nn.unflatten_dense_tensors(flat, tensors)


def _unflatten_sparse_tensors(flat, tensors):
    # [FUNCTION DOC] HELPER: Splits the (flat indices, flat values) pair back into per-tensor indices and values using the reference sparse tensors' shapes, and builds a new sparse tensor of each reference's type with its size. Returns a tuple of them.
    """View flat buffer (containing indices and values) using the sizes of
    tensors. Assume that tensors are of same sparse type, and that flat is given
    by _flatten_sparse_tensors.

    Args:
        flat (tuple(Tensor, Tensor)): flattened indices and values of sparse
          tensors to unflatten.
        tensors (Iterable[Tensor]): sparse tensors whose sizes will be used to
          unflatten flat.

    Returns:
        Unflattened sparse tensors with sizes same as tensors and values from
        flat.
    """
    flat_indices, flat_values = flat
    indices = torch._C._nn.unflatten_dense_tensors(
        flat_indices, [torch.Tensor._indices(t) for t in tensors]
    )
    values = torch._C._nn.unflatten_dense_tensors(
        flat_values, [torch.Tensor._values(t) for t in tensors]
    )
    outputs = []
    for t, i, v in zip(tensors, indices, values):
        outputs.append(t.new(i, v, t.size()))
    return tuple(outputs)


def _reorder_tensors_as(tensors, ordered_tensors):
    # [FUNCTION DOC] HELPER: Given tensors grouped by type (as produced by chunking), returns a tuple reordered to follow the type sequence of ordered_tensors; consumes each type's tensors in their existing order. Raises StopIteration if a type runs out.
    """Assume that tensors are of same order as ordered_tensors within their
    types, e.g., from _take_tensors. Reorder them to be of same order as
    ordered_tensors.

    Args:
        tensors (Iterable[Tensor]): tensors to be reordered. They should be of
          the same order as ordered_tensors within their own types.
        ordered_tensors (Iterable[Tensor]): tensors whose order will be the
          reference.

    Returns:
        Ordered tuple of tensors with contents from tensors and order of
        ordered_tensors.
    """
    type_dict = defaultdict(list)
    for tensor in tensors:
        type_dict[tensor.type()].append(tensor)
    type_dict_ = {t: iter(coll) for t, coll in type_dict.items()}
    return tuple(next(type_dict_[tensor.type()]) for tensor in ordered_tensors)


def _take_tensors(tensors, size_limit):
    # [FUNCTION DOC] HELPER: Generator grouping tensors into chunks of the same type whose total byte size stays within size_limit (sparse tensors counted as indices plus values bytes). Yields a full chunk when adding the next tensor would exceed the limit (a chunk always holds at least one tensor), then yields remaining partial chunks at the end. Order is preserved only within each type.
    """Group tensors into chunks. This generator yields a chunk at each time,
    each containing tensors of same type up to certain byte limit in total size.

    Args:
        tensors (Sequence): A sequence of tensors to be separated into chunks.
        size_limit (int): The limit of each chunk in bytes.

    Yields:
        Blocks of tensors of same type and within size_limit. The yielded
        tensors are only ordered as the original sequence within its types.
    """
    buf_dict: defaultdict[str, list] = defaultdict(lambda: [[], 0])
    for tensor in tensors:
        t = tensor.type()
        if tensor.is_sparse:
            indices = torch.Tensor._indices(tensor)
            values = torch.Tensor._values(tensor)
            size = (
                indices.numel() * indices.element_size()
                + values.numel() * values.element_size()
            )
        else:
            size = tensor.numel() * tensor.element_size()
        buf_and_size = buf_dict[t]
        if buf_and_size[1] + size > size_limit and buf_and_size[1] > 0:
            yield buf_and_size[0]
            buf_and_size = buf_dict[t] = [[], 0]
        buf_and_size[0].append(tensor)  # pyrefly: ignore [missing-attribute]
        buf_and_size[1] += size  # pyrefly: ignore [unsupported-operation]
    for buf, _ in buf_dict.values():
        if len(buf) > 0:
            yield buf


def render_call(fn, args, kwargs):
    # [FUNCTION DOC] Returns a string rendering of a call: the resolved function name (or its string form) followed by the repr of args and keyword args, with tensor printing suppressed to summaries (threshold and edge items zero).
    str_fn = torch.overrides.resolve_name(fn)
    if str_fn is None:
        str_fn = str(fn)

    str_args: list[str] = []
    with torch._tensor_str.printoptions(threshold=0, edgeitems=0):
        str_args.extend(repr(a) for a in args)
        str_args.extend(f"{k}={repr(v)}" for k, v in kwargs.items())
        r = f"{str_fn}({', '.join(str_args)})"
    return r


# NOTE [ Python Traceback Reference Cycle Problem ]
#
# When using sys.exc_info(), it is important to **not** store the exc_info[2],
# which is the traceback, because otherwise you will run into the traceback
# reference cycle problem, i.e., the traceback holding reference to the frame,
# and the frame (which holds reference to all the object in its temporary scope)
# holding reference the traceback.


class KeyErrorMessage(str):
    # [CLASS DOC] HELPER: String subclass whose repr is the string itself, so KeyError messages print readably instead of quoted/escaped.
    r"""str subclass that returns itself in repr"""

    __slots__ = ()

    def __repr__(self):
        # [FUNCTION DOC] HELPER: Returns self unchanged (no quoting).
        return self


class ExceptionWrapper:
    # [CLASS DOC] Captures an exception's type and formatted traceback text (not the traceback object, avoiding reference cycles) plus a location phrase, so it can be sent across threads or processes and re-raised. Attributes: exc_type, exc_msg, where.
    r"""Wraps an exception plus traceback to communicate across threads"""

    def __init__(self, exc_info=None, where="in background"):
        # [FUNCTION DOC] Records the exception type and the formatted traceback text from the given exc_info triple, or from the currently handled exception when None; stores the where phrase.
        # It is important that we don't store exc_info, see
        # NOTE [ Python Traceback Reference Cycle Problem ]
        if exc_info is None:
            exc_info = sys.exc_info()
        self.exc_type = exc_info[0]

        self.exc_msg = "".join(traceback.format_exception(*exc_info))
        self.where = where

    def reraise(self):
        # [FUNCTION DOC] Raises, in the current thread, an exception of the original type with a message of the form 'Caught <type> <where>' followed by the original traceback text. KeyError messages use a readable-repr wrapper. Types with a message attribute are constructed with a message keyword. If the type cannot be constructed from a single message, raises RuntimeError with that message instead. Never returns normally.
        r"""Reraises the wrapped exception in the current thread"""
        # Format a message such as: "Caught ValueError in DataLoader worker
        # process 2. Original Traceback:", followed by the traceback.
        msg = f"Caught {self.exc_type.__name__} {self.where}.\nOriginal {self.exc_msg}"  # pyrefly: ignore [missing-attribute]
        if self.exc_type is KeyError:
            # KeyError calls repr() on its argument (usually a dict key). This
            # makes stack traces unreadable. It will not be changed in Python
            # (https://bugs.python.org/issue2651), so we work around it.
            msg = KeyErrorMessage(msg)
        elif getattr(self.exc_type, "message", None):
            # Some exceptions have first argument as non-str but explicitly
            # have message field
            # pyrefly: ignore [not-callable]
            raise self.exc_type(
                # pyrefly: ignore [unexpected-keyword]
                message=msg
            )
        try:
            exception = self.exc_type(msg)  # pyrefly: ignore [not-callable]
        except Exception:
            # If the exception takes multiple arguments or otherwise can't
            # be constructed, don't try to instantiate since we don't know how to
            raise RuntimeError(msg) from None
        raise exception


def cpu_count() -> int | None:
    # [FUNCTION DOC] Returns the number of CPUs usable by this process (affinity-aware where supported), else the total CPU count, which may be None.
    """Return the number of CPUs available to the current process.

    Prefers ``os.sched_getaffinity`` (respects cgroups / taskset) and
    falls back to ``os.cpu_count``.
    """
    # os.process_cpu_count was added in CPython 3.13, see
    # https://docs.python.org/3/library/os.html#os.process_cpu_count
    if hasattr(os, "sched_getaffinity"):
        return len(os.sched_getaffinity(0))
    return os.cpu_count()


def _get_available_device_type():
    # [FUNCTION DOC] HELPER: Returns the first available accelerator type name checking in order cuda, mps, xpu, mtia, then a registered private-use backend; None when none is available.
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    if hasattr(torch, "xpu") and torch.xpu.is_available():  # type: ignore[attr-defined]
        return "xpu"
    if hasattr(torch, "mtia") and torch.mtia.is_available():
        return "mtia"
    custom_backend_name = torch._C._get_privateuse1_backend_name()
    custom_device_mod = getattr(torch, custom_backend_name, None)
    if custom_device_mod and custom_device_mod.is_available():
        return custom_backend_name
    # add more available device types here
    return None


def _get_device_attr(get_member):
    # [FUNCTION DOC] HELPER: Applies the getter function to the runtime module of the available accelerator type and returns its result; returns None when no accelerator is available.
    device_type = _get_available_device_type()
    if device_type and device_type.lower() == "cuda":
        return get_member(torch.cuda)
    if device_type and device_type.lower() == "mps":
        return get_member(torch.mps)
    if device_type and device_type.lower() == "xpu":
        return get_member(torch.xpu)  # type: ignore[attr-defined]
    if device_type and device_type.lower() == "mtia":
        return get_member(torch.mtia)
    if device_type == torch._C._get_privateuse1_backend_name():
        return get_member(getattr(torch, device_type))
    # add more available device types here
    return None


def _get_current_device_index():
    # [FUNCTION DOC] HELPER: Returns the current device index of the available accelerator type, or None if none.
    # current device index
    return _get_device_attr(lambda m: m.current_device())


def _get_all_device_indices():
    # [FUNCTION DOC] HELPER: Returns the list of all device indices of the available accelerator type (None if no accelerator).
    # all device index
    return _get_device_attr(lambda m: list(range(m.device_count())))


def _get_devices_properties(device_ids):
    # [FUNCTION DOC] HELPER: Returns a list of device property objects for each given device index on the available accelerator. KNOWN ISSUE: with no accelerator each entry is None.
    # all device properties
    return [_get_device_attr(lambda m: m.get_device_properties(i)) for i in device_ids]


def get_current_device_index() -> int:
    # [FUNCTION DOC] Returns the current CUDA device index, or -1 when no CUDA devices exist. Scriptable counterpart of the generic current-device helper.
    r"""Checks if there are CUDA devices available and
    returns the device index of the current default CUDA device.
    Returns -1 in case there are no CUDA devices available.
    Arguments: ``None``
    """
    if torch.cuda.device_count() > 0:
        return torch.cuda.current_device()
    return -1


def _get_device_index(
    device: Any,
    optional: bool = False,
    allow_cpu: bool = False,
) -> int:
    # [FUNCTION DOC] HELPER: Normalizes a device spec to an integer index. Strings become devices; CPU devices raise ValueError unless allow_cpu (then -1); device objects give their index; integers are returned as is. If no index results: when optional, returns the current device index (scripting-aware), otherwise raises ValueError. Note a device object without index yields no index and so follows the same path.
    r"""Gets the device index from :attr:`device`, which can be a torch.device
    object, a Python integer, or ``None``.

    If :attr:`device` is a torch.device object, returns the device index if it
    has index. Note that for a device without a specified index,
    i.e., ``torch.device('xxx')``, this will return the current default
    device of that type if :attr:`optional` is ``True``. If :attr:`allow_cpu` is ``True``,
    CPU devices will be accepted and ``-1`` will be returned in this case.

    If :attr:`device` is a Python integer, it is returned as is.

    If :attr:`device` is ``None``, this will return the current default
    device of the supported runtime platform if :attr:`optional` is ``True``.
    i.e., the current default CUDA device will be returned if CUDA runtime is supported.
    """
    if isinstance(device, str):
        device = torch.device(device)
    device_idx: int | None = None
    if isinstance(device, torch.device):
        if not allow_cpu and device.type == "cpu":
            raise ValueError(f"Expected a non cpu device, but got: {device}")
        device_idx = -1 if device.type == "cpu" else device.index
    if isinstance(device, int):
        device_idx = device
    if device_idx is None:
        if optional:
            # The eager API _get_current_device_index uses `lambda` functions which are
            # not supported in JIT and hence not scriptable. The JIT equivalent API to get
            # the current device index is `get_current_device_index()` which can
            # be scripted. We use is_scripting to check the mode we are in and call the
            # appropriate API.
            if torch.jit.is_scripting():
                device_idx = get_current_device_index()
            else:
                device_idx = _get_current_device_index()
        else:
            raise ValueError(
                f"Expected a torch.device with a specified index or an integer, but got:{device}"
            )
    return device_idx


def _handle_complex(tensor):
    # [FUNCTION DOC] HELPER: Returns a real-valued view of a complex tensor, or the tensor itself otherwise; uninitialized lazy parameters are passed through without querying complexity.
    """
    Returns a real view of a tensor if complex dtype else just the tensor
    need to check if a UninitializedParameter because otherwise checking is_complex is an error for a LazyModule
    """
    return (
        torch.view_as_real(tensor)
        if not isinstance(tensor, torch.nn.UninitializedParameter)
        and tensor.is_complex()
        else tensor
    )


def _element_size(dtype):
    # [FUNCTION DOC] HELPER: Returns bytes per element of a dtype: complex via bit width divided by 4, floating via width divided by 8, bool is 1, other integers via width divided by 8. Raises RuntimeError if not a dtype.
    """
    Returns the element size for a dtype, in bytes
    """
    if not isinstance(dtype, torch.dtype):
        raise RuntimeError(f"expected torch.dtype, but got {type(dtype)}")

    if dtype.is_complex:
        return torch.finfo(dtype).bits >> 2
    elif dtype.is_floating_point:
        return torch.finfo(dtype).bits >> 3
    elif dtype == torch.bool:
        # NOTE: torch.bool is not supported in torch.iinfo()
        return 1
    else:
        return torch.iinfo(dtype).bits >> 3


class _ClassPropertyDescriptor:
    # [CLASS DOC] HELPER: Descriptor that exposes a class-level property: on attribute access it binds the stored getter to the owner class and calls it. Attribute: fget. A setter is not supported.
    def __init__(self, fget, fset=None):
        # [FUNCTION DOC] HELPER: Stores the getter; the setter argument is accepted but ignored.
        self.fget = fget

    def __get__(self, instance, owner=None):
        # [FUNCTION DOC] HELPER: Resolves the owner (type of the instance if not given), binds the getter and returns its call result.
        if owner is None:
            owner = type(instance)
        return self.fget.__get__(instance, owner)()


def classproperty(func):
    # [FUNCTION DOC] Decorator turning a function into a class-level read-only property; wraps it as a class method unless it already is a class or static method. Returns the descriptor.
    if not isinstance(func, (classmethod, staticmethod)):
        func = classmethod(func)
    return _ClassPropertyDescriptor(func)


if TYPE_CHECKING:
    # TorchScript does not support `@deprecated`
    # This is a workaround to avoid breaking TorchScript
    @deprecated(
        "`torch._utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
        category=FutureWarning,
    )
    def is_compiling() -> bool:
        # [FUNCTION DOC] Type-checking-only variant, deprecated: returns whether torch.compile or export tracing is active.
        return torch.compiler.is_compiling()

else:

    def is_compiling() -> bool:
        # [FUNCTION DOC] Deprecated: emits a warning pointing to the compiler namespace function, then returns whether torch.compile or export tracing is active (the warning is manual because TorchScript lacks the deprecation decorator).
        """
        Indicates whether we are tracing/compiling with torch.compile() or torch.export().
        """
        warnings.warn(  # use `warnings.warn` instead of `@deprecated`
            "`torch._utils.is_compiling` is deprecated. Use `torch.compiler.is_compiling` instead.",
            # FutureWarning,  # TorchScript does not support Warning type
            stacklevel=2,
        )
        return torch.compiler.is_compiling()


def _functionalize_sync(t):
    # [FUNCTION DOC] HELPER: Syncs a functionalized tensor. For a Python functional tensor, temporarily removes any active functional dispatch mode so it does not intercept view ops during the sync, syncs the inner tensor, and restores the mode in a finally block; otherwise syncs the tensor directly.
    # This code lives in python instead of C++ since conditioning on a certain python subclass
    # is much more of a pain in C++.
    from torch._subclasses.functional_tensor import FunctionalTensor

    if isinstance(t, FunctionalTensor):
        # If a FunctionalTensorMode is active while syncing, we don't want it to intercept any ops that get called
        # when we sync our inner tensor.
        # Why?
        # (1) If there are input mutations in the graph, then they will be re-applied during
        #     AOTAutograd when we call _sync() from inside of our functionalization kernels.
        # (2) _sync() causes us to regenerate our updated the tensor from the updated base,
        #     which dispatches to a bunch of view ops
        # (3) The input to these view ops is our inner FunctionalTensorWrapper
        #     (since the sync was called from C++), not the python FunctionalTensor
        # (4) if a python FunctionalTensorMode is active, it will complain when it intercepts
        #     the view op, since it will see an input that is a C++ FunctionalTensorWrapper
        #     (aka a normal torch.Tensor) instead of a python `FunctionalTensor).
        maybe_functional_mode = torch._C._unset_dispatch_mode(
            torch._C._TorchDispatchModeKey.FUNCTIONAL
        )
        try:
            torch._functionalize_sync(t.elem)  # type: ignore[attr-defined]
        finally:
            if maybe_functional_mode is not None:
                torch._C._set_dispatch_mode(maybe_functional_mode)
    else:
        torch._functionalize_sync(t)  # type: ignore[attr-defined]


@functools.lru_cache(2)
def _get_device_module(device_type: str):
    # [FUNCTION DOC] HELPER: Cached (size 2). Returns the torch submodule named after the device type; raises RuntimeError if there is none.
    device_module = getattr(torch, device_type, None)
    if device_module is None:
        raise RuntimeError(
            f"Device '{device_type}' does not have a corresponding module registered as 'torch.{device_type}'."
        )
    return device_module


def _dummy_type(name: str) -> type:
    # [FUNCTION DOC] HELPER: Creates a placeholder class of the given name whose construction always fails with RuntimeError (used when a backend type is unavailable).
    def get_err_fn(is_init: bool):
        # [FUNCTION DOC] HELPER: Returns an error-raising function configured for either the initializer or the allocator, differing in how the class name is obtained.
        def err_fn(obj, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Always raises RuntimeError saying the dummy base class cannot be instantiated.
            if is_init:
                class_name = obj.__class__.__name__
            else:
                class_name = obj.__name__
            raise RuntimeError(f"Tried to instantiate dummy base class {class_name}")

        return err_fn

    return type(
        name, (object,), {"__init__": get_err_fn(True), "__new__": get_err_fn(False)}
    )


class _LazySeedTracker:
    # [CLASS DOC] HELPER: Remembers the latest queued manual_seed_all and manual_seed callbacks (with tracebacks) and their relative order, since seeding is memoryless. Attributes: manual_seed_all_cb, manual_seed_cb, call_order.
    # Since seeding is memory-less, only track the latest seed.
    # Note: `manual_seed_all` followed by `manual_seed` overwrites
    # the seed on current device. We track the order of **latest**
    # calls between these two API.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Initializes both callbacks to None and an empty call order.
        self.manual_seed_all_cb = None
        self.manual_seed_cb = None
        self.call_order = []

    def queue_seed_all(self, cb, traceback):
        # [FUNCTION DOC] HELPER: Stores the latest all-devices seed callback and traceback and places it last in the call order.
        self.manual_seed_all_cb = (cb, traceback)  # pyrefly: ignore [bad-assignment]
        # update seed_all to be latest
        self.call_order = [self.manual_seed_cb, self.manual_seed_all_cb]

    def queue_seed(self, cb, traceback):
        # [FUNCTION DOC] HELPER: Stores the latest current-device seed callback and traceback and places it last in the call order.
        # KNOWN ISSUE: the call order list can contain None for a never-queued kind; callers must skip those.
        self.manual_seed_cb = (cb, traceback)  # pyrefly: ignore [bad-assignment]
        # update seed to be latest
        self.call_order = [self.manual_seed_all_cb, self.manual_seed_cb]

    def get_calls(self) -> list:
        # [FUNCTION DOC] HELPER: Returns the ordered list of pending seed calls.
        return self.call_order


logger = logging.getLogger(__name__)
P = ParamSpec("P")


class CallbackRegistry(Generic[P]):
    # [CLASS DOC] Simple named registry of callbacks sharing one signature. Attributes: name (used in log messages) and callback_list (in registration order).
    def __init__(self, name: str):
        # [FUNCTION DOC] Stores the name and creates an empty callback list.
        self.name = name
        self.callback_list: list[Callable[P, None]] = []

    def add_callback(self, cb: Callable[P, None]) -> None:
        # [FUNCTION DOC] Appends a callback to the registry.
        self.callback_list.append(cb)

    def fire_callbacks(self, *args: P.args, **kwargs: P.kwargs) -> None:
        # [FUNCTION DOC] Calls every registered callback in order with the given arguments; an exception in a callback is logged (not raised) and the remaining callbacks still run.
        for cb in self.callback_list:
            try:
                cb(*args, **kwargs)
            except Exception:
                logger.exception(
                    "Exception in callback for %s registered with gpu trace", self.name
                )


def try_import(module_name: str) -> ModuleType | None:
    # [FUNCTION DOC] Returns the module with the given name if already imported; otherwise, if it can be found, creates it from its spec, registers it in the module table, executes it and returns it; returns None if not found. Raises AssertionError if the spec has no loader. Side effect: registers the module before executing it.
    # Implementation based on
    # https://docs.python.org/3/library/importlib.html#checking-if-a-module-can-be-imported
    if (module := sys.modules.get(module_name, None)) is not None:
        return module

    if (spec := importlib.util.find_spec(module_name)) is not None:
        module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = module

        # https://docs.python.org/3/library/importlib.html#importlib.machinery.ModuleSpec.loader
        # "The finder should always set this attribute"
        if spec.loader is None:
            raise AssertionError("The loader attribute should always be set")
        spec.loader.exec_module(module)
        return module

    return None


# IMPORT_MAPPING and NAME_MAPPING are adapted from https://github.com/python/cpython/blob/main/Lib/_compat_pickle.py
# for use in the weights_only Unpickler.

IMPORT_MAPPING = {
    "__builtin__": "builtins",
    "copy_reg": "copyreg",
    "Queue": "queue",
    "repr": "reprlib",
    "_abcoll": "collections.abc",
    # Non-mutual mappings.
    "UserDict": "collections",
    "UserList": "collections",
    "UserString": "collections",
    "whichdb": "dbm",
    "StringIO": "io",
    "cStringIO": "io",
}


# This contains rename rules that are easy to handle.  We ignore the more
# complex stuff (e.g. mapping the names in the urllib and types modules).
# These rules should be run before import names are fixed.
NAME_MAPPING = {
    ("__builtin__", "xrange"): ("builtins", "range"),
    ("__builtin__", "reduce"): ("functools", "reduce"),
    ("__builtin__", "intern"): ("sys", "intern"),
    ("__builtin__", "unichr"): ("builtins", "chr"),
    ("__builtin__", "unicode"): ("builtins", "str"),
    ("__builtin__", "long"): ("builtins", "int"),
    ("itertools", "izip"): ("builtins", "zip"),
    ("itertools", "imap"): ("builtins", "map"),
    ("itertools", "ifilter"): ("builtins", "filter"),
    ("itertools", "ifilterfalse"): ("itertools", "filterfalse"),
    ("itertools", "izip_longest"): ("itertools", "zip_longest"),
    ("UserDict", "IterableUserDict"): ("collections", "UserDict"),
    ("UserList", "UserList"): ("collections", "UserList"),
    ("UserString", "UserString"): ("collections", "UserString"),
    # Non-mutual mappings.
    ("__builtin__", "basestring"): ("builtins", "str"),
    ("exceptions", "StandardError"): ("builtins", "Exception"),
    ("UserDict", "UserDict"): ("collections", "UserDict"),
}

# Protocol 2 pickle (torch.save's default) maps builtin exceptions to the
# Python 2 "exceptions" module via REVERSE_NAME_MAPPING; map them back so
# allowlisted exception types resolve under their builtins.* names.
NAME_MAPPING.update(
    {
        ("exceptions", name): ("builtins", name)
        for name in _compat_pickle.PYTHON2_EXCEPTIONS
    }
)


def _chunk_or_narrow_cat(
    tensor: "torch.Tensor",
    num_chunks: int,
    narrow_dim: int,
    cat_dim: int = 0,
) -> "torch.Tensor":
    # [FUNCTION DOC] HELPER: Splits the tensor into num_chunks along narrow_dim and concatenates them along cat_dim. When distributed is available, tracing is active and the tensor has free unbacked symbolic sizes, uses equal-size narrow slices (floor division) instead of chunk to stay symbolic safe; otherwise uses chunk. Returns the new tensor.
    """
    Splits tensor along narrow_dim into num_chunks and concatenates along cat_dim.
    Uses torch.chunk in eager mode, but torch.narrow under tracing to be unbacked-symint safe.
    """
    if torch.distributed.is_available():
        from torch.distributed._functional_collectives import _are_we_tracing
        from torch.fx.experimental.symbolic_shapes import has_free_unbacked_symbols

        # TODO(pianpwk): remove the unbacked symbols check and fix AsyncTP pattern matching
        # for test_micro_pipeline_tp.py.
        if _are_we_tracing() and has_free_unbacked_symbols(tensor):
            chunk_size = tensor.size(narrow_dim) // num_chunks
            chunks = [
                torch.narrow(tensor, narrow_dim, i * chunk_size, chunk_size)
                for i in range(num_chunks)
            ]
            return torch.cat(chunks, dim=cat_dim)
    return torch.cat(torch.chunk(tensor, num_chunks, dim=narrow_dim), dim=cat_dim)


def _maybe_view_chunk_cat(
    res: "torch.Tensor", group_size: int, gather_dim: int
) -> "torch.Tensor":
    # [FUNCTION DOC] HELPER: Equivalent to splitting the result of an all-gather (stacked on dim 0) into group_size chunks and concatenating them along gather_dim. Returns res unchanged when gather_dim is 0. If dim 0 size equals group_size and all dims between 0 and gather_dim have size 1 (decided with guard-or-false, so unknown symbolic sizes take the slow path), returns a reshaped view with dim 0 folded into gather_dim; otherwise falls back to the chunk-and-cat helper (a copy).
    """
    This is intuitively the same as torch.cat(torch.chunk(res, group_size,
    dim=0), dim=gather_dim), but returns a view if data movement is not
    necessary.  This operation arises in NCCL all_gather, where you always get
    a result which is concatenated on dim=0, even though actually you may need
    to undo this concatenation and then re-cat on the gather dim.

    When is data-movement not necessary?  Intuitively, we need to understand if
    the unflatten in this reference implementation of this code triggers a
    copy or not:

        chunks = torch.unflatten(res, 0, [group_size, -1])
        return torch.flatten(torch.movedim(chunks, 0, gather_dim), gather_dim, gather_dim + 1)

    Assume res is contiguous (it will be coming out of the collective).  We
    essentially need to know if the movedim maintains the contiguity of the
    tensor.  Moving a dimension typically does NOT preserve contiguity, unless
    EVERY dimension it is moved across is size 1.

    Example: shape [4, d1, d2] with group_size=4, gather_dim=1 -> [1, 4*d1, d2]

        [4, d1, d2] -> [4, 1, d1, d2] -> [1, 4, d1, d2] (contiguous!)

    Example: shape [4, 2, d2] with group_size=4, gather_dim=2 -> [1, 2, 4*d2]

        [4, 2, d2] -> [4, 1, 2, d2] -> [1, 2, 4, d2] (not contiguous!)

    Args:
        res: Tensor with gathered data in dim 0, shape [group_size, ...]
        group_size: Number of ranks in the group
        gather_dim: Dimension to gather along in the output

    Returns:
        Tensor with data rearranged to gather along gather_dim
    """

    if gather_dim == 0:
        # When gather_dim is 0, chunk+cat is a no-op
        return res

    shape = list(res.shape)

    # Optimization: Can use view instead of split+cat when:
    # 1. res.shape[0] == group_size (invariant after all_gather)
    # 2. All dims between 0 and gather_dim (exclusive) have size 1
    from torch.fx.experimental.symbolic_shapes import guard_or_false

    numel_between = math.prod(shape[1:gather_dim]) if gather_dim > 1 else 1

    if guard_or_false(shape[0] == group_size) and guard_or_false(numel_between == 1):
        # View optimization: reshape to collapse dim 0 into gather_dim
        final_shape = (
            [1]  # Dim 0 becomes 1
            + shape[1:gather_dim]  # Dims 1 to gather_dim-1 unchanged
            + [shape[0] * shape[gather_dim]]  # gather_dim gets multiplied by group_size
            + shape[gather_dim + 1 :]  # Rest unchanged
        )
        return res.view(final_shape)
    else:
        # General case: fall back to split + cat
        # This is better than torch.flatten as cat can be vectorized, whereas
        # the contiguous kernel is always bad.
        return _chunk_or_narrow_cat(res, group_size, narrow_dim=0, cat_dim=gather_dim)


class _Frame(TypedDict):
    # [CLASS DOC] HELPER: Typed dictionary for one stack frame in a memory snapshot: filename, line, name, plus optional FX node op, name, target and original trace added by augmentation.
    """Frame information from memory profiler snapshots."""

    filename: str
    line: int
    name: str
    # Fields added by FX augmentation (optional)
    fx_node_op: NotRequired[str]
    fx_node_name: NotRequired[str]
    fx_node_target: NotRequired[str]
    fx_original_trace: NotRequired[str]


class _Block(TypedDict):
    # [CLASS DOC] HELPER: Typed dictionary for one allocator block in a snapshot: size, requested size, address, state and its frames.
    """Memory block information."""

    size: int
    requested_size: int
    address: int
    state: str
    frames: list[_Frame]


class _Segment(TypedDict):
    # [CLASS DOC] HELPER: Typed dictionary for one allocator segment in a snapshot: address, sizes, stream, type and its blocks.
    """Memory segment information."""

    address: int
    total_size: int
    stream: int
    segment_type: str
    allocated_size: int
    active_size: int
    blocks: list[_Block]


class _TraceEntry(TypedDict):
    # [CLASS DOC] HELPER: Typed dictionary for one memory history trace event: action, optional address, frames, size, stream and optional free device memory.
    """Memory trace entry information."""

    action: str
    addr: NotRequired[int]
    frames: list[_Frame]
    size: int
    stream: int
    device_free: NotRequired[int]


class _Snapshot(TypedDict):
    # [CLASS DOC] HELPER: Typed dictionary for a whole memory snapshot: list of segments and optional per-device trace lists.
    """Memory snapshot structure."""

    segments: list[_Segment]
    device_traces: NotRequired[list[list[_TraceEntry]]]


def _augment_frames(frames: list[_Frame]) -> int:
    # [FUNCTION DOC] HELPER: For each frame whose filename looks like an FX generated module file and is found in the FX metadata registry, maps its line (minus the prologue offset) to a graph node and writes the node op, name, target (as text) and, when present, the original stack trace into the frame in place. Frames lacking a filename, line, or any matching metadata are skipped. Returns the number of augmented frames.
    """
    Augment a list of frames with FX debug information. For each frame corresponding
    to an FX-generated Python file, this function attaches additional FX node
    metadata (op, name, target, and original trace).

    Args:
        frames (list[_Frame]): List of frame dictionaries to augment

    Returns:
        int: The count of frames that were augmented.
    """
    from torch.fx.graph_module import FX_GRAPH_MODULE_FILE_PREFIX
    from torch.fx.traceback import _FX_METADATA_REGISTRY

    # Regex pattern to match FX generated files
    _FX_GENERATED_PATTERN = re.compile(
        rf"{re.escape(FX_GRAPH_MODULE_FILE_PREFIX)}.*\.py$"
    )

    count = 0

    for frame in frames:
        filename = frame.get("filename")
        lineno = frame.get("line")
        if not filename or not lineno:
            continue

        # Check if this looks like an FX generated file
        if not _FX_GENERATED_PATTERN.search(os.path.basename(filename)):
            continue

        metadata = _FX_METADATA_REGISTRY.get(filename)
        if metadata is None:
            continue

        lineno_map = metadata.get("lineno_map", {})
        node_metadata = metadata.get("node_metadata", {})
        prologue_start = metadata.get("prologue_start", 0)

        # Get the node index for this line
        node_idx = lineno_map.get(lineno - prologue_start)
        if node_idx is None:
            continue

        node_info = node_metadata.get(node_idx)
        if node_info is None:
            continue

        # Populate FX metadata fields
        frame["fx_node_op"] = node_info.get("op")
        frame["fx_node_name"] = node_info.get("name")
        frame["fx_node_target"] = str(node_info.get("target"))

        # Attach original stack trace if available
        original_trace = node_info.get("stack_trace")
        if original_trace:
            frame["fx_original_trace"] = original_trace

        count += 1

    return count


def _augment_memory_snapshot_stack_traces(
    snapshot: str | _Snapshot,
) -> _Snapshot:
    # [FUNCTION DOC] HELPER: Accepts a snapshot dictionary or a path to a pickled snapshot (loaded from disk; unpickling untrusted files is unsafe) and augments every frame in segment blocks and device traces in place using the in-process FX metadata registry, so it only works in the process that compiled the graphs. Returns the snapshot dictionary.
    # # KNOWN ISSUE: the docstring claims an fx_node_info field is added, but only op, name, target and original trace fields are written.
    """
    Augment a memory snapshot with original source stack traces from FX metadata.

    IMPORTANT: This function reads from a global in-memory registry (_FX_METADATA_REGISTRY)
    that is populated during graph module compilation. It must be called in the same
    Python process where the FX graphs were compiled. It cannot be used to augment
    snapshots loaded from disk in a different process.

    Args:
        snapshot (str or _Snapshot): Either a memory snapshot dict or path to a snapshot pickle file

    Returns:
        _Snapshot: The augmented snapshot dictionary with fx_node_op, fx_node_name,
            fx_original_trace, and fx_node_info fields added to frames
    """

    snapshot_dict: _Snapshot
    if isinstance(snapshot, str):
        # Load the memory snapshot
        with open(snapshot, "rb") as f:
            snapshot_dict = cast(_Snapshot, pickle.load(f))
    else:
        snapshot_dict = snapshot

    # Process blocks in segments (for regular allocations)
    for segment in snapshot_dict.get("segments", []):
        for block in segment.get("blocks", []):
            if "frames" in block:
                _augment_frames(block["frames"])

    # Process device traces (for memory history)
    for trace_list in snapshot_dict.get("device_traces", []):
        for trace_entry in trace_list:
            if isinstance(trace_entry, dict) and "frames" in trace_entry:
                _augment_frames(trace_entry["frames"])

    return snapshot_dict


def _is_privateuse1_backend_available():
    # [FUNCTION DOC] HELPER: Returns true when the registered private-use backend module exists and its availability check returns true; falsy otherwise (may return None or False).
    """
    Determines whether the privateuse1 backend is registered and available.

    Returns:
        Return True if the privateuse1 backend is registered and available.
    """
    privateuse1_backend_name = torch._C._get_privateuse1_backend_name()
    privateuse1_backend_module = getattr(torch, privateuse1_backend_name, None)
    return (
        is_available := getattr(privateuse1_backend_module, "is_available", None)
    ) and is_available()


def getenv(name: str) -> "str | None":
    # [FUNCTION DOC] Reads an environment variable through torch's C++ environment access (shared mutex with C++ code); returns its value or None if unset.
    """Read an environment variable through torch's serialized env access.

    Prefer this over :func:`os.getenv` when torch is loaded: it shares c10's
    environment mutex, so the read is consistent with concurrent C++ code and
    other :func:`torch._utils.setenv` calls that also go through c10.

    Args:
        name: Name of the environment variable.

    Returns:
        The variable's value, or ``None`` if it is not set.
    """
    return torch._C._getenv(name)


def setenv(name: str, value: str, overwrite: bool = True) -> None:
    # [FUNCTION DOC] Sets an environment variable through torch's C++ environment access; when overwrite is false an existing value is left unchanged.
    """Set an environment variable through torch's serialized env access.

    Prefer this over assigning to :data:`os.environ` when torch is loaded: it
    shares c10's environment mutex, so the write is consistent with concurrent
    C++ code and other :func:`torch._utils.getenv` calls that also go through
    c10.

    Args:
        name: Name of the environment variable.
        value: Value to set.
        overwrite: If ``False`` and the variable is already set, leave it
            unchanged. Defaults to ``True``.
    """
    torch._C._setenv(name, value, overwrite)


def unsetenv(name: str) -> None:
    # [FUNCTION DOC] Removes an environment variable through torch's C++ environment access.
    """Remove an environment variable through torch's serialized env access.

    The counterpart to :func:`setenv`; shares c10's environment mutex.

    Args:
        name: Name of the environment variable to remove.
    """
    torch._C._unsetenv(name)


# Saved originals so os.environ interception can be reverted; ``None`` means the
# hook is not currently installed.
_original_os_putenv: "Callable[..., None] | None" = None
_original_os_unsetenv: "Callable[..., None] | None" = None


def _torch_putenv(key: "str | bytes", value: "str | bytes") -> None:
    # [FUNCTION DOC] HELPER: Replacement for the operating system put-environment call: decodes key and value (filesystem decoding) and forwards to the torch setenv with overwrite.
    setenv(os.fsdecode(key), os.fsdecode(value))


def _torch_unsetenv(key: "str | bytes") -> None:
    # [FUNCTION DOC] HELPER: Replacement for the operating system unset-environment call: decodes the key and forwards to the torch unsetenv.
    unsetenv(os.fsdecode(key))


def install_os_environ_hook() -> None:
    # [FUNCTION DOC] Saves the original os putenv and unsetenv functions in module globals and replaces them with the torch-routed versions so that environment mutations go through the c10 mutex. Idempotent: does nothing if already installed. Global side effect on the os module.
    """Route ``os.environ`` mutations through torch's serialized env access.

    CPython's ``os.environ.__setitem__`` / ``__delitem__`` call the module-level
    ``os.putenv`` / ``os.unsetenv`` and then update ``os.environ``'s cached dict.
    This replaces those two functions so that mutating ``os.environ`` (and any
    direct ``os.putenv`` / ``os.unsetenv`` call) goes through c10's environment
    mutex, keeping Python and C++ env access consistent. The cached dict is still
    updated by CPython afterwards, so ``os.environ`` reads stay correct.

    Idempotent: calling it again while installed is a no-op. Use
    :func:`remove_os_environ_hook` to revert.
    """
    global _original_os_putenv, _original_os_unsetenv
    if _original_os_putenv is not None:
        return
    _original_os_putenv = os.putenv
    _original_os_unsetenv = os.unsetenv
    os.putenv = _torch_putenv  # type: ignore[assignment]
    os.unsetenv = _torch_unsetenv  # type: ignore[assignment]


def remove_os_environ_hook() -> None:
    # [FUNCTION DOC] Restores the saved original os putenv and unsetenv functions and clears the saved globals; does nothing if the hook is not installed.
    """Undo :func:`install_os_environ_hook`; a no-op if it was not installed."""
    global _original_os_putenv, _original_os_unsetenv
    if _original_os_putenv is None or _original_os_unsetenv is None:
        return
    os.putenv = _original_os_putenv
    os.unsetenv = _original_os_unsetenv
    _original_os_putenv = None
    _original_os_unsetenv = None
