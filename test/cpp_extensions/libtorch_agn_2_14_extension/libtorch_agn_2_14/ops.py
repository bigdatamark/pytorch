# [MODULE DOC] Python wrappers for the ops of the libtorch_agn_2_14 test extension. Defines the
# namespace constant, explicit wrappers for ops defined in this version (calling the default
# overload under torch.ops), and module-level attribute and directory hooks that proxy ops
# inherited from earlier version extensions. Requires the extension library to be loaded for ops
# to resolve; no import-time side effects.
import torch


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9, 2_10, 2_11, 2_12, and 2_13
# csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_14.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_14.ops.my_sum(...)            # from 2.10
#
# Ops defined in this package's csrc/ (e.g. my_has_storage) are registered on
# ``torch.ops.libtorch_agn_2_14`` and wrapped explicitly below.
# =============================================================================

_NAMESPACE = "libtorch_agn_2_14"


def my_has_storage(t) -> bool:
    # [FUNCTION DOC] Calls the op that reports whether the tensor has storage allocated; returns
    # a bool.
    """Stable ``Tensor::has_storage()`` (2.14+)."""
    return torch.ops.libtorch_agn_2_14.my_has_storage.default(t)


def my_mps_scale_negate_clamp(t, scale, negate, low, high) -> torch.Tensor:
    # [FUNCTION DOC] MPS only: runs a Metal kernel that multiplies a float32 tensor by scale,
    # negates when requested, and clamps to the low/high bounds, passing the scalars by
    # torch_mps_set_arg_bytes. Returns a new tensor; the C++ side rejects non-float32 or empty
    # input.
    """Stable ``torch_mps_set_arg_bytes`` (2.14+, MPS only)."""
    return torch.ops.libtorch_agn_2_14.my_mps_scale_negate_clamp.default(
        t, scale, negate, low, high
    )


def my_mps_set_arg_bytes_raw(t, size, null_ptr) -> torch.Tensor:
    # [FUNCTION DOC] MPS only: binds size bytes from a static buffer (or a null pointer when
    # null_ptr is set) to a kernel argument via torch_mps_set_arg_bytes to test validation;
    # returns an uninitialized tensor shaped like t.
    return torch.ops.libtorch_agn_2_14.my_mps_set_arg_bytes_raw.default(
        t, size, null_ptr
    )


def my_mps_set_arg_bytes_lifetime(t) -> torch.Tensor:
    # [FUNCTION DOC] MPS only: sets scalar kernel args from local variables, then mutates those
    # variables before dispatch, checking the bytes were copied at set time; returns the output
    # tensor (the kernel should have used scale 3, no negate, effectively unbounded clamp).
    return torch.ops.libtorch_agn_2_14.my_mps_set_arg_bytes_lifetime.default(t)


def __getattr__(name):
    # [FUNCTION DOC] Module-level attribute hook: for a public name (not starting with an
    # underscore) looks the op up in the torch.ops namespace of this version and returns its
    # default overload, so ops compiled from earlier versions are reachable here. Raises
    # AttributeError for underscore names and for names missing from the namespace.
    """Proxy for inherited ops from previous versions."""
    if name.startswith("_"):
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    ops_namespace = getattr(torch.ops, _NAMESPACE)
    op = getattr(ops_namespace, name, None)
    if op is None:
        raise AttributeError(f"No op named '{name}' in {_NAMESPACE}")
    return op.default


def __dir__():
    # [FUNCTION DOC] Module-level directory hook: returns a sorted, de-duplicated list of the
    # public callable names defined in this module plus all public names in the torch.ops
    # namespace of this version (native plus inherited ops).
    """List all available ops (native + inherited)."""
    native = [
        name
        for name in globals()
        if not name.startswith("_") and callable(globals().get(name))
    ]
    ops_namespace = getattr(torch.ops, _NAMESPACE)
    inherited = [n for n in dir(ops_namespace) if not n.startswith("_")]
    return sorted(set(native + inherited))
