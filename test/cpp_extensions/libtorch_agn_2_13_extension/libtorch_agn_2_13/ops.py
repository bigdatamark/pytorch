# [MODULE DOC] Python wrappers for the ops of the libtorch_agn_2_13 test extension. Defines the
# namespace constant, explicit wrappers for ops defined in this version (calling the default
# overload under torch.ops), and module-level attribute and directory hooks that proxy ops
# inherited from earlier version extensions. Requires the extension library to be loaded for ops
# to resolve; no import-time side effects.
import torch


# =============================================================================
# Proxy for inherited ops (from libtorch_agn_2_9, 2_10, 2_11, and 2_12 csrc/)
#
# Ops compiled from previous versions' csrc directories are accessible via
# the module-level __getattr__. For example:
#     libtorch_agn_2_13.ops.sgd_out_of_place(...)  # from 2.9
#     libtorch_agn_2_13.ops.my_sum(...)            # from 2.10
#
# Ops defined in this package's csrc/ (e.g. test_stream_native_handle) are
# registered on ``torch.ops.libtorch_agn_2_13`` and wrapped explicitly below.
# =============================================================================

_NAMESPACE = "libtorch_agn_2_13"


def test_stream_native_handle(device_index) -> int:
    # [FUNCTION DOC] Calls the stable op test_stream_native_handle with a device index and
    # returns the integer native stream handle (for comparison with torch.accelerator).
    """Stable ``Stream::nativeHandle`` vs ``torch.accelerator`` (2.13+)."""
    return torch.ops.libtorch_agn_2_13.test_stream_native_handle.default(device_index)


def test_kernel_launch_on_stream(input, fill_value: int):
    # [FUNCTION DOC] Calls the op that launches a CUDA kernel on the stream obtained from its
    # native handle, filling the input tensor with fill_value; returns the op result tensor.
    """Launch CUDA kernel using stream from nativeHandle()."""
    return torch.ops.libtorch_agn_2_13.test_kernel_launch_on_stream.default(
        input, fill_value
    )


def my_rand_with_generator(size, generator):
    # [FUNCTION DOC] Calls the op that generates a random tensor of the given size via the
    # dispatcher while passing a Generator; returns the tensor.
    """rand via torch_call_dispatcher threading a Generator (2.13+)."""
    return torch.ops.libtorch_agn_2_13.my_rand_with_generator.default(size, generator)


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
