# [MODULE DOC] Operator profiles for generating fake kernels: records of the input and
# output tensor metadata (rank, dtype, device, layout) for custom ops. Provides the
# TensorMetadata and OpProfile data classes, a context manager that temporarily registers
# fake kernels derived from profiles, and yaml save/load helpers. Import-time side effect:
# creates a logger.
import contextlib
import io
import logging
import os
from collections.abc import Callable, Generator
from dataclasses import dataclass
from typing import Any, Optional

import torch
from torch._library.custom_ops import _maybe_get_opdef
from torch.types import FileLike


log = logging.getLogger(__name__)


class MissingOpProfile(RuntimeError):
    # [CLASS DOC] Error raised by a generated fake kernel when no recorded profile matches
    # the given inputs.
    """
    This is raised when we don't have an operator profile available for the
    given inputs.
    """


@dataclass(frozen=True)
class TensorMetadata:
    # [CLASS DOC] Immutable (hashable) summary of a tensor: rank, dtype, device, layout.
    # Used to match call inputs against recorded profiles and to build fake outputs.
    rank: int
    dtype: torch.dtype
    device: torch.device
    layout: torch.layout

    @staticmethod
    def maybe_from_tensor(t: Any) -> Optional["TensorMetadata"]:
        # [FUNCTION DOC] Static method: returns a TensorMetadata for a Tensor argument, or
        # None if t is not a Tensor.
        if not isinstance(t, torch.Tensor):
            return None
        return TensorMetadata(t.dim(), t.dtype, t.device, t.layout)


@dataclass(frozen=True)
class OpProfile:
    # [CLASS DOC] Immutable (hashable) record of one observed call of an operator:
    # args_profile holds per-argument TensorMetadata (None for non-tensor args) and
    # out_profile is a single TensorMetadata or a tuple of them for multiple outputs.
    args_profile: tuple[TensorMetadata | None]
    out_profile: TensorMetadata | tuple[TensorMetadata]


def _generate_fake_kernel(op_name: str, op_profile: set[OpProfile]) -> Callable:
    # [FUNCTION DOC] HELPER: Builds a fake kernel for op_name from a set of profiles. The
    # kernel scans the profiles for the first whose argument metadata matches the call
    # (positional args followed by keyword values) and returns fake outputs with the
    # recorded metadata; raises MissingOpProfile if none matches.
    def _match_args(args_profile: tuple[TensorMetadata | None], args: Any) -> bool:
        # [FUNCTION DOC] HELPER: True if every given arg's metadata (None for non-tensors)
        # equals the profile entry at the same position.
        # KNOWN ISSUE: indexing the profile by position raises an index error if more args
        # are passed than the profile has entries.
        return all(
            TensorMetadata.maybe_from_tensor(arg) == args_profile[i]
            for i, arg in enumerate(args)
        )

    def _generate_res(
        out_profile: TensorMetadata | tuple[TensorMetadata],
    ) -> torch.Tensor | list[torch.Tensor]:
        # [FUNCTION DOC] HELPER: Creates a fake output tensor (or a list of them for a tuple
        # profile) using the active fake-impl context to make data-dependent sizes.
        ctx = torch.library.get_ctx()

        def _generate_tensor_out(t: TensorMetadata) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Builds an empty strided tensor of the profiled rank,
            # dtype, device and layout, with each dimension a new dynamic size. Strides are
            # cumulative products of the sizes starting from the first dimension (a
            # column-major style layout).
            fake_shape = [ctx.new_dynamic_size() for _ in range(t.rank)]
            fake_strides = [-1] * t.rank
            expected = 1
            fake_stride = expected
            # pyrefly: ignore [bad-assignment]
            for i in range(t.rank):
                fake_strides[i] = fake_stride  # type: ignore[assignment]
                fake_stride = fake_stride * fake_shape[i]  # type: ignore[assignment]

            return torch.empty_strided(
                fake_shape,
                fake_strides,
                device=t.device,
                dtype=t.dtype,
                layout=t.layout,
            )

        if isinstance(out_profile, TensorMetadata):
            return _generate_tensor_out(out_profile)
        else:
            return [_generate_tensor_out(t) for t in out_profile]

    def _fake_kernel(*args, **kwargs):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: The generated fake kernel; returns outputs for the first
        # matching profile or raises MissingOpProfile.
        for profile in op_profile:
            if _match_args(profile.args_profile, (*args, *kwargs.values())):
                return _generate_res(profile.out_profile)

        raise MissingOpProfile(
            f"No fake kernel was found for {op_name}, and although we have "
            "previously registered some profiles to generate a fake kernel, "
            f"no profiles match the given inputs: {args, kwargs}."
        )

    return _fake_kernel


@contextlib.contextmanager
def unsafe_generate_fake_kernels(op_profiles: dict[str, set[OpProfile]]) -> Generator:
    # [FUNCTION DOC] Context manager that registers profile-derived fake kernels for each
    # operator (names of the form namespace.opname) in op_profiles, overriding existing
    # fakes, and logs a warning per op. CustomOpDef ops get register_fake called (their old
    # abstract fn is saved); others get a temporary FRAGMENT library. Yields the list of
    # temporary libraries. On exit destroys those libraries and restores saved abstract fns,
    # raising AssertionError if an opdef vanished. Unsafe: assumes fixed output ranks.
    """
    Registers a fake kernel based on the given operator profiles. This fake
    kernel registration will override any existing fake kernel registrations.

    The input is a dictionary mapping operator names to a set of operator
    profiles, which we will use to generate fake kernels. The operator profiles
    are a record of the input and output tensor metadata. Based on this
    information we will match a given input to the recorded profile, and return
    an output with the same metadata as in the recorded profile. If a profile
    doesn't exist then an exception will be thrown.

    The fake kernel generation is considered unsafe because it relies on the
    rigid, pre-defined operator profiles that do not account for potential
    variations in output behavior. Specifically, the generated kernels assume a
    fixed relationship between input and output ranks. However, in reality, it's
    possible that data-dependent operations may produce outputs of different
    ranks even when given inputs of the same rank. The generated fake kernels
    are inflexible and unable to accommodate these nuances, making them
    potentially unsafe.

    Args:
        op_profiles (dict[str, set[OpProfile]]): A dictionary mapping operator
            name to a set of operator profiles from which we will generate fake
            kernels.

    Examples:

        >>> # Example: Registering an op-profile from draft-export
        >>> import torch
        >>> from torch.export._draft_export import draft_export
        >>>
        >>> @torch.library.custom_op("mylib::foo", mutates_args=())
        >>> def foo(x: Tensor, y: Tensor) -> Tensor:
        >>>     return x + y
        >>>
        >>> class M(torch.nn.Module):
        >>>     def forward(self, a, b):
        >>>         res = torch.ops.mylib.foo(a, b)  # no fake impl
        >>>         return res
        >>>
        >>> ep = draft_export(M(), (torch.ones(3, 4), torch.ones(3, 4))
        >>>
        >>> with torch._library.fake_profile.unsafe_generate_fake_kernels(ep._report.op_profiles):
        >>>     decomp = ep.run_decompositions()

    """

    libs: list[torch.library.Library] = []
    # Stores old fake impls from custom ops declared through @custom_op
    old_fake_impls: dict[str, Callable] = {}
    for op_name, profiles in op_profiles.items():
        log.warning(
            "Registering fake profile for %s. This will override any existing "
            "fake kernel registration.",
            op_name,
        )

        op_name_split = op_name.split(".")
        namespace, op_name_str = op_name_split[0], op_name_split[1]
        op_str = f"{namespace}::{op_name_str}"

        fake_kernel = _generate_fake_kernel(op_str, profiles)

        if opdef := _maybe_get_opdef(op_str):
            # If the op is a CustomOpDef, save the existing abstract_fn so that
            # we can restore it after this contextmanager
            if opdef._abstract_fn is not None:
                old_fake_impls[op_str] = opdef._abstract_fn
            opdef.register_fake(fake_kernel)

        else:
            # Create a new library so that we can register a new fake impl.
            # These libraries will then be destroyed after the contextmanager,
            # which will automatically restore the previously registered fake
            # impls.
            newlib = torch.library.Library(namespace, "FRAGMENT")
            torch.library.register_fake(
                op_str, fake_kernel, lib=newlib, allow_override=True
            )
            libs.append(newlib)

    try:
        yield libs
    finally:
        # Destroying the libraries will automatically restore the previously
        # registered fake impls
        for lib in libs:
            lib._destroy()

        # Restore abstract_fns for CustomOpDefs
        for op_str, old_fake in old_fake_impls.items():
            opdef = _maybe_get_opdef(op_str)
            if opdef is None:
                raise AssertionError(f"opdef for {op_str} must not be None")
            opdef.register_fake(old_fake)


def get_torch_version() -> str:
    # [FUNCTION DOC] Returns the current torch major.minor version as a string.
    version = torch.__version__.split(".")
    return f"{int(version[0])}.{int(version[1])}"


def generate_yaml_from_profiles(op_profiles: dict[str, set[OpProfile]]) -> str:
    # [FUNCTION DOC] Serializes profiles to a yaml string with the torch version and a map
    # of operator name to a list of profiles (dtype and layout as serialized enum values,
    # device as a string). Imports yaml and export serde tables lazily.
    # KNOWN ISSUE: None entries in args_profile are dropped when serializing, so positions
    # of non-tensor arguments are lost and a reloaded profile may not align with call args.
    """
    Generates a yaml string from the given operator profiles which can be saved
    to a file. The yaml string can be loaded back into an operator profile
    structure using `read_profiles_from_yaml`.
    """

    import yaml

    from torch._export.serde.serialize import (
        _TORCH_TO_SERIALIZE_DTYPE,
        _TORCH_TO_SERIALIZE_LAYOUT,
    )

    def serialize_tensor_metadata(t: TensorMetadata) -> dict:
        # [FUNCTION DOC] HELPER: Converts a TensorMetadata to a plain mapping of rank,
        # serialized dtype value, device string and serialized layout value.
        return {
            "rank": t.rank,
            "dtype": _TORCH_TO_SERIALIZE_DTYPE[t.dtype].value,
            "device": str(t.device),
            "layout": _TORCH_TO_SERIALIZE_LAYOUT[t.layout].value,
        }

    def serialize_op_profile(op: OpProfile) -> dict:
        # [FUNCTION DOC] HELPER: Converts an OpProfile to a mapping with a list of serialized
        # args (skipping None) and a serialized output or list of outputs.
        return {
            "args_profile": [
                serialize_tensor_metadata(arg)
                for arg in op.args_profile
                if arg is not None
            ],
            "out_profile": (
                serialize_tensor_metadata(op.out_profile)
                if isinstance(op.out_profile, TensorMetadata)
                else [serialize_tensor_metadata(out) for out in op.out_profile]
            ),
        }

    serialized_data = {
        operator: [serialize_op_profile(profile) for profile in profiles]
        for operator, profiles in op_profiles.items()
    }
    return yaml.dump(
        {"torch_version": get_torch_version(), "operators": serialized_data},
        sort_keys=False,
    )


def save_op_profiles(op_profiles: dict[str, set[OpProfile]], f: FileLike) -> None:
    # [FUNCTION DOC] Writes the yaml form of op_profiles to f, which may be a path (file is
    # written as text) or a BytesIO (utf-8 bytes written). Raises ValueError for any other
    # type of f. Side effect: file I/O.
    """
    Serializes the given operator profiles into a yaml format and saves it to
    the given file. The operator profile can be loaded back using `load_op_profiles`.
    """
    yaml_str = generate_yaml_from_profiles(op_profiles)

    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

        with open(f, "w") as file:
            file.write(yaml_str)

    elif isinstance(f, io.BytesIO):
        f.write(yaml_str.encode("utf-8"))

    else:
        raise ValueError(f"Invalid type of file {f}")


def read_profiles_from_yaml(yaml_str: str) -> dict[str, set[OpProfile]]:
    # [FUNCTION DOC] Parses yaml produced by save_op_profiles into a mapping of operator
    # name to a set of OpProfile. Raises RuntimeError if the saved torch version differs
    # from the current major.minor version. Lists of outputs become tuples.
    """
    Reads the yaml saved by `save_op_profiles` and returns the operator profiles.
    """

    import yaml

    from torch._export.serde.serialize import (
        _SERIALIZE_TO_TORCH_DTYPE,
        _SERIALIZE_TO_TORCH_LAYOUT,
    )

    def deserialize_tensor_metadata(data: dict) -> TensorMetadata:
        # [FUNCTION DOC] HELPER: Rebuilds a TensorMetadata from its serialized mapping.
        return TensorMetadata(
            rank=data["rank"],
            dtype=_SERIALIZE_TO_TORCH_DTYPE[data["dtype"]],
            device=torch.device(data["device"]),
            layout=_SERIALIZE_TO_TORCH_LAYOUT[data["layout"]],
        )

    def deserialize_op_profile(data: dict) -> OpProfile:
        # [FUNCTION DOC] HELPER: Rebuilds an OpProfile; a list output becomes a tuple of
        # TensorMetadata, otherwise a single TensorMetadata.
        args_profile = tuple(
            deserialize_tensor_metadata(arg) for arg in data["args_profile"]
        )
        out_profile_data = data["out_profile"]
        out_profile: tuple[TensorMetadata] | TensorMetadata = (
            tuple(deserialize_tensor_metadata(out) for out in out_profile_data)  # type: ignore[assignment]
            if isinstance(out_profile_data, list)
            else deserialize_tensor_metadata(out_profile_data)
        )
        return OpProfile(args_profile=args_profile, out_profile=out_profile)  # type: ignore[arg-type]

    loaded_data = yaml.safe_load(yaml_str)
    loaded_torch_version = loaded_data["torch_version"]

    if loaded_torch_version != get_torch_version():
        raise RuntimeError(
            "Unable to load outdated profile. It was saved with torch version: "
            f"{loaded_torch_version} but the current torch version is: {get_torch_version()}"
        )

    operators_data = loaded_data["operators"]
    return {
        operator: {deserialize_op_profile(profile) for profile in profiles}
        for operator, profiles in operators_data.items()
    }


def load_op_profiles(f: FileLike) -> dict[str, set[OpProfile]]:
    # [FUNCTION DOC] Reads yaml text from a path or BytesIO f and returns the parsed
    # profiles via read_profiles_from_yaml. Raises ValueError for other types of f, and
    # RuntimeError on a torch version mismatch. Side effect: file I/O.
    """
    Loads the saved operator profiles from `save_op_profiles`.
    """
    if isinstance(f, (str, os.PathLike)):
        f = os.fspath(f)

        with open(f) as file:
            yaml_str = file.read()

    elif isinstance(f, io.BytesIO):
        yaml_str = f.read().decode("utf-8")

    else:
        raise ValueError(f"Invalid type of file {f}")

    return read_profiles_from_yaml(yaml_str)
