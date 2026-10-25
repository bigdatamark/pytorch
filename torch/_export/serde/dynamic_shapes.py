# [MODULE DOC] Serialization and deserialization of the dynamic_shapes spec given to export. Defines two
# dataclasses (RootDim for a root dimension's range and derived names, DynamicShapesSpec for the whole
# spec) and the functions that dump a spec (with the example args) to a dataclass or plain mapping and
# load it back into a structure of Dim objects, hints, ints and None. No import-time side effects.
import dataclasses
from typing import Any

import torch
from torch._dynamo.exc import UserError, UserErrorType
from torch.export.dynamic_shapes import (
    _check_dynamic_shapes,
    _DerivedDim,
    _DimHint,
    _tree_map_with_path,
    Dim,
)
from torch.utils._pytree import tree_map

from .serialize import _dataclass_to_dict


@dataclasses.dataclass
class RootDim:
    # [CLASS DOC] Serializable description of one root Dim: its minimum, its maximum (None meaning unbounded) and the
    # sorted names of dims derived from it by linear expressions. Plain data holder, no methods.
    """
    This represents a Dim object.
    """

    min: int
    max: int | None
    derived: list[str]


@dataclasses.dataclass
class DynamicShapesSpec:
    # [CLASS DOC] Serializable container pairing a standardized dynamic_shapes tree (leaves are ints, strings naming
    # dims or hints, or None) with a mapping from root dim name to its RootDim description. Plain data holder.
    """
    This stores a dynamic_shapes spec for de/serialization.
    """

    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None
    dims: dict[str, RootDim]


def _postprocess_serialized_shapes(
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    dims: dict[str, dict[str, int | list[str] | None]],
    to_dict: bool | None = False,
) -> DynamicShapesSpec | dict[str, Any]:
    # [FUNCTION DOC] HELPER: Builds a DynamicShapesSpec from the serialized shapes tree and the collected dim info. Dims are
    # sorted by name and converted to RootDim records (an infinite max becomes None, derived names are
    # sorted). Returns the dataclass, or its plain-mapping form when to_dict is true. Pure.
    """
    Sorts dims and dumps to dictionary format.
    """
    from torch.utils._sympy.numbers import int_oo

    # pyrefly: ignore [bad-assignment]
    dims = {
        k: RootDim(
            min=v["min"],  # type: ignore[arg-type]
            max=None if v["max"] is int_oo else v["max"],  # type: ignore[arg-type]
            derived=sorted(v["derived"]),  # type: ignore[arg-type]
        )
        for k, v in sorted(dims.items())
    }
    # pyrefly: ignore [bad-argument-type]
    spec = DynamicShapesSpec(dynamic_shapes=dynamic_shapes, dims=dims)
    if to_dict:
        return _dataclass_to_dict(spec)
    else:
        return spec


def _dump_dynamic_shapes(
    dynamic_shapes: dict[str, Any] | tuple[Any] | list[Any] | None,
    args: tuple[Any],
    kwargs: dict[str, Any] | None = None,
    to_dict: bool | None = False,
) -> DynamicShapesSpec | dict[str, Any]:
    # [FUNCTION DOC] Serializes a dynamic_shapes spec as used by export, using the example args and kwargs to tell
    # tensors from other inputs. Returns a spec (or plain mapping if to_dict) holding a standardized
    # tree (one list of per-dimension entries per tensor, None for non-tensors) and a mapping of root
    # dim names to range and derived names. If dynamic_shapes is None returns a mapping with None shapes
    # and no dims (a plain mapping even when to_dict is false). Runs the same validity check as export and may
    # raise its errors. Dim hints are stored as strings of class and kind; dims by name.
    """
    Utility function for dynamic shapes serialization, serializing a dynamic_shapes spec.
    Returns a DynamicShapesSpec dataclass containing 2 fields, "dynamic_shapes" and "dims".
    Uses args & kwargs to distinguish between tensor-level and dim-level specs (only for Nones).

    dynamic_shapes: A pytree structure mirroring the dynamic_shapes input to export():
        - Each tensor input is represented with a list of values, non-tensor inputs with None.
        - dynamic dimensions (i.e. symbols) in tensors and Dim enums are represented with strings.
        - static dimensions are represented with ints.

    dims: A dictionary mapping each symbol name to the min/max range and derived dim names.

    For example:
    ```
    dx = Dim("dx", min=4, max=16)
    dy = dx + 1

    inputs = (
        [
            torch.randn(4, 4),
            torch.randn(5, 4),
        ],
        torch.randn(4),
        torch.randn(4, 4),
        "hello",
    )
    dynamic_shapes = {
        "a": [
            (dx, 4),
            (dy, 4),
        ],
        "b": (Dim.STATIC,),
        "c": None,
        "d": None,
    }
    out = _dump_dynamic_shapes(dynamic_shapes, inputs, to_dict=True)
    ```
    would generate the following output:
    ```
    {
        "dynamic_shapes": (
            [
                ["dx", 4],
                ["dx + 1", 4],
            ],
            ["_DimHint.STATIC"],
            ["_DimHint.STATIC", "_DimHint.STATIC"],
            None,
        ),
        "dims": {
            "dx": {
                "min": 4,
                "max": 16,
                "derived": ["dx + 1"],
            },
        },
    }
    ```
    """
    dims: dict[str, dict[str, Any]] = {}

    def _standardize_shapes(path, tensor, shape):  # type: ignore[no-untyped-def]
        # [FUNCTION DOC] HELPER: Per-leaf mapper. Returns None for non-tensor inputs; a list of static hints (one per
        # dimension) when the tensor has no spec; otherwise a list with one entry per dimension, taking the
        # spec entry (from a mapping by index, or a sequence by position) or the concrete size when the
        # entry is missing or None. Raises an assertion error if the spec is neither mapping nor sequence.
        """
        Helps standardize the dynamic_shapes tree structure we serialize,
        returning lists for each tensor shape, handling tensor-level Nones.
        """
        if not isinstance(tensor, torch.Tensor):
            return None
        if shape is None:
            return [Dim.STATIC] * len(tensor.shape)

        out = []
        if isinstance(shape, dict):
            for i, s in enumerate(tensor.shape):
                out.append(s if shape.get(i) is None else shape.get(i))
        else:
            if not isinstance(shape, (tuple, list)):
                raise AssertionError(f"expected tuple or list, got {type(shape)}")
            for i, s in enumerate(tensor.shape):
                out.append(s if shape[i] is None else shape[i])
        return out

    def _track_dim_from_dims(
        val: None | int | _DimHint | Dim,
    ) -> None | int | str:
        # [FUNCTION DOC] HELPER: Converts one standardized leaf to its serialized form and records dims. None and ints
        # pass through; hints become a string of class name and kind name; Dim objects become their name,
        # and the root dim (with min, max and an empty derived set) is registered in the enclosing dims
        # mapping, with derived dims added to their root's derived set. Raises an assertion error for
        # other types. Mutates the enclosing dims mapping.
        """
        Tracks dims, ranges, derived dims from the standardized dynamic_shapes spec.
        """
        if val is None or isinstance(val, int):  # non-tensor input or static
            return val
        if isinstance(val, _DimHint):  # store enum as string
            return val.__class__.__name__ + "." + val.type.name

        if not isinstance(val, Dim):
            raise AssertionError(f"expected Dim, got {type(val)}")

        # track root dim
        root = val.root if isinstance(val, _DerivedDim) else val  # type: ignore[attr-defined]
        if root.__name__ not in dims:
            dims[root.__name__] = {
                "min": root.min,  # type: ignore[attr-defined,union-attr]
                "max": root.max,  # type: ignore[attr-defined,union-attr]
                "derived": set(),
            }

        # track derived dims
        if isinstance(val, _DerivedDim):
            dims[root.__name__]["derived"].add(val.__name__)

        return val.__name__

    if dynamic_shapes is None:
        return {"dynamic_shapes": None, "dims": {}}

    # convert to tuple of specs, for each arg/kwarg
    kwargs = kwargs or {}
    if isinstance(dynamic_shapes, dict):
        dynamic_shapes = dynamic_shapes.values()  # type: ignore[assignment]
    # pyrefly: ignore [bad-assignment, bad-argument-type]
    dynamic_shapes = tuple(dynamic_shapes)
    combined_args = tuple(args) + tuple(kwargs.values())

    # run same check when we're processing shapes for export - is this too lazy?
    _check_dynamic_shapes(dict(enumerate(combined_args)), dynamic_shapes)  # type: ignore[arg-type]

    tree_shapes = _tree_map_with_path(
        _standardize_shapes, combined_args, dynamic_shapes, tree_name="inputs"
    )
    serialized_shapes = tree_map(_track_dim_from_dims, tree_shapes)
    return _postprocess_serialized_shapes(serialized_shapes, dims, to_dict=to_dict)


def _load_dynamic_shapes(
    spec: DynamicShapesSpec | dict[str, Any],
    from_dict: bool | None = False,
) -> dict[str, Any] | tuple[Any] | list[Any] | None:
    # [FUNCTION DOC] Inverse of the dump function. Accepts a spec dataclass, or a mapping when from_dict is true
    # (validated: exact keys, string dim names, int min, list of string derived names), and returns the
    # dynamic_shapes tree for export, or None if the spec's shapes are None. Rebuilds root Dim objects with
    # their ranges, then each derived name by parsing it as a linear expression in its root symbol
    # (rejecting non-symbol names, multi-symbol or non-linear expressions) and applying the multiplier and offset.
    # Leaves are mapped back to ints, None, dim hints, or the cached Dim objects. Raises UserError for any
    # invalid input or untracked dim name.
    # KNOWN ISSUE: the mapping validation requires max to be an int and its error text claims None is
    # allowed; a None max (unbounded, as produced by dumping) is rejected, so unbounded dims do not round trip via mappings.
    # Also the condition checking for None after the int check is unreachable.
    """
    Utility function for dynamic shapes serialization.
    Deserializes a DynamicShapesSpec or corresponding dictionary into a dynamic_shapes input to export().
    """
    import sympy

    from torch.fx.experimental.symbolic_shapes import _is_supported_equivalence

    if from_dict:
        if not isinstance(spec, dict):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"With from_dict=True, expected `spec` to be a dict, got {type(spec)}",
            )
        if sorted(spec.keys()) != ["dims", "dynamic_shapes"]:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "With from_dict=True, expected `spec` to have keys `dims` and `dynamic_shapes`, "
                f"instead found {spec.keys()}",
            )
        dims = {}
        for k, v in spec["dims"].items():
            if not isinstance(k, str):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` keys to be strings for symbols, got key {type(k)}",
                )
            if sorted(v.keys()) != ["derived", "max", "min"]:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected `spec['dims']` values to have keys `derived`, `max`, and `min`, "
                    f"instead found {v.keys()}",
                )
            if not isinstance(v["min"], int):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `min` to an int, got {k}: {v['min']}",
                )
            if not isinstance(v["max"], int) or v["max"] is None:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected dims in `spec['dims']` to map `max` to an int or None, got {k}: {v['max']}",
                )
            if not isinstance(v["derived"], list) or any(
                not isinstance(d, str) for d in v["derived"]
            ):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    "Expected dims in `spec['dims']` to map `derived` to a list of derived expressions, "
                    f"got {k}: {v['derived']}",
                )
            dims[k] = RootDim(**v)
        dynamic_shapes = spec["dynamic_shapes"]
    else:
        if not isinstance(spec, DynamicShapesSpec):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec` to be a DynamicShapesSpec, got {type(spec)}",
            )
        dims = spec.dims
        dynamic_shapes = spec.dynamic_shapes

    if dynamic_shapes is None:
        return None

    dim_cache = {}
    for name, info in dims.items():
        symbol = sympy.sympify(name)
        if not isinstance(symbol, sympy.Symbol):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                f"Expected `spec['dims']` keys to be symbols, got {name}",
            )
        dim_cache[name] = Dim(name, min=info.min, max=info.max)  # cache root dim
        for _expr in info.derived:
            expr = sympy.sympify(_expr)
            if len(expr.free_symbols) != 1 or symbol not in expr.free_symbols:
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions in to have {name} as the only free symbol, got {expr}",
                )
            if not _is_supported_equivalence(expr):
                raise UserError(
                    UserErrorType.INVALID_INPUT,
                    f"Expected derived expressions to be linear expressions, got {expr}",
                )
            modulus, remainder = sympy.polys.polytools.div(expr, symbol)
            ddim = dim_cache[name]
            if modulus != 1:
                ddim = int(modulus) * ddim  # type: ignore[assignment, operator]
            if remainder != 0:
                ddim = ddim + int(remainder)  # type: ignore[assignment, operator]
            dim_cache[_expr] = ddim  # cache derived dims

    def deserialize_shape(
        val: None | int | str,
    ) -> None | int | Dim | _DimHint:
        # [FUNCTION DOC] HELPER: Maps one serialized leaf back: None and ints unchanged; the strings for the auto, dynamic and
        # static hints to new hint objects; any other string to the cached Dim or derived dim of that name.
        # Raises UserError for non-string leaves or names not tracked in the dims mapping.
        if val is None or isinstance(val, int):
            return val
        elif val == "_DimHint.AUTO":
            return _DimHint.AUTO()
        elif val == "_DimHint.DYNAMIC":
            return _DimHint.DYNAMIC()
        elif val == "_DimHint.STATIC":
            return _DimHint.STATIC()
        if not isinstance(val, str):
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected leaves in `spec['dynamic_shapes']` to be ints, None, Dim.AUTO/STATIC, symbols, "
                f" or derived expressions, got {val}",
            )
        if val not in dim_cache:
            raise UserError(
                UserErrorType.INVALID_INPUT,
                "Expected dims in `spec['dynamic_shapes']` to be tracked in `spec['dims']`, "
                f"got {val} which is not in {dims.keys()}",
            )
        return dim_cache[val]  # type: ignore[return-value]

    return tree_map(deserialize_shape, dynamic_shapes)
