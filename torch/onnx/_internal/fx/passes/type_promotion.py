# mypy: allow-untyped-defs
# Owner(s): ["module: onnx"]
# [MODULE DOC] FX pass that makes implicit type promotion explicit for ONNX export. Defines
# promotion rule classes (elementwise, division, reduction, all/any, sum-like), a generated
# plus a hand-curated rule set, a table keyed by namespace and op name, a generator that
# distills elementwise rules from torch reference ops, an interpreter that inserts dtype
# casts around nodes, and the InsertTypePromotion pass driving it. Import time: creates the
# module logger and builds the two rule sets (rule objects only; ops are checked later).
from __future__ import annotations

import abc
import dataclasses
import inspect
import logging
from typing import Any, TYPE_CHECKING

import torch
import torch._dispatch.python
import torch._ops
import torch.fx
import torch.fx.traceback as fx_traceback
from torch import _prims_common, _refs
from torch._prims_common import (
    ELEMENTWISE_TYPE_PROMOTION_KIND,
    wrappers as _prims_common_wrappers,
)
from torch._refs import linalg as _linalg_refs, nn as _nn_refs, special as _special_refs
from torch._refs.nn import functional as _functional_refs
from torch.fx.experimental import proxy_tensor
from torch.onnx._internal.fx import _pass, type_utils as fx_type_utils
from torch.utils import _pytree


if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import ModuleType

    from torch._subclasses import fake_tensor


logger = logging.getLogger(__name__)


def _try_getclosurevars(func):
    # [FUNCTION DOC] HELPER: Returns the closure variable info of a callable, or None if the
    # object is not a supported kind of callable (type error is swallowed).
    try:
        return inspect.getclosurevars(func)
    except TypeError:
        return None


@dataclasses.dataclass
class TypePromotionSnapshot:
    # [CLASS DOC] Immutable-by-convention result of previewing a promotion rule: a mapping from
    # positional argument index to target dtype, a mapping from keyword name to target dtype,
    # and the expected output dtype. Arguments not listed are left alone.
    """Type promotion snapshot for a fx node and its inputs.

    Contains the promoted dtype for args and kwargs that needs promoting.
    Contains the expected node output dtype.
    """

    args_dtypes: Mapping[int, torch.dtype]
    """Mapping from arg position to dtype to promote to."""

    kwargs_dtypes: Mapping[str, torch.dtype]
    """Mapping from kwarg name to dtype to promote to."""

    out_dtype: torch.dtype
    """Expected output dtype of the node."""


class TypePromotionRule(abc.ABC):
    # [CLASS DOC] Abstract base for a promotion rule bound to one op (namespace plus op name).
    # State: namespace and op_name. Subclasses must supply hash, repr, equality and
    # preview_type_promotion; this base provides is_valid.
    """Base class for type promotion rule per 'torch.ops.{namespace}.{op_name}'."""

    def __init__(self, namespace: str, op_name: str) -> None:
        # [FUNCTION DOC] Stores the op namespace and op name for this rule.
        self.namespace = namespace
        self.op_name = op_name

    # Make this abstract as well because subclass needs to override __eq__().
    # A class that overrides __eq__() and does not define __hash__() will have its __hash__() implicitly set to None.
    # Ref: https://docs.python.org/3/reference/datamodel.html#object.__hash__
    @abc.abstractmethod
    def __hash__(self) -> int: ...

    @abc.abstractmethod
    def __repr__(self) -> str: ...

    @abc.abstractmethod
    def __eq__(self, other: object) -> bool: ...

    def is_valid(self) -> bool:
        # [FUNCTION DOC] Checks that the namespace and op name resolve to an operator packet in
        # the torch ops registry. Returns false (with a logged warning) if the op is missing or is
        # not a packet. Side effect: looking up an unknown namespace creates an empty module.
        """Check if the rule is valid."""
        # This always returns a module. If the module does not exist it will be created.
        module = getattr(torch.ops, self.namespace)
        py_op = getattr(module, self.op_name, None)
        if py_op is None:
            logger.warning(
                "Cannot find op: %s in module: %s", self.op_name, self.namespace
            )
            return False
        if not isinstance(py_op, torch._ops.OpOverloadPacket):
            logger.warning(
                "Op: torch.ops.%s.%s is not an OpOverloadPacket, got: %s",
                self.namespace,
                self.op_name,
                type(py_op),
            )
            return False

        return True

    @abc.abstractmethod
    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # [FUNCTION DOC] Abstract: computes which args and kwargs get which dtype and the
        # expected output dtype for concrete call arguments. Returns a snapshot.
        """Preview type promotion results for provided set of args and kwargs.

        Returns a TypePromotionSnapshot object that contains the promoted dtypes for
        the arguments and the expected output dtype.
        """
        ...


class ElementwiseTypePromotionRule(TypePromotionRule):
    # [CLASS DOC] Rule for elementwise ops. State: positions of positional args and names of
    # keyword args that take part in promotion, and the promotion kind from the prims helpers.
    # Previewing computes the common dtype of the participating args. A class flag (off) controls
    # whether the lower-precision compute dtype is used for inputs; when off, inputs are cast
    # straight to the result dtype for the default and int-to-float kinds.
    """Defines how to perform elementwise type promotion for 'torch.ops.{namespace}.{op_name}'."""

    _USE_OPMATH: bool = False
    """Whether to use opmath to compute the promoted input dtype.
    If used, upcasts will be inserted everywhere for lower precision models.
    Set to False and have torchlib handle upcasts in op implementation internally.
    """

    def __init__(
        self,
        namespace: str,
        op_name: str,
        promote_args_positions: Sequence[int],
        promote_kwargs_names: Sequence[str],
        promotion_kind: _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND,
    ) -> None:
        # [FUNCTION DOC] Stores the op identity, arg positions, kwarg names and promotion kind.
        """Constructs a TypePromotionRule for elementwise operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.add'.
            op_name: Name of the op. E.g. 'add' in 'torch.ops.aten.add'.
            promote_args_positions: Positions of args to promote.
            promote_kwargs_names: Names of kwargs to promote.
            promotion_kind: Type promotion kind. Refer to [_prims_common.elementwise_dtypes](https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promote_args_positions = promote_args_positions
        self.promote_kwargs_names = promote_kwargs_names
        self.promotion_kind = promotion_kind

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Text form listing namespace, op, positions, names and kind.
        return (
            f"ElementwiseTypePromotionRule('{self.namespace}', '{self.op_name}', "
            f"{self.promote_args_positions}, {self.promote_kwargs_names}, {self.promotion_kind})"
        )

    # pyrefly: ignore [bad-override]
    def __eq__(self, other: object, /) -> bool:
        # [FUNCTION DOC] HELPER: Equal if other is an elementwise rule with the same namespace, op
        # name, arg positions, kwarg names and promotion kind.
        if not isinstance(other, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == other.namespace
            and self.op_name == other.op_name
            and self.promote_args_positions == other.promote_args_positions
            and self.promote_kwargs_names == other.promote_kwargs_names
            and self.promotion_kind == other.promotion_kind
        )

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash of the class plus namespace and op name only (coarser than
        # equality, which is allowed).
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def _consolidate_input_dtype(
        self, computed_dtype: torch.dtype, result_dtype: torch.dtype
    ) -> torch.dtype:
        # [FUNCTION DOC] HELPER: Chooses the dtype inputs are cast to. With the compute-dtype flag
        # off and a default or int-to-float kind, returns the result dtype (avoids extra upcasts);
        # otherwise returns the computation dtype.
        """
        Although opmath is the right thing to do to retain on-par precision, it inserts
        upcasts everywhere in the graph. This is particularly hard for backend to optimize
        since there is no way to differentiate between inserted upcasts and model code
        casts. Hence we consolidate the input dtype to the result dtype to avoid this.
        """
        if not self._USE_OPMATH and self.promotion_kind in (
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
            _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
        ):
            return result_dtype
        return computed_dtype

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # [FUNCTION DOC] Gathers the configured positional args (present and not None) and kwargs
        # (present and not None), flattens them, and asks the prims helper for the compute and
        # result dtypes under the rule's kind. Returns a snapshot mapping every gathered arg
        # position and kwarg name to the consolidated input dtype, with the result dtype as the
        # output dtype. Errors from the helper (e.g. no tensor inputs) propagate.
        candidate_args = {
            i: args[i]
            for i in self.promote_args_positions
            if i < len(args) and args[i] is not None
        }
        candidate_kwargs = {
            name: kwargs[name]
            for name in self.promote_kwargs_names
            if name in kwargs and kwargs[name] is not None
        }

        computed_dtype, result_dtype = _prims_common.elementwise_dtypes(
            *_pytree.arg_tree_leaves(*candidate_args.values(), **candidate_kwargs),
            type_promotion_kind=self.promotion_kind,
        )

        consolidated_input_dtype = self._consolidate_input_dtype(
            computed_dtype, result_dtype
        )

        return TypePromotionSnapshot(
            dict.fromkeys(candidate_args.keys(), consolidated_input_dtype),
            dict.fromkeys(candidate_kwargs.keys(), consolidated_input_dtype),
            result_dtype,
        )


class DivElementwiseTypePromotionRule(ElementwiseTypePromotionRule):
    # [CLASS DOC] Elementwise rule for the aten div op over args zero and one. The promotion kind
    # is chosen per call from the rounding mode keyword (none: int-to-float; trunc or floor:
    # default), so the kind attribute is overwritten during previews.
    """Reference type promotion rule from torch._refs.div.

    Rule depends on the value of the `rounding_mode` argument.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Builds the div rule: namespace aten, op div, positions zero and one, no
        # kwargs, default kind (replaced per call).
        super().__init__(
            "aten",
            "div",
            promote_args_positions=(0, 1),
            promote_kwargs_names=(),
            promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # [FUNCTION DOC] Reads the rounding mode keyword, sets the promotion kind accordingly
        # (none: int-to-float; trunc or floor: default) and defers to the parent preview.
        # Raises a value error for any other mode. Side effect: mutates the rule's kind.
        # KNOWN ISSUE: mutating shared rule state per call is not thread safe and makes equality
        # and the mutable kind depend on call history.
        rounding_mode = kwargs.get("rounding_mode")
        if rounding_mode is None:
            # true_divide
            self.promotion_kind = (
                _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
            )
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "trunc":
            # trunc_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        if rounding_mode == "floor":
            # floor_divide
            self.promotion_kind = _prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
            return super().preview_type_promotion(args, kwargs)
        raise ValueError(f"Unknown rounding_mode: {rounding_mode}")


class ReductionTypePromotionRule(TypePromotionRule):
    # [CLASS DOC] Rule for reduction ops. State: a reduction output kind from the prims helpers.
    # Previewing promotes only the first argument (the input tensor) to the computation dtype and
    # reports the result dtype, honoring an optional dtype keyword.
    def __init__(
        self,
        namespace: str,
        op_name: str,
        promotion_kind: _prims_common.REDUCTION_OUTPUT_TYPE_KIND,
    ) -> None:
        # [FUNCTION DOC] Stores the op identity and the reduction output kind.
        """Constructs a TypePromotionRule for reduction operators.

        Args:
            namespace: Namespace of the op. E.g. 'aten' in 'torch.ops.aten.sum'.
            op_name: Name of the op. E.g. 'sum' in 'torch.ops.aten.sum'.
            promotion_kind: Type promotion kind. Refer to [_prims_common.reduction_dtypes]((https://github.com/pytorch/pytorch/blob/main/torch/_prims_common/__init__.py)) for detail.  # noqa: B950
        """
        super().__init__(namespace, op_name)
        self.promotion_kind = promotion_kind

    def __repr__(self) -> str:
        # [FUNCTION DOC] HELPER: Text form listing namespace, op and kind.
        return f"ReductionTypePromotionRule('{self.namespace}', '{self.op_name}', {self.promotion_kind})"

    # pyrefly: ignore [bad-override]
    def __eq__(self, other: object, /) -> bool:
        # [FUNCTION DOC] HELPER: Intended to compare namespace, op name and kind with another rule.
        # KNOWN ISSUE: the type check tests for the elementwise rule class instead of the
        # reduction class, so a reduction rule is never equal to another reduction rule (not even
        # itself via a copy) and could only match an elementwise rule that lacks the kind attribute
        # semantics; set membership still works only through identity.
        if not isinstance(other, ElementwiseTypePromotionRule):
            return False
        return (
            self.namespace == other.namespace
            and self.op_name == other.op_name
            and self.promotion_kind == other.promotion_kind
        )

    def __hash__(self) -> int:
        # [FUNCTION DOC] HELPER: Hash of the class plus namespace and op name.
        return f"{type(self)}:{self.namespace}.{self.op_name}".__hash__()

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # [FUNCTION DOC] Takes the first positional arg (must be a tensor, else assertion error;
        # also an assertion error if there are no args) and the optional dtype keyword, asks the
        # prims helper for computation and result dtypes, and falls back to the computation dtype
        # if no result dtype is given. Returns a snapshot casting arg zero to the computation dtype
        # with no kwarg changes.
        if len(args) < 1:
            raise AssertionError(
                f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
            )
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"{type(arg)=} is not torch.Tensor")
        dtype: torch.dtype | None = kwargs.get("dtype")

        computation_dtype, result_dtype = _prims_common.reduction_dtypes(
            arg, self.promotion_kind, dtype
        )
        if result_dtype is None:
            # Inspecting code, this can only happen when `promotion_kind` is `KEEP_PROMOTED_TYPE`.
            # Hence set same as computation_dtype.
            result_dtype = computation_dtype

        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class AllOrAnyReductionTypePromotionRule(ReductionTypePromotionRule):
    # [CLASS DOC] Reduction rule for the aten all and any ops: always computes in bool and keeps
    # uint8 results for uint8 inputs.
    """Reference type promotion rule from torch.ops.aten.all or torch.ops.aten.any.

    This is a special case where computation dtype is always torch.bool.
    The result dtype is always uint8 if `dtype` kwarg is uint8, otherwise torch.bool.
    """

    def __init__(self, op_name: str) -> None:
        # [FUNCTION DOC] Builds an aten rule for the given op name with the always-bool kind.
        super().__init__(
            "aten",
            op_name,
            _prims_common.REDUCTION_OUTPUT_TYPE_KIND.ALWAYS_BOOL,
        )

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # [FUNCTION DOC] Requires a tensor first arg (assertion error otherwise). Casts it to bool;
        # the result dtype is uint8 for a uint8 input, else bool.
        # KNOWN ISSUE: the class docstring says the result depends on the dtype keyword, but the
        # code looks only at the input tensor's dtype and ignores the keyword.
        if len(args) < 1:
            raise AssertionError(
                f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
            )
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"{type(arg)=} is not torch.Tensor")
        computation_dtype = torch.bool
        # Preserves uint8 -- probably a legacy mask thing
        result_dtype = torch.uint8 if arg.dtype == torch.uint8 else torch.bool
        return TypePromotionSnapshot(
            {0: computation_dtype},
            {},
            result_dtype,
        )


class SumLikeReductionTypePromotionRule(ReductionTypePromotionRule):
    # [CLASS DOC] Reduction rule for sum-like ops (sum, prod, cumsum, cumprod): integral and bool
    # inputs accumulate in int64 unless a dtype keyword overrides it.
    """Reference type promotion rule from torch.ops.aten.sum.

    This is a special case where computation dtype is always torch.int64 for integral arg,
    unless overridden by `dtype` kwarg.
    """

    def preview_type_promotion(
        self, args: tuple, kwargs: dict
    ) -> TypePromotionSnapshot:
        # [FUNCTION DOC] Requires a tensor first arg (assertion error otherwise). If no dtype
        # keyword is given, picks int64 for bool or integer inputs and the input dtype otherwise,
        # then defers to the parent preview with only that dtype keyword (other kwargs dropped).
        if len(args) < 1:
            raise AssertionError(
                f"Reduction op torch.ops.{self.namespace}.{self.op_name} expects at least one argument"
            )
        arg = args[0]
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"{type(arg)=} is not torch.Tensor")
        dtype: torch.dtype | None = kwargs.get("dtype")
        # The below logic is copied from `torch/_refs/__init__.py` reduction ops impl.
        if dtype is None:
            if _prims_common.is_boolean_dtype(
                arg.dtype
            ) or _prims_common.is_integer_dtype(arg.dtype):
                dtype = torch.int64
            else:
                dtype = arg.dtype
        return super().preview_type_promotion(args, {"dtype": dtype})


# NOTE: [Update type promotion rule]
# BELOW TABLE IS GENERATED FROM `TypePromotionRuleSetGenerator.generate_from_torch_refs`.
# DO NOT EDIT MANUALLY !!!
# For missing rules or discrepancies, please
# 1. Run `pytest test/onnx/test_fx_type_promotion.py` to validate if the generated rule set is current.
#    If it is not, update with new generated set.
# 2. If discrepancies still exist, consider debugging torch._refs or report a bug.
# 3. If rules are still missing, add them to `_EXTRA_TYPE_PROMOTION_RULE_SET` or report a bug.
# Check `TypePromotionRule` class for how each rule is defined and used.
_GENERATED_ATEN_TYPE_PROMOTION_RULE_SET = {
    ElementwiseTypePromotionRule(
        "aten", "abs", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "abs_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "acosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "add_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcdiv_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addcmul_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "addr", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "asinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan2_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "atanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_left_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "bitwise_right_shift_",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "bitwise_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cat", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cauchy_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ceil_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "celu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "clamp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "copysign_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cos_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "cosh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "deg2rad_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "digamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "dot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "elu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "eq_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erf_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "erfinv_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exp_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "expm1_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "exponential_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fill", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "floor_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmax", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmin", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "fmod_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "frac_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gcd_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ge_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gelu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "geometric_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "glu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "gt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "hardtanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "heaviside_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "huber_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "hypot_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "i0_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igamma_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "igammac_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "isfinite", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isnan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isneginf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isposinf", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "isreal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "l1_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lcm_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "le", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "le_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "leaky_relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lerp_", [0, 1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lgamma_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log10_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log1p_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log2_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "log_normal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logaddexp2", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_and_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_not_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_or_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logical_xor_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "logit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "logsumexp", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "lt_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "maximum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "minimum", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mish_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mse_loss", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "mul_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "ne_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "neg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nextafter_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "nll_loss", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "normal", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "pdist", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "poisson_nll_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "prelu", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rad2deg_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "reciprocal_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "relu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "remainder_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "round", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "rsqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "selu_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sgn_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sigmoid_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sign_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "signbit", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.ALWAYS_BOOL
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sin_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sinh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "smooth_l1_loss",
        [0, 1],
        [],
        ELEMENTWISE_TYPE_PROMOTION_KIND.COMPLEX_TO_FLOAT,
    ),
    ElementwiseTypePromotionRule(
        "aten", "softplus", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sqrt_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "square", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "square_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.BOOL_TO_LONG
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "sub_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tan_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "tanh_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "threshold_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "true_divide_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "trunc_", [0], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "vdot", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT
    ),
    ElementwiseTypePromotionRule(
        "aten", "where", [1, 2], [], ELEMENTWISE_TYPE_PROMOTION_KIND.NO_OPMATH
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
    ElementwiseTypePromotionRule(
        "aten", "xlogy_", [0, 1], [], ELEMENTWISE_TYPE_PROMOTION_KIND.INT_TO_FLOAT
    ),
}

# Manually curated extra type promotion rules. Please see NOTE [Update type promotion rule]
# before adding new rules.
_EXTRA_TYPE_PROMOTION_RULE_SET = {
    # torch._refs skips type promotion decoration for `clamp_min` and `clamp_max` since
    # the call is routed to the decorated `aten.clamp` op.
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_max",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    ElementwiseTypePromotionRule(
        "aten",
        "clamp_min",
        promote_args_positions=(0, 1),
        promote_kwargs_names=(),
        promotion_kind=_prims_common.ELEMENTWISE_TYPE_PROMOTION_KIND.DEFAULT,
    ),
    # torch.ops.aten.div.Tensor_mode applies different type promotion rules
    # depending on the value of the `mode` argument.
    DivElementwiseTypePromotionRule(),
    # Manually curating reduction ops since the logic is written inside the op reference
    # implementation.
    AllOrAnyReductionTypePromotionRule("all"),
    AllOrAnyReductionTypePromotionRule("any"),
    ReductionTypePromotionRule(
        "aten",
        "amax",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    ReductionTypePromotionRule(
        "aten",
        "amin",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    # torch.ops.aten.mean is a special case that does not need type promotion.
    ReductionTypePromotionRule(
        "aten",
        "std",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "std_mean",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    ReductionTypePromotionRule(
        "aten",
        "var",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.COMPLEX_TO_FLOAT,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumprod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "cumsum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "prod",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
    SumLikeReductionTypePromotionRule(
        "aten",
        "sum",
        promotion_kind=_prims_common.REDUCTION_OUTPUT_TYPE_KIND.SAME,
    ),
}


class ElementwiseTypePromotionRuleSetGenerator:
    # [CLASS DOC] HELPER: Stateless utility (class methods only) that inspects the decorated
    # reference ops in several torch reference modules to rebuild elementwise promotion rules.
    # Used to regenerate or validate the generated rule table, not at export time.
    """Hackily distilling info from reference ops decorated with elementwise type promotion rule.

    The goal is to retrieve the decorator

    ```python
        @elementwise_type_promotion_wrapper(
            type_promoting_args=("a", "b"),
            type_promotion_kind=type_promotion_kind,
        )
    ```

    from the reference ops. It provides info as for which arguments are promoted
    and what kind of promotion is applied.
    """

    @classmethod
    def generate_from_torch_refs(cls) -> set[ElementwiseTypePromotionRule]:
        # [FUNCTION DOC] Parses the main, nn, linalg, special and nn-functional reference modules
        # and returns the union of the valid rules found, as a set.
        """Parse type promotion rules from reference ops under torch._C._refs."""
        rule_set = set()
        rule_set.update(cls._parse_torch_refs(_refs))
        rule_set.update(cls._parse_torch_refs(_nn_refs))
        rule_set.update(cls._parse_torch_refs(_linalg_refs))
        rule_set.update(cls._parse_torch_refs(_special_refs))
        rule_set.update(cls._parse_torch_refs(_functional_refs))
        return rule_set

    @classmethod
    def _parse_torch_refs(
        cls, ref_module: ModuleType
    ) -> set[ElementwiseTypePromotionRule]:
        # [FUNCTION DOC] HELPER: For each exported name of a reference module, parses the rule from
        # the op and keeps it if found and valid (op exists as a packet). Logs the module name.
        # Returns the set of kept rules.
        logger.info("Processing module: %s", ref_module.__name__)
        rule_set = set()
        for name in ref_module.__all__:
            decorated_op = getattr(ref_module, name)
            rule = cls._parse_type_promotion_rule_from_refs_op(decorated_op)
            if rule is not None and rule.is_valid():
                rule_set.add(rule)

        return rule_set

    @classmethod
    def _parse_type_promotion_rule_from_refs_op(
        cls,
        decorated_op: Callable,
    ) -> ElementwiseTypePromotionRule | None:
        # [FUNCTION DOC] HELPER: Follows the chain of wrapped functions through closure variables
        # until the elementwise type promotion wrapper object is found. Then maps the wrapper's
        # promoting argument names to positions (positional parameters, by signature index) or
        # keyword names (keyword-only parameters) and builds an aten rule using the op's name and
        # the wrapper's kind. Returns None, logging a warning, if no wrapper is found.
        # KNOWN ISSUE: the position counter advances for every parameter, including variadic and
        # keyword-only ones, which only matters if they precede positional ones (not expected).
        """Retrieve and parse type promotion decorator from op under torch._refs."""
        fn = decorated_op
        type_promo_wrapper = None
        while fn_closure_vars := _try_getclosurevars(fn):
            if "fn" not in fn_closure_vars.nonlocals:
                break
            if "self" in fn_closure_vars.nonlocals and isinstance(
                fn_closure_vars.nonlocals["self"],
                _prims_common_wrappers.elementwise_type_promotion_wrapper,
            ):
                type_promo_wrapper = fn_closure_vars.nonlocals["self"]
                break
            fn = fn_closure_vars.nonlocals["fn"]

        if type_promo_wrapper is not None:
            signature = inspect.signature(decorated_op)

            pos = 0
            promote_args_positions = []
            promote_kwargs_names = []

            if type_promo_wrapper.type_promoting_arg_names is not None:
                for name, param in signature.parameters.items():
                    if name in type_promo_wrapper.type_promoting_arg_names:
                        if param.kind in (
                            param.POSITIONAL_OR_KEYWORD,
                            param.POSITIONAL_ONLY,
                        ):
                            promote_args_positions.append(pos)
                        elif param.kind == param.KEYWORD_ONLY:
                            promote_kwargs_names.append(name)
                    pos += 1

            return ElementwiseTypePromotionRule(
                "aten",
                decorated_op.__name__,
                promote_args_positions=promote_args_positions,
                promote_kwargs_names=promote_kwargs_names,
                promotion_kind=type_promo_wrapper.type_promotion_kind,
            )

        logger.warning(
            "Cannot find type promotion rule for: %s.%s",
            decorated_op.__module__,
            decorated_op.__name__,
        )
        return None


class TypePromotionTable:
    # [CLASS DOC] Lookup table from the string namespace.op to a promotion rule. Built at
    # construction from the generated and extra rule sets (extras win on conflict); more rules
    # can be added later.
    """Type promotion table for torch.ops."""

    def __init__(self) -> None:
        # [FUNCTION DOC] Creates the table and adds every generated rule, then every extra rule.
        # Raises a value error if any rule refers to a non-existent op.
        self._rule_table = {}
        for rule in _GENERATED_ATEN_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)
        for rule in _EXTRA_TYPE_PROMOTION_RULE_SET:
            self.add_rule(rule)

    def add_rule(self, rule: TypePromotionRule) -> None:
        # [FUNCTION DOC] Validates the rule and stores it under namespace.op, replacing any
        # existing rule. Raises a value error if the rule's op is invalid.
        # KNOWN ISSUE: the docstring mentions a module argument that does not exist.
        """Add a type promotion rule for a python op in a torch.ops module.

        Args:
            rule: Type promotion rule.
            module: Module containing the op. E.g. torch.ops.aten.

        Raises:
            ValueError: If the rule is invalid.
        """
        if not rule.is_valid():
            raise ValueError(f"Invalid type promotion rule: {rule}")
        self._rule_table[f"{rule.namespace}.{rule.op_name}"] = rule

    def get_rule(self, py_op: torch._ops.OpOverloadPacket) -> TypePromotionRule | None:
        # [FUNCTION DOC] Returns the rule whose key equals the string form of the op packet
        # (namespace.op), or None if there is none.
        """Get type promotion rule for a python op under 'torch.ops.<namespace>'."""
        return self._rule_table.get(str(py_op), None)


def get_type_promotion_rule(
    node: torch.fx.Node,
    type_promotion_table: TypePromotionTable,
) -> TypePromotionRule | None:
    # [FUNCTION DOC] Returns the table's rule for a node whose target is an operator overload
    # (looked up by its packet), or None if the target is not an overload or has no rule.
    """Get type promotion rule for a node.

    Args:
        node: Node to get type promotion rule for.
        type_promotion_table: Type promotion table.

    Returns:
        Type promotion rule for the node. None if no rule is found or if the node is not
        representing a torch operator.
    """
    op = node.target
    if not isinstance(op, torch._ops.OpOverload):
        return None
    if (rule := type_promotion_table.get_rule(op.overloadpacket)) is None:
        return None

    return rule


def find_compatible_op_overload(
    op: torch._ops.OpOverloadPacket, args: tuple, kwargs: dict
) -> torch._ops.OpOverload:
    # [FUNCTION DOC] Tries each overload of the packet in order and returns the first whose
    # schema accepts the given args and kwargs. Raises a runtime error listing every overload's
    # failure if none matches. Raises an assertion error if an overload attribute is not an
    # overload object. Read-only.
    """Find compatible OpOverload for an OpOverloadPacket using provided args and kwargs.

    Each "call_function" fx.Node in the fx.GraphModule has a target that represents a torch._ops.OpOverload.
    The OpOverload contains an OpOverloadPacket that holds all the available overloads for the operation.

    During the type promotion pass, there are cases where the types of the args and kwargs may change,
    such as promoting Python numbers to tensors. Consequently, the original OpOverload might not be
    compatible with the updated args and kwargs. This function is used to identify the compatible
    OpOverload for the given args and kwargs.

    Args:
        op: OpOverloadPacket to find compatible OpOverload for.
        args: The positional arguments to consider for compatibility.
        kwargs: The keyword arguments to consider for compatibility.

    Returns:
        torch._ops.OpOverload: The compatible OpOverload found for the given args and kwargs.

    Raises:
        RuntimeError: If no compatible op overload is found.

    Examples:
        >>> import torch
        >>> packet = torch.ops.aten.pow
        >>> args = (torch.tensor([1.0, 2.0]), 2)
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Scalar'
        >>> args = (torch.tensor([1.0, 2.0]), torch.tensor(2.0))
        >>> find_compatible_op_overload(packet, args, {})._overloadname
        'Tensor_Tensor'
    """
    exceptions = {}
    for overload_name in op.overloads():
        op_overload = getattr(op, overload_name)
        if not isinstance(op_overload, torch._ops.OpOverload):
            raise AssertionError(f"Expected OpOverload, got {type(op_overload)}")
        try:
            torch._C._check_schema_allow_fake_script_object(
                op_overload._schema, *args, **kwargs
            )
        except RuntimeError as e:
            exceptions[overload_name] = e
        else:
            return op_overload

    error_messages = "\n".join(
        f"Overload name {name}:\n{exception}" for name, exception in exceptions.items()
    )
    raise RuntimeError(
        f"Cannot find a compatible OpOverload for {op} with the provided "
        f"arguments. Tried overloads:\n{error_messages}"
    )


class _TypePromotionInterpreter(torch.fx.Interpreter):
    # [CLASS DOC] HELPER: Graph interpreter that, before running each node, applies the matching
    # promotion rule by inserting cast nodes for inputs (and for the output when the computation
    # dtype differs from the expected one), updating node targets and metadata in place. State:
    # the rule table. Running it mutates the graph being interpreted.
    """Interpreter that inserts type promotion for each node."""

    def __init__(
        self,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable,
    ) -> None:
        # [FUNCTION DOC] HELPER: Initializes the interpreter on the module and keeps the table.
        super().__init__(module)
        self.type_promotion_table = type_promotion_table

    def _run_node_and_set_meta(self, node) -> Any:
        # [FUNCTION DOC] HELPER: Runs the node with the base interpreter, stores the output in the
        # environment, copies current traced metadata keys not already on the node, and recomputes
        # the node's value metadata from the output. Returns the output. Mutates the node.
        """Run node and set meta according to `fx_traceback.get_current_meta()`.

        This should be used on new nodes or nodes that have been modified.
        By default `Interpreter.run_node` does not update `node.meta`.
        Set `node.meta` to the current meta, except for `node.meta["val"]`, which is
        recomputed.
        """
        out = super().run_node(node)
        # Update interpreter env state with new output value.
        self.env[node] = out
        node.meta.update(
            (k, v)
            for k, v in fx_traceback.get_current_meta().items()
            if k not in node.meta
        )
        node.meta["val"] = proxy_tensor.extract_val(out)
        return out

    def _create_node(
        self,
        graph: torch.fx.Graph,
        op_type: str,
        target: torch.fx.node.Target,
        args: tuple,
        kwargs: dict,
    ) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Appends a node of the given op type to the graph at the current
        # insertion point (assertion error for an unknown op type), runs it to fill metadata and
        # the environment, and returns it.
        """Create a node and set its metadata."""
        if op_type not in (
            "call_function",
            "call_method",
            "get_attr",
            "call_module",
            "placeholder",
            "output",
        ):
            raise AssertionError(f"Unexpected op_type: {op_type}")
        node = getattr(graph, op_type)(target, args, kwargs)
        self._run_node_and_set_meta(node)
        return node

    def _rerun_node_after_type_promotion(
        self,
        node: torch.fx.Node,
        expected_out_dtype: torch.dtype,
    ) -> None:
        # [FUNCTION DOC] HELPER: After a node's inputs changed, retargets it to a compatible
        # overload and reruns it. Asserts the old value metadata exists, the target is an overload,
        # the output kind is unchanged, and the old tensor dtype equals the expected dtype. If the
        # new tensor dtype differs (op math), inserts a conversion node after it and redirects all
        # users to it. Raises not-implemented for symbolic, list or tuple outputs and a runtime
        # error for any other output type. Mutates the graph.
        """Rerun a node after type promotion and update node.meta["val"] with the output value."""
        node_val = node.meta.get("val", None)
        if node_val is None:
            raise AssertionError(f"Node {node} node.meta['val'] is not set.")
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        target = node.target
        if not isinstance(target, torch._ops.OpOverload):
            raise AssertionError(f"Expected OpOverload, got {type(target)}")
        node.target = find_compatible_op_overload(target.overloadpacket, args, kwargs)

        new_node_val = self._run_node_and_set_meta(node)
        if not isinstance(new_node_val, type(node_val)):
            raise AssertionError(
                f"run_node output type should not change between runs. "
                f"Got {type(new_node_val)}, expect {type(node_val)}."
            )

        if isinstance(node_val, torch.Tensor):
            prev_node_dtype = node_val.dtype

            if prev_node_dtype != expected_out_dtype:
                raise AssertionError(
                    f"node.meta['val'].dtype({prev_node_dtype}) does not agree with "
                    f"type promotion rule({expected_out_dtype})."
                )

            if new_node_val.dtype != expected_out_dtype:
                # With explicit type promotion, the expected result dtype may not be
                # the same as the computation dtype. This is referred to as "op math".
                # We need to explicitly cast the output back to the expected dtype.
                # See more about "op math" topic at `_prims_common.elementwise_dtypes`.
                graph = node.graph
                with graph.inserting_after(node):
                    output_cast_node = self._create_node(
                        graph,
                        "call_function",
                        torch.ops.prims.convert_element_type.default,
                        (node,),
                        {"dtype": expected_out_dtype},
                    )
                    node.replace_all_uses_with(output_cast_node)
                    output_cast_node.args = (node,)
                    logger.info(
                        "Node '%s' output dtype becomes %s due to op math. "
                        "Cast back to %s.",
                        node,
                        new_node_val.dtype,
                        expected_out_dtype,
                    )

        elif fx_type_utils.is_torch_symbolic_type(node_val):
            raise NotImplementedError(
                "Type promotion does not support node output of sym types."
            )
        elif isinstance(node_val, (list, tuple)):
            raise NotImplementedError(
                "Type promotion does not support node output of list or tuple."
            )
        else:
            raise RuntimeError(f"Unexpected node output type: {type(node_val)}.")

    def _maybe_promote_arg(
        self,
        node: torch.fx.Node,
        fx_arg: torch.fx.node.Argument,
        dtype: torch.dtype | None,
    ) -> torch.fx.node.Argument:
        # [FUNCTION DOC] HELPER: Returns the argument unchanged if dtype is None or it already has
        # that dtype. Tensor-valued nodes of another dtype get a conversion node inserted before
        # the consumer; symbolic numbers and plain Python numbers of another dtype become a scalar
        # tensor node of the target dtype; tuples and lists are handled element-wise preserving the
        # container type. Raises not-implemented for other argument kinds (including a node whose
        # value is neither tensor nor symbolic). Returns the new or old argument.
        """Promote fx_arg to dtype if necessary."""
        if dtype is None:
            logger.info(
                "Argument %s is not promoted. Not mentioned by type promotion rule.",
                fx_arg,
            )
            return fx_arg

        if isinstance(fx_arg, torch.fx.Node):
            arg_val = self.env[fx_arg]
            if isinstance(arg_val, torch.Tensor):
                if (old_dtype := arg_val.dtype) != dtype:
                    # Promote tensor to dtype.
                    graph = node.graph
                    with graph.inserting_before(node):
                        logger.info(
                            "Argument %s(%s) is promoted to %s.",
                            fx_arg,
                            old_dtype,
                            dtype,
                        )
                        return self._create_node(
                            graph,
                            "call_function",
                            torch.ops.prims.convert_element_type.default,
                            (fx_arg,),
                            {"dtype": dtype},
                        )
                logger.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
                return fx_arg
            elif fx_type_utils.is_torch_symbolic_type(arg_val):
                arg_type = type(arg_val)
                equivalent_dtype = fx_type_utils.from_scalar_type_to_torch_dtype(
                    arg_type
                )
                if equivalent_dtype is None:
                    raise AssertionError(f"Unexpected arg_type: {arg_type}")
                if equivalent_dtype != dtype:
                    # Promote Sym number to tensor of dtype.
                    graph = node.graph
                    with graph.inserting_before(node):
                        logger.info(
                            "Argument %s(Scalar of equivalent dtype: %s) "
                            "is promoted to %s.",
                            fx_arg,
                            equivalent_dtype,
                            dtype,
                        )
                        return self._create_node(
                            graph,
                            "call_function",
                            torch.ops.aten.scalar_tensor.default,
                            (fx_arg,),
                            {"dtype": dtype},
                        )
                logger.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
                return fx_arg
        elif (
            equivalent_dtype := fx_type_utils.from_scalar_type_to_torch_dtype(
                type(fx_arg)
            )
        ) is not None:
            if equivalent_dtype != dtype:
                # Promote number to tensor of dtype.
                # The op should have overload that supports tensor for this arg, otherwise
                # the type promotion rule should not suggest promoting this arg.
                graph = node.graph
                with graph.inserting_before(node):
                    logger.info(
                        "Argument %s(Scalar of equivalent dtype: %s) "
                        "is promoted to %s.",
                        fx_arg,
                        equivalent_dtype,
                        dtype,
                    )
                    return self._create_node(
                        graph,
                        "call_function",
                        torch.ops.aten.scalar_tensor.default,
                        (fx_arg,),
                        {"dtype": dtype},
                    )
            logger.info("Argument %s is not promoted. Already %s.", fx_arg, dtype)
            return fx_arg
        elif isinstance(fx_arg, (tuple, list)):
            logger.info("Argument %s is a tuple/list. Promoting each element.", fx_arg)
            return type(fx_arg)(
                self._maybe_promote_arg(node, fx_arg_elem, dtype)
                for fx_arg_elem in fx_arg
            )

        raise NotImplementedError(f"Unknown fx arg type: {type(fx_arg)}")

    def _maybe_promote_node(
        self,
        node: torch.fx.Node,
        rule: TypePromotionRule,
    ) -> torch.fx.Node:
        # [FUNCTION DOC] HELPER: Previews the rule on the node's concrete args, promotes each
        # positional and keyword argument per the snapshot, and, if anything changed, assigns the
        # new args and kwargs and reruns the node with the expected output dtype. Returns the node.
        """Promote node inputs and outputs according to type promotion rule."""
        args, kwargs = self.fetch_args_kwargs_from_env(node)
        type_promotion_info = rule.preview_type_promotion(args, kwargs)
        new_args = []
        new_kwargs = {}
        for i, arg in enumerate(node.args):
            new_args.append(
                self._maybe_promote_arg(
                    node, arg, type_promotion_info.args_dtypes.get(i, None)
                )
            )

        for name, arg in node.kwargs.items():
            new_kwargs[name] = self._maybe_promote_arg(
                node, arg, type_promotion_info.kwargs_dtypes.get(name, None)
            )
        new_args = tuple(new_args)

        if node.args != new_args or node.kwargs != new_kwargs:
            node.args = new_args
            node.kwargs = new_kwargs
            self._rerun_node_after_type_promotion(node, type_promotion_info.out_dtype)

        return node

    def run_node(self, n: torch.fx.Node) -> Any:
        # [FUNCTION DOC] Override of the interpreter hook: under the current-node context, looks
        # up a rule for the node and, if any, promotes its inputs/outputs in the graph; then runs
        # the node with the base implementation and returns its value.
        """This method is an override which inserts type promotion nodes as needed.

        For each `call_function` node, an initial check is conducted to determine if a type
        promotion rule is applicable. If a relevant rule exists, type casting nodes are
        introduced for the corresponding arguments. The OpOverload of the node is updated
        to one that accommodates the promoted types. Should the output type be different,
        type casting node is inserted for this output.

        The call `super().run_node(node)` is guaranteed to be invoked for each node.
        In the case of new or modified nodes, the result of `super().run_node(node)` is
        used to update its `node.meta["val"]` value.
        """
        with self._set_current_node(n):
            if rule := get_type_promotion_rule(n, self.type_promotion_table):
                self._maybe_promote_node(n, rule)

        return super().run_node(n)


class InsertTypePromotion(_pass.Transform):
    # [CLASS DOC] Graph transform pass that makes type promotion explicit. State: an interpreter
    # built on the module with a promotion table (default one if none given). Running it replays
    # the graph with fake argument values under the fake mode, mutating the module in place.
    """Explicitly insert type promotion ops to the graph.

    Underneath, the main pass is driven by `_TypePromotionInterpreter`, which is a subclass
    of `torch.fx.Interpreter` to interpret the fx.Graph and perform the insertion of type
    promotion operations.

    By re-running the new and modified nodes using the interpreter, we can update the
    metadata, specifically the fake tensor stored under node.meta["val"], and ensure it
    reflects the latest changes.
    """

    def __init__(
        self,
        module: torch.fx.GraphModule,
        type_promotion_table: TypePromotionTable | None = None,
    ) -> None:
        # [FUNCTION DOC] Initializes the transform on the module and creates the interpreter with
        # the given table, or a freshly built default table when none (or an empty one) is given.
        super().__init__(module)
        self.interpreter = _TypePromotionInterpreter(
            module, type_promotion_table or TypePromotionTable()
        )

    def _fetch_fake_args(
        self,
    ) -> Sequence[
        fake_tensor.FakeTensor
        | float
        | int
        | bool
        | torch.SymInt
        | torch.SymFloat
        | torch.SymBool
        | None
    ]:
        # [FUNCTION DOC] HELPER: Collects the value metadata of each placeholder node in graph
        # order. If reading metadata raises a runtime error, an unused placeholder yields None,
        # while a used one raises a runtime error asking for pre-existing fake args. Returns the
        # list.
        """Fetch fake args from fx graph.

        For each argument, try to fetch fake tensor from the matching placeholder node.
        """
        fake_args = []
        for node in self.module.graph.nodes:
            if node.op == "placeholder":
                try:
                    # Meta value can be torch.Tensor, int, float, bool,
                    # torch.SymInt, torch.SymFloat, torch.SymBool.
                    meta_value = _val = node.meta.get("val", None)
                except RuntimeError as e:
                    if not node.users:
                        # If the placeholder is not used, we can safely ignore it and put
                        # None as placeholder.
                        meta_value = None
                    else:
                        raise RuntimeError(
                            "Cannot fetch symbolic fake args from fx graph. "
                            "InsertTypePromotion pass needs to run with pre-existing fake args, "
                            "Otherwise the pass will produce inaccurate dynamic shape. "
                        ) from e

                fake_args.append(meta_value)
        return fake_args

    def _run(self, *args, **kwargs) -> torch.fx.GraphModule:
        # [FUNCTION DOC] Pass entry point. Rejects any positional or keyword arguments (assertion
        # error) since inputs come from graph metadata. Fetches fake args, requires a detectable
        # fake mode (assertion error otherwise), and runs the interpreter under that mode, the
        # python dispatcher and node-metadata preservation. Returns the mutated module.
        if args:
            raise AssertionError(
                "`InsertTypePromotion` deduces symbolic fake arguments from the graph. "
                "It does not accept concrete arguments as input because this pass requires "
                "re-running the graph. When executed with newly faked concrete arguments, "
                "the pass loses the symbolic dynamic shape information."
            )
        if kwargs:
            raise AssertionError("`kwargs` is not supported")

        fake_args = self._fetch_fake_args()
        fake_mode = self.fake_mode
        if fake_mode is None:
            raise AssertionError("Cannot detect fake_mode.")

        # Use the python dispatcher to run through some python kernels which
        # can better handle symints. Without this, some SymInts can become static
        # when there are dynamic shapes.
        dispatcher_mode = torch._dispatch.python.enable_python_dispatcher()
        with fake_mode, dispatcher_mode, fx_traceback.preserve_node_meta():
            self.interpreter.run(*fake_args)

        return self.module
