# [MODULE DOC] Shared helpers for the ComplexTensor tests. Provides
# COMPLEX_DTYPES, the Variant enum (Op, GradCheck, Distributed), Descriptor (a
# frozen record naming op, variant, device type, dtype, used to look up skips and
# tolerances with None as wildcard), and a TestCase subclass with consistency and
# gradient checkers. Also builds the op-info lists at import time: complex_op_db,
# implemented_op_db, force_test_op_db, and the set of ops without tests.
# Import-time side effect: emits a UserWarning listing implemented ops that have
# no matching op-info. Skips and tolerances are imported lazily from
# test_complex_tensor (package-relative or top-level).
from __future__ import annotations

from dataclasses import dataclass, field, fields
from enum import auto, Enum
from typing import Any, TYPE_CHECKING

import torch
import torch.distributed as dist
from torch._subclasses.complex_tensor._ops.common import (
    _as_complex_tensor,
    _as_interleaved,
    _get_op_name,
    COMPLEX_OPS_TABLE,
    COMPLEX_TO_REAL,
    FORCE_TEST_LIST,
    is_binary_nonlinear_op,
    OpOverloadPacket,
)
from torch.autograd.gradcheck import gradcheck
from torch.testing._internal.common_methods_invocations import op_db
from torch.testing._internal.common_utils import TestCase as PytorchTestCase
from torch.utils._pytree import tree_flatten, tree_map


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch.distributed.tensor import DTensor
    from torch.testing._internal.opinfo.core import OpInfo

COMPLEX_DTYPES = set(COMPLEX_TO_REAL)

# Native complex BLAS uses a single GEMM; ComplexTensor uses four real GEMMs.
BINARY_NONLINEAR_CUDA_TOL = {"rtol": 5e-6, "atol": 2e-5}


class Variant(Enum):
    # [CLASS DOC] Kinds of check: Op (plain tensor ops), GradCheck (gradient check)
    # and Distributed (DTensor-wrapped inputs). No methods.
    Op = auto()
    GradCheck = auto()
    Distributed = auto()


def _as_local(arg: DTensor | Any) -> torch.Tensor | Any:
    # [FUNCTION DOC] HELPER: Returns the full local tensor if the argument is a
    # DTensor (and distributed is available), otherwise returns the argument
    # unchanged.
    if not (dist.is_available() and isinstance(arg, dist.tensor.DTensor)):
        return arg

    return arg.full_tensor()


def _as_complex_dtensor(arg: torch.Tensor | Any) -> torch.Tensor | Any:
    # [FUNCTION DOC] HELPER: Converts a tensor to a ComplexTensor and wraps it as a
    # DTensor from local data; non-tensor arguments are returned unchanged.
    if not isinstance(arg, torch.Tensor):
        return arg

    return dist.tensor.DTensor.from_local(_as_complex_tensor(arg))


TRANSFORM_FUNCS = {
    Variant.Op: _as_complex_tensor,
    Variant.Distributed: _as_complex_dtensor,
}


@dataclass(frozen=True, kw_only=True)
class Descriptor:
    # [CLASS DOC] Frozen keyword-only record identifying a test case: op overload
    # packet, variant (may be None), and optional device type and dtype (None means
    # any). Hashable, so it is used as a key in the skip and tolerance tables. Key
    # method: matches.
    op: OpOverloadPacket
    variant: Variant | None
    device_type: str | None = field(default=None)
    dtype: torch.dtype | None = field(default=None)

    def matches(self, other: Descriptor) -> bool:
        # [FUNCTION DOC] Returns true if every field of this descriptor equals the other's
        # or is None on at least one side (None acts as a wildcard); returns false if the
        # field sets differ or any non-None pair differs.
        fields1 = fields(self)
        fields2 = fields(other)
        if fields1 != fields2:
            return False

        for f in fields1:
            f1 = getattr(self, f.name)
            f2 = getattr(other, f.name)
            if f1 is not None and f2 is not None and f1 != f2:
                return False

        return True


class TestCase(PytorchTestCase):
    # [CLASS DOC] Base test case for ComplexTensor tests, extending the PyTorch test
    # case. Holds no state. Methods: assertSameResult (compare two callables),
    # conditional_skip and get_extra_kwargs (table lookups), check_consistency
    # (forward comparison) and check_grad (gradcheck).
    def assertSameResult(
        self,
        expected: Callable[[], Any],
        actual: Callable[[], Any],
        *args,
        **kwargs,
    ) -> None:
        # [FUNCTION DOC] Calls both zero-argument callables, capturing any exception. If
        # only the actual one raises, that exception is re-raised; if only the expected
        # one raises, fails the test on the type mismatch. If neither raised, asserts the
        # results have the same tree structure and compares each leaf (converted to
        # interleaved complex and gathered if DTensor) with assertEqual, forwarding extra
        # args and kwargs such as tolerances.
        # KNOWN ISSUE: when both callables raise, the exception types are not compared.
        try:
            result_e = expected()
            exception_e = None
        except Exception as e:
            result_e = None
            exception_e = e

        try:
            result_a = actual()
            exception_a = None
        except Exception as e:
            result_a = None
            exception_a = e

        if (exception_e is None) != (exception_a is None):
            if exception_a is not None and exception_e is None:
                raise exception_a
            self.assertIs(
                type(exception_e),
                type(exception_a),
                lambda msg: f"{msg}\n{exception_e=}\n{exception_a=}",
            )

        if exception_e is None:
            flattened_e, spec_e = tree_flatten(result_e)
            flattened_a, spec_a = tree_flatten(result_a)

            self.assertEqual(
                spec_e,
                spec_a,
                "Both functions must return a result with the same tree structure.",
            )
            for value_e, value_a in zip(flattened_e, flattened_a, strict=True):
                value_e = _as_interleaved(_as_local(value_e))
                value_a = _as_interleaved(_as_local(value_a))

                self.assertEqual(value_e, value_a, *args, **kwargs)

    def conditional_skip(self, desc: Descriptor) -> None:
        # [FUNCTION DOC] Skips the running test (skipTest with the reason) if any entry of
        # the SKIPS table matches the descriptor. Lazily imports the table from
        # test_complex_tensor.
        try:
            from .test_complex_tensor import SKIPS
        except ImportError:
            from test_complex_tensor import SKIPS
        for xfail_info, reason in SKIPS.items():
            if xfail_info.matches(desc):
                self.skipTest(reason)

    def get_extra_kwargs(self, desc: Descriptor) -> dict[str, Any]:
        # [FUNCTION DOC] Returns the comparison kwargs (rtol and atol mapping) for a
        # descriptor: the first matching EXTRA_KWARGS entry, else a looser tolerance for
        # binary nonlinear ops on cuda in the op or distributed variants, else an empty
        # mapping.
        try:
            from .test_complex_tensor import EXTRA_KWARGS
        except ImportError:
            from test_complex_tensor import EXTRA_KWARGS

        for extra_info, extra_kw in EXTRA_KWARGS.items():
            if extra_info.matches(desc):
                return extra_kw

        if (
            desc.device_type == "cuda"
            and desc.variant in {Variant.Op, Variant.Distributed}
            and is_binary_nonlinear_op(desc.op)
        ):
            return BINARY_NONLINEAR_CUDA_TOL

        return {}

    def check_consistency(
        self, device: str, dtype: torch.dtype, op: OpInfo, variant: Variant
    ) -> None:
        # [FUNCTION DOC] Compares ComplexTensor and native complex execution of an op for
        # all sample inputs. variant must be Op or Distributed, else AssertionError.
        # Builds a descriptor, applies skips and tolerances, converts each sample with the
        # variant's transform, and asserts that both calls give the same result or error.
        # Returns nothing.
        if variant not in {Variant.Op, Variant.Distributed}:
            raise AssertionError(
                f"`check_consistency` called with the wrong `variant`: {variant}"
            )
        desc = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device_type=torch.device(device).type,
            dtype=dtype,
            variant=variant,
        )
        self.conditional_skip(desc)
        kwargs = self.get_extra_kwargs(desc)

        transform_fn = TRANSFORM_FUNCS[variant]

        sample_inputs = op.sample_inputs(device, dtype)

        for sample_input in sample_inputs:

            def expected(sample_input=sample_input):
                # [FUNCTION DOC] HELPER: Runs the op on the native sample input (captured by
                # default argument).
                return op(
                    sample_input.input,
                    *sample_input.args,
                    **sample_input.kwargs,
                )

            subclass_sample = sample_input.transform(transform_fn)

            def actual(subclass_sample=subclass_sample):
                # [FUNCTION DOC] HELPER: Runs the op on the transformed sample (captured by
                # default argument).
                return op(
                    subclass_sample.input,
                    *subclass_sample.args,
                    **subclass_sample.kwargs,
                )

            self.assertSameResult(expected, actual, **kwargs)

    def check_grad(self, device: str, dtype: torch.dtype, op: OpInfo) -> None:
        # [FUNCTION DOC] Gradient-checks an op with ComplexTensor inputs. Applies skips
        # for the grad variant, generates samples requiring grad, converts complex tensors
        # to ComplexTensor (without the no-grad transform, to keep requires_grad), builds
        # a wrapper function (list-input ops are handled separately) and runs gradcheck
        # with batched grad and forward AD, raising on failure.
        desc = Descriptor(
            op=get_overload_packet_from_name(op.name),
            device_type=torch.device(device).type,
            dtype=dtype,
            variant=Variant.GradCheck,
        )
        self.conditional_skip(desc)

        sample_inputs = op.sample_inputs(device, dtype, requires_grad=True)

        for sample_input in sample_inputs:
            # We don't use `sample_input.transform` here as it
            # uses a `no_grad` context which makes one lose the
            # `requires_grad` flag.
            args, kwargs = (sample_input.input, *sample_input.args), sample_input.kwargs
            args, kwargs = tree_map(
                _as_complex_tensor,
                (args, kwargs),
                is_leaf=lambda x: isinstance(x, torch.Tensor) and x.dtype.is_complex,
            )

            if isinstance(args[0], torch.Tensor):
                input_list = list(args)

                def func(*t: torch.Tensor):
                    # [FUNCTION DOC] HELPER: Calls the op on the positional tensors with the
                    # captured keyword args.
                    return op(*t, **kwargs)
            else:
                # For ops like `stack` and `cat`, we need to specify
                # the list of inputs a bit differently
                input_list = list(args[0])

                def func(*t: torch.Tensor):
                    # [FUNCTION DOC] HELPER: Calls the op with the tensors regrouped as a list
                    # first argument (for ops like stack and cat) plus the remaining captured
                    # positional and keyword args.
                    return op(list(t), *args[1:], **kwargs)

            # Do the actual gradcheck
            self.assertTrue(
                gradcheck(
                    func,
                    input_list,
                    raise_exception=True,
                    fast_mode=True,
                    check_batched_grad=True,
                    check_forward_ad=True,
                )
            )


aten = torch.ops.aten

complex_op_db = tuple(
    filter(lambda op: any(op.supports_dtype(ct, "cpu") for ct in COMPLEX_DTYPES), op_db)
)


def get_overload_packet_from_name(name: str) -> OpOverloadPacket:
    # [FUNCTION DOC] Returns the first op overload packet named name found by scanning
    # all torch.ops namespaces. Raises RuntimeError if no namespace has it.
    for domain_name in torch.ops:
        op_namespace = getattr(torch.ops, domain_name)
        op: OpOverloadPacket | None = getattr(op_namespace, name, None)
        if op is not None:
            return op

    raise RuntimeError(f"No op with {name=} found.")


force_test_names = set(map(_get_op_name, FORCE_TEST_LIST))
implemented_op_names = (
    set(map(_get_op_name, COMPLEX_OPS_TABLE.keys())) - force_test_names
)
implemented_op_db = tuple(
    filter(lambda op: op.name in implemented_op_names, complex_op_db)
)
force_test_op_db = tuple(filter(lambda op: op.name in force_test_names, op_db))

tested_op_names = {op.name for op in implemented_op_db} | {
    op.name for op in force_test_op_db
}
non_tested_ops = {
    op for op in COMPLEX_OPS_TABLE if _get_op_name(op) not in tested_op_names
}


# TODO (hameerabbasi): There are a number of ops that don't have any associated
# OpInfos. We still need to write tests for those ops.
if len(non_tested_ops) != 0:
    import textwrap
    import warnings

    list_missing_ops = "\n".join(sorted([str(op) for op in non_tested_ops]))
    warnings.warn(
        "Not all implemented ops are tested. List of ops missing tests:"
        f"\n{textwrap.indent(list_missing_ops, '    ')}",
        UserWarning,
        stacklevel=2,
    )
