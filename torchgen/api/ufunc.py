# [MODULE DOC] Argument and type mappings for codegen of structured ufunc kernels: the
# dispatch stub signature (CPU), the CUDA ufunctor constructor/apply split, and the inner
# ufunc template function. Provides Binding builders for each, driven by a
# NativeFunctionsGroup. No import-time side effects.
from __future__ import annotations

from dataclasses import dataclass

import torchgen.api.types as api_types
from torchgen.api import cpp, structured
from torchgen.api.types import (
    ArgName,
    BaseCppType,
    BaseCType,
    Binding,
    ConstRefCType,
    CType,
    NamedCType,
    scalarT,
)
from torchgen.model import (
    Argument,
    BaseTy,
    BaseType,
    DispatchKey,
    FunctionSchema,
    NativeFunctionsGroup,
    Type,
)


def schema_kernel_name(func: FunctionSchema, dispatch_key: DispatchKey) -> str:
    # [FUNCTION DOC] Name of the generated ufunc kernel for an out schema and dispatch key, of the
    # form ufunc_<base name>_<dispatch key>. Raises AssertionError if func is not an out schema.
    if not func.is_out_fn():
        raise AssertionError("ufunc.kernel_name should only be invoked on out schemas")
    return f"ufunc_{func.name.name}_{dispatch_key}"


def kernel_name(g: NativeFunctionsGroup, dispatch_key: DispatchKey) -> str:
    # [FUNCTION DOC] Ufunc kernel name for the out variant of group g and the dispatch key.
    return schema_kernel_name(g.out.func, dispatch_key)


# Tensors are omitted (as they are stored in TensorIterator), everything else is
# passed along  (technically, we can pass tensors along too, it just wastes
# argument registers)
#
# NB: used for CPU only
def dispatchstub_type(t: Type, *, binds: ArgName) -> NamedCType | None:
    # [FUNCTION DOC] C++ type of an argument in a CPU dispatch stub signature. Value types map as
    # in cpp.valuetype_type (no symint), Scalar becomes a const reference, and Tensor yields None
    # because tensors are carried by the TensorIterator. Raises AssertionError on other types.
    # Dispatch stubs are always plain ints
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r

    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, ConstRefCType(BaseCType(scalarT)))
    elif t == BaseType(BaseTy.Tensor):
        return None
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


def opmath_type(scalar_t: BaseCppType) -> BaseCppType:
    # [FUNCTION DOC] HELPER: Map the scalar_t C++ type to opmath_t. Raises NotImplementedError
    # for any other input.
    if scalar_t == api_types.scalar_t:
        return api_types.opmath_t
    raise NotImplementedError


# NB: Tensors in constructor are stored in opmath_t, not scalar_t
# because Tensor in constructor = its a scalar tensor partially applied =
# it can be higher precision and we want to compute in that higher precision
#
# NB: CUDA only
def ufunctor_ctor_type(t: Type, *, binds: ArgName, scalar_t: BaseCppType) -> NamedCType:
    # [FUNCTION DOC] C++ type of an argument stored in a CUDA ufunctor constructor. Value types
    # map as in cpp.valuetype_type; both Scalar and Tensor become the opmath type for scalar_t.
    # Raises AssertionError on other types.
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r

    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(opmath_type(scalar_t)))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# Only Tensors ever get passed directly to operator()
#
# NB: CUDA only
# (Actually, this works for CPU too)
def ufunctor_apply_type(
    t: Type, *, binds: ArgName, scalar_t: BaseCppType
) -> NamedCType:
    # [FUNCTION DOC] C++ type of an argument passed to the ufunctor operator(): only Tensor is
    # accepted, becoming scalar_t. Raises AssertionError for any other type.
    if t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, BaseCType(scalar_t))
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


# The actual ufunc template function the user writes.  Everything here
# is done in the computation type.  compute_t is opmath_t in CUDA and scalar_t
# in CPU
def ufunc_type(t: Type, *, binds: ArgName, compute_t: CType) -> NamedCType:
    # [FUNCTION DOC] C++ type of an argument of the user-written ufunc template function. Value
    # types map as in cpp.valuetype_type; Scalar and Tensor both use compute_t.
    # Raises AssertionError on other types.
    r = cpp.valuetype_type(t, binds=binds, symint=False)
    if r is not None:
        return r

    if t == BaseType(BaseTy.Scalar):
        return NamedCType(binds, compute_t)
    elif t == BaseType(BaseTy.Tensor):
        return NamedCType(binds, compute_t)
    else:
        raise AssertionError(f"unrecognized type {repr(t)}")


def ufunctor_ctor_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    # [FUNCTION DOC] HELPER: Build a Binding (no default) for argument a using its ufunctor
    # constructor type.
    return Binding(
        nctype=ufunctor_ctor_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )


def ufunctor_apply_argument(a: Argument, scalar_t: BaseCppType) -> Binding:
    # [FUNCTION DOC] HELPER: Build a Binding (no default) for argument a using its ufunctor
    # operator() parameter type.
    return Binding(
        nctype=ufunctor_apply_type(a.type, binds=a.name, scalar_t=scalar_t),
        name=a.name,
        default=None,
        argument=a,
    )


def ufunc_argument(a: Argument, compute_t: CType) -> Binding:
    # [FUNCTION DOC] HELPER: Build a Binding (no default) for argument a using its ufunc type.
    return Binding(
        nctype=ufunc_type(a.type, binds=a.name, compute_t=compute_t),
        name=a.name,
        default=None,
        argument=a,
    )


@dataclass(frozen=True)
class UfunctorBindings:
    # [CLASS DOC] Immutable pair of binding lists for a CUDA ufunctor: ctor holds arguments taken
    # by the host-side constructor, apply holds those taken by the device-side operator().
    ctor: list[Binding]
    apply: list[Binding]


# ufunctors are a CUDA-only concept representing functors that take some of
# their arguments on a host-side constructor, and the rest in the device-side
# apply.  E.g.,
#
# template <typename scalar_t>
# struct CUDAFunctorOnSelf_add {
#   using opmath_t = at::opmath_type<scalar_t>;
#   opmath_t other_;
#   opmath_t alpha_;
#   CUDAFunctorOnSelf_add(opmath_t other, opmath_t alpha) : other_(other), alpha_(alpha) {}
#   __device__ scalar_t operator()(scalar_t self) {
#     return ufunc::add(static_cast<opmath_t>(self), other_, alpha_);
#   }
# };
#
# The ctor refers to the constructor CUDAFunctorOnSelf_add, while apply refers
# to the operator() definition
def ufunctor_arguments(
    g: NativeFunctionsGroup, *, scalar_tensor_idx: int | None, scalar_t: BaseCppType
) -> UfunctorBindings:
    # [FUNCTION DOC] Split the non-out arguments of the functional variant of g into constructor
    # and apply bindings. Non-tensor arguments go to the constructor; tensor arguments go to
    # apply, except that the tensor at position scalar_tensor_idx (counted among tensors) is
    # treated as a scalar and put in the constructor. Returns UfunctorBindings. Raises
    # AssertionError if scalar_tensor_idx is not consumed (index too large).
    ctor = []
    apply = []
    for a in g.functional.func.arguments.flat_non_out:
        if a.type.is_tensor_like():
            if scalar_tensor_idx == 0:
                # put it in the ctor anyway
                ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
                scalar_tensor_idx = None
            else:
                if scalar_tensor_idx is not None:
                    scalar_tensor_idx -= 1
                apply.append(ufunctor_apply_argument(a, scalar_t=scalar_t))
        else:
            ctor.append(ufunctor_ctor_argument(a, scalar_t=scalar_t))
    if scalar_tensor_idx is not None:
        raise AssertionError("scalar_tensor_idx should be None at end of processing")
    return UfunctorBindings(ctor=ctor, apply=apply)


# ufuncs are the inner loop template functions that you wrote in ufunc/add.h
# which do the actual computation in question.  E.g.,
#
# template <typename T>
# C10_HOST_DEVICE T add(T self, T other, T alpha) __ubsan_ignore_undefined__ {
#   return self + alpha * other;
# }
#
# In this file, we refer to T as compute_t which is bound by caller
def ufunc_arguments(g: NativeFunctionsGroup, *, compute_t: CType) -> list[Binding]:
    # [FUNCTION DOC] Bindings for the parameters of the ufunc template function: every non-out
    # argument of the functional variant of g, typed with compute_t for tensors and scalars.
    return [
        ufunc_argument(a, compute_t=compute_t)
        for a in g.functional.func.arguments.flat_non_out
    ]


# Stubs are the DispatchStub trampolines that CPU kernels use to get to their
# vectorized versions.  E.g.,
#
# using structured_binary_fn_alpha = void(*)(TensorIteratorBase&, const Scalar& alpha);
# DECLARE_DISPATCH(structured_binary_fn_alpha, add_stub);
def stub_arguments(g: NativeFunctionsGroup) -> list[Binding]:
    # [FUNCTION DOC] Bindings for the dispatch stub extra arguments: the non-out arguments of the
    # out variant of g that are not tensor-like, typed through structured.argument.
    # stubs drop all tensor arguments (they are implicit in the TensorIterator
    # argument and keep everything else)
    return [
        r
        for a in g.out.func.arguments.flat_non_out
        if not a.type.is_tensor_like()
        for r in structured.argument(a)
    ]
