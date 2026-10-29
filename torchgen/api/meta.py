# [MODULE DOC] Naming helper for structured-kernel meta functions. Meta functions follow the
# dispatcher calling convention but take no mutable arguments and return tensor metadata
# rather than tensors. Only a name computation is provided; no import-time side effects.
from torchgen.model import NativeFunctionsGroup


# Follows dispatcher calling convention, but:
#   - Mutable arguments not allowed.  Meta functions are always
#     written in functional form.  Look at FunctionSchema.signature()
#   - No tensor returns; instead we return a TensorMeta describing
#     the tensor in question


def name(g: NativeFunctionsGroup) -> str:
    # [FUNCTION DOC] Returns the meta function name for a native functions group: the full
    # operator name (with overload) of the functional variant, with dots replaced by
    # underscores. Pure; no side effects.
    # use the overload name from the functional version
    return str(g.functional.func.name).replace(".", "_")
