# [MODULE DOC]
# Generates the C++ declarations that go into NativeFunctions.h: kernel declarations for
# unstructured functions and structured kernel classes deriving from the meta classes.
# No import-time side effects.
from __future__ import annotations

import torchgen.api.meta as meta
import torchgen.api.structured as structured
from torchgen.api.types import kernel_signature
from torchgen.context import with_native_function_and_index
from torchgen.model import BackendIndex, NativeFunction, NativeFunctionsGroup
from torchgen.utils import mapMaybe


@with_native_function_and_index
def gen_unstructured(f: NativeFunction, backend_index: BackendIndex) -> str | None:
    # [FUNCTION DOC] HELPER: Returns a C++ kernel declaration line for the function using
    # the kernel name from the backend index (static for external backends, else TORCH_API),
    # or None if no kernel is registered or the kernel is a legacy one.
    sig = kernel_signature(f, backend_index)
    metadata = backend_index.get_kernel(f)
    if metadata is None:
        return None
    if "legacy::" in metadata.kernel:
        return None
    else:
        prefix = "static" if backend_index.external else "TORCH_API"
        return f"{prefix} {sig.decl(name=metadata.kernel)};"


@with_native_function_and_index
def gen_structured(g: NativeFunctionsGroup, backend_index: BackendIndex) -> list[str]:
    # [FUNCTION DOC] HELPER: Returns a one-element list with the C++ declaration of a
    # structured kernel struct deriving from the meta class and declaring the impl method
    # with the structured impl arguments; returns an empty list if no kernel is registered.
    meta_name = meta.name(g)
    out_args = structured.impl_arguments(g)
    metadata = backend_index.get_kernel(g)
    if metadata is None:
        return []
    prefix = "" if backend_index.external else "TORCH_API "
    return [
        f"""\
struct {prefix}structured_{metadata.kernel} : public at::meta::structured_{meta_name} {{
void impl({", ".join(a.decl() for a in out_args)});
}};
"""
    ]


# Generates NativeFunctions.h, a list of forward declarations of all
# actual kernel definitions we keep in aten/src/ATen/native/
@with_native_function_and_index
def compute_native_function_declaration(
    g: NativeFunctionsGroup | NativeFunction, backend_index: BackendIndex
) -> list[str]:
    # [FUNCTION DOC]
    # Returns the C++ declaration strings for a function or group on a backend: structured
    # kernel structs when the group's kernel is structured, otherwise one declaration per
    # function with a kernel (possibly empty).
    # Raises an error for a structured kernel on an external backend (unsupported).
    metadata = backend_index.get_kernel(g)
    if isinstance(g, NativeFunctionsGroup):
        if metadata is not None and metadata.structured:
            if backend_index.external:
                # Structured hasn't been tested with external backends yet.
                raise AssertionError(
                    "Structured external backend functions are not implemented yet."
                )
            else:
                return gen_structured(g, backend_index)
        else:
            return list(
                mapMaybe(lambda f: gen_unstructured(f, backend_index), g.functions())
            )
    else:
        x = gen_unstructured(g, backend_index)
        return [] if x is None else [x]
