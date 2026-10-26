# [MODULE DOC] Deprecated functorch.* entry points. Each function emits a FutureWarning and
# forwards to its torch.func / torch equivalent (vmap, grad, grad_and_value, vjp, jvp, jacrev,
# jacfwd, hessian, functionalize, make_functional, make_functional_with_buffers,
# combine_state_for_ensemble). Import-time side effect: setup_docs is called for each wrapper to
# copy the new API's docstring and append a deprecation warning note.
"""
The APIs in this file are exposed as `functorch.*`. They are thin wrappers
around the torch.func.* APIs that have deprecation warnings -- we're trying
to move people to the torch.func.* equivalents.

NB: We don't use *args, **kwargs in the signatures because that changes the
documentation.
"""

from __future__ import annotations

import textwrap
import warnings
from typing import Any, TYPE_CHECKING

import torch._functorch.apis as apis
import torch._functorch.eager_transforms as _impl
import torch._functorch.make_functional as _nn_impl
import torch.nn as nn


if TYPE_CHECKING:
    from collections.abc import Callable

    from torch._functorch.eager_transforms import argnums_t
    from torch._functorch.vmap import in_dims_t, out_dims_t


def get_warning(
    api: str, new_api: str | None = None, replace_newlines: bool = False
) -> str:
    # [FUNCTION DOC] Builds the deprecation message for functorch.api naming the replacement
    # (default torch.func.api). With replace_newlines, newline characters are removed (the
    # trailing spaces keep words separated). Returns the string.
    if new_api is None:
        new_api = f"torch.func.{api}"
    warning = (
        f"We've integrated functorch into PyTorch. As the final step of the \n"
        f"integration, `functorch.{api}` is deprecated as of PyTorch \n"
        f"2.0 and will be deleted in a future version of PyTorch >= 2.3. \n"
        f"Please use `{new_api}` instead; see the PyTorch 2.0 release notes \n"
        f"and/or the `torch.func` migration guide for more details \n"
        f"https://pytorch.org/docs/main/func.migrating.html"
    )
    if replace_newlines:
        warning = warning.replace("\n", "")
    return warning


def warn_deprecated(api: str, new_api: str | None = None) -> None:
    # [FUNCTION DOC] Emits a FutureWarning with the single-line deprecation message; stacklevel
    # 3 attributes it to the caller of the deprecated wrapper.
    warning = get_warning(api, new_api, replace_newlines=True)
    warnings.warn(warning, FutureWarning, stacklevel=3)


def setup_docs(
    functorch_api: Callable[..., Any],
    torch_func_api: Callable[..., Any] | None = None,
    new_api_name: str | None = None,
) -> None:
    # [FUNCTION DOC] Sets functorch_api's docstring to the docstring of the torch.func equivalent
    # (looked up by name in eager_transforms if not given) plus an indented deprecation warning
    # block. Does nothing if the source docstring is missing (python -OO). Mutates __doc__.
    api_name = functorch_api.__name__
    if torch_func_api is None:
        torch_func_api = getattr(_impl, api_name)
    # See https://docs.python.org/3/using/cmdline.html#cmdoption-OO
    if torch_func_api.__doc__ is None:
        return

    warning = get_warning(api_name, new_api_name)
    warning_note = "\n.. warning::\n\n" + textwrap.indent(warning, "    ")
    warning_note = textwrap.indent(warning_note, "    ")
    functorch_api.__doc__ = torch_func_api.__doc__ + warning_note


def vmap(
    func: Callable[..., Any],
    in_dims: in_dims_t = 0,
    out_dims: out_dims_t = 0,
    randomness: str = "error",
    *,
    chunk_size: int | None = None,
) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns apis.vmap(func, in_dims, out_dims, randomness, chunk_size).
    warn_deprecated("vmap", "torch.vmap")
    return apis.vmap(func, in_dims, out_dims, randomness, chunk_size=chunk_size)


def grad(
    func: Callable[..., Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns apis.grad(func, argnums, has_aux).
    warn_deprecated("grad")
    return apis.grad(func, argnums, has_aux)


def grad_and_value(
    func: Callable[..., Any], argnums: argnums_t = 0, has_aux: bool = False
) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns apis.grad_and_value(func, argnums, has_aux).
    warn_deprecated("grad_and_value")
    return apis.grad_and_value(func, argnums, has_aux)


def vjp(func: Callable[..., Any], *primals: Any, has_aux: bool = False) -> Any:
    # [FUNCTION DOC] Warns of deprecation and returns the result of eager_transforms.vjp on func and primals.
    warn_deprecated("vjp")
    return _impl.vjp(func, *primals, has_aux=has_aux)


def jvp(
    func: Callable[..., Any],
    primals: Any,
    tangents: Any,
    *,
    strict: bool = False,
    has_aux: bool = False,
) -> Any:
    # [FUNCTION DOC] Warns of deprecation and returns eager_transforms.jvp(func, primals, tangents, strict, has_aux).
    warn_deprecated("jvp")
    return _impl.jvp(func, primals, tangents, strict=strict, has_aux=has_aux)


def jacrev(
    func: Callable[..., Any],
    argnums: int | tuple[int, ...] = 0,
    *,
    has_aux: bool = False,
    chunk_size: int | None = None,
    _preallocate_and_copy: bool = False,
) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns eager_transforms.jacrev with all arguments
    # forwarded (argnums, has_aux, chunk_size, _preallocate_and_copy).
    warn_deprecated("jacrev")
    return _impl.jacrev(
        func,
        argnums,
        has_aux=has_aux,
        chunk_size=chunk_size,
        _preallocate_and_copy=_preallocate_and_copy,
    )


def jacfwd(
    func: Callable[..., Any],
    argnums: argnums_t = 0,
    has_aux: bool = False,
    *,
    randomness: str = "error",
) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns eager_transforms.jacfwd(func, argnums, has_aux, randomness).
    warn_deprecated("jacfwd")
    return _impl.jacfwd(func, argnums, has_aux, randomness=randomness)


def hessian(func: Callable[..., Any], argnums: int = 0) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns eager_transforms.hessian(func, argnums).
    warn_deprecated("hessian")
    return _impl.hessian(func, argnums=argnums)


def functionalize(
    func: Callable[..., Any], *, remove: str = "mutations"
) -> Callable[..., Any]:
    # [FUNCTION DOC] Warns of deprecation and returns eager_transforms.functionalize(func, remove).
    warn_deprecated("functionalize")
    return _impl.functionalize(func, remove=remove)


def make_functional(model: nn.Module, disable_autograd_tracking: bool = False) -> Any:
    # [FUNCTION DOC] Warns of deprecation (suggesting functional_call) and returns make_functional(model, disable_autograd_tracking).
    warn_deprecated("make_functional","torch.func.functional_call")
    return _nn_impl.make_functional(model, disable_autograd_tracking)


def make_functional_with_buffers(
    model: nn.Module, disable_autograd_tracking: bool = False
) -> Any:
    # [FUNCTION DOC] Warns of deprecation (suggesting functional_call) and returns make_functional_with_buffers(model, disable_autograd_tracking).
    warn_deprecated("make_functional_with_buffers", "torch.func.functional_call")
    return _nn_impl.make_functional_with_buffers(model, disable_autograd_tracking)


def combine_state_for_ensemble(models: list[nn.Module]) -> Any:
    # [FUNCTION DOC] Warns of deprecation (suggesting stack_module_state) and returns combine_state_for_ensemble(models).
    warn_deprecated("combine_state_for_ensemble", "torch.func.stack_module_state")
    return _nn_impl.combine_state_for_ensemble(models)


setup_docs(vmap, apis.vmap, "torch.vmap")
setup_docs(grad, apis.grad)
setup_docs(grad_and_value, apis.grad_and_value)
setup_docs(vjp)
setup_docs(jvp)
setup_docs(jacrev)
setup_docs(jacfwd)
setup_docs(hessian)
setup_docs(functionalize)
setup_docs(make_functional, _nn_impl.make_functional, "torch.func.functional_call")
setup_docs(
    make_functional_with_buffers, _nn_impl.make_functional, "torch.func.functional_call"
)
setup_docs(
    combine_state_for_ensemble,
    _nn_impl.combine_state_for_ensemble,
    "torch.func.stack_module_state",
)
