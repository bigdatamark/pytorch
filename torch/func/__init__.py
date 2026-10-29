# [PACKAGE DOC]
# torch.func package: public functional-transform API (JAX-like). Purely a
# re-export layer over the private functorch implementation modules.
# Exports: grad, grad_and_value, vmap, vjp, jvp, linearize, jacrev, jacfwd,
# hessian, functionalize, functional_call, stack_module_state, rearrange,
# replace_all_batch_norm_modules_ and debug_unwrap. Defines no logic of its own;
# import-time side effects are only those of importing the functorch modules.
from torch._functorch.apis import grad, grad_and_value, vmap
from torch._functorch.batch_norm_replacement import replace_all_batch_norm_modules_
from torch._functorch.eager_transforms import (
    debug_unwrap,
    functionalize,
    hessian,
    jacfwd,
    jacrev,
    jvp,
    linearize,
    vjp,
)
from torch._functorch.einops import rearrange
from torch._functorch.functional_call import functional_call, stack_module_state


__all__ = [
    "grad",
    "grad_and_value",
    "vmap",
    "replace_all_batch_norm_modules_",
    "functionalize",
    "hessian",
    "jacfwd",
    "jacrev",
    "jvp",
    "linearize",
    "rearrange",
    "vjp",
    "functional_call",
    "stack_module_state",
    "debug_unwrap",
]
