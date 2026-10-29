# mypy: allow-untyped-defs
# [MODULE DOC] Legacy hook-based weight normalization (arxiv 1602.07868).
# Defines WeightNorm, a forward pre-hook that rebuilds a weight from magnitude (name_g) and
# direction (name_v) parameters, plus the public weight_norm and remove_weight_norm
# functions. Deprecated in favor of torch.nn.utils.parametrizations.weight_norm.
# No import-time side effects.
r"""Weight Normalization from https://arxiv.org/abs/1602.07868."""

from typing import Any, TypeVar
from typing_extensions import deprecated

from torch import _weight_norm, norm_except_dim
from torch.nn.modules import Module
from torch.nn.parameter import Parameter, UninitializedParameter


__all__ = ["WeightNorm", "weight_norm", "remove_weight_norm"]


class WeightNorm:
    # [CLASS DOC] Forward pre-hook implementing weight normalization of one named parameter.
    # Attributes: name (parameter name) and dim (dimension kept when computing norms; -1 means
    # the whole tensor). Applying it replaces the parameter by name_g and name_v parameters
    # and recomputes the plain weight attribute before every forward call.
    name: str
    dim: int

    def __init__(self, name: str, dim: int) -> None:
        # [FUNCTION DOC] Store name and dim; a dim of None is converted to -1 (whole-tensor norm).
        if dim is None:
            dim = -1
        self.name = name
        self.dim = dim

    # TODO Make return type more specific
    def compute_weight(self, module: Module) -> Any:
        # [FUNCTION DOC] Compute the weight from the module's name_g and name_v parameters.
        # Returns the normalized-direction times magnitude tensor via the native weight norm op.
        g = getattr(module, self.name + "_g")
        v = getattr(module, self.name + "_v")
        return _weight_norm(v, g, self.dim)

    @staticmethod
    @deprecated(
        "`torch.nn.utils.weight_norm` is deprecated "
        "in favor of `torch.nn.utils.parametrizations.weight_norm`.",
        category=FutureWarning,
    )
    def apply(module, name: str, dim: int) -> "WeightNorm":
        # [FUNCTION DOC] Install weight normalization on a module parameter and return the hook.
        # Deprecated (emits FutureWarning). Raises RuntimeError if a WeightNorm hook already exists
        # for the name, ValueError for an uninitialized lazy parameter. Side effects: deletes the
        # original parameter, registers name_g (norm over all dims except dim) and name_v,
        # sets the computed weight as a plain attribute, and registers the forward pre-hook.
        # A dim of None is treated as -1.
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, WeightNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two weight_norm hooks on the same parameter {name}"
                )

        if dim is None:
            dim = -1

        fn = WeightNorm(name, dim)

        weight = getattr(module, name)
        if isinstance(weight, UninitializedParameter):
            raise ValueError(
                "The module passed to `WeightNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying weight normalization"
            )
        # remove w from parameter list
        del module._parameters[name]

        # add g and v as new parameters and express w as g/||v|| * v
        module.register_parameter(
            name + "_g", Parameter(norm_except_dim(weight, 2, dim).data)
        )
        module.register_parameter(name + "_v", Parameter(weight.data))
        setattr(module, name, fn.compute_weight(module))

        # recompute weight before every forward()
        module.register_forward_pre_hook(fn)

        return fn

    def remove(self, module: Module) -> None:
        # [FUNCTION DOC] Undo weight normalization on a module.
        # Computes the current weight, deletes the attribute and the g and v parameters, and
        # registers a new Parameter holding the detached weight data under the original name.
        weight = self.compute_weight(module)
        delattr(module, self.name)
        del module._parameters[self.name + "_g"]
        del module._parameters[self.name + "_v"]
        setattr(module, self.name, Parameter(weight.data))

    def __call__(self, module: Module, inputs: Any) -> None:
        # [FUNCTION DOC] Forward pre-hook entry point: recompute and assign the weight.
        setattr(module, self.name, self.compute_weight(module))


T_module = TypeVar("T_module", bound=Module)


def weight_norm(module: T_module, name: str = "weight", dim: int = 0) -> T_module:
    # [FUNCTION DOC] Apply weight normalization to a parameter of the module and return the module.
    # Mutates the module in place via WeightNorm.apply (so warns about deprecation and may
    # raise the errors described there).
    r"""Apply weight normalization to a parameter in the given module.

    .. math::
         \mathbf{w} = g \dfrac{\mathbf{v}}{\|\mathbf{v}\|}

    Weight normalization is a reparameterization that decouples the magnitude
    of a weight tensor from its direction. This replaces the parameter specified
    by :attr:`name` (e.g. ``'weight'``) with two parameters: one specifying the magnitude
    (e.g. ``'weight_g'``) and one specifying the direction (e.g. ``'weight_v'``).
    Weight normalization is implemented via a hook that recomputes the weight
    tensor from the magnitude and direction before every :meth:`~Module.forward`
    call.

    By default, with ``dim=0``, the norm is computed independently per output
    channel/plane. To compute a norm over the entire weight tensor, use
    ``dim=None``.

    See https://arxiv.org/abs/1602.07868

    .. warning::

        This function is deprecated.  Use :func:`torch.nn.utils.parametrizations.weight_norm`
        which uses the modern parametrization API.  The new ``weight_norm`` is compatible
        with ``state_dict`` generated from old ``weight_norm``.

        Migration guide:

        * The magnitude (``weight_g``) and direction (``weight_v``) are now expressed
          as ``parametrizations.weight.original0`` and ``parametrizations.weight.original1``
          respectively.  If this is bothering you, please comment on
          https://github.com/pytorch/pytorch/issues/102999

        * To remove the weight normalization reparameterization, use
          :func:`torch.nn.utils.parametrize.remove_parametrizations`.

        * The weight is no longer recomputed once at module forward; instead, it will
          be recomputed on every access.  To restore the old behavior, use
          :func:`torch.nn.utils.parametrize.cached` before invoking the module
          in question.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter
        dim (int, optional): dimension over which to compute the norm

    Returns:
        The original module with the weight norm hook

    Example::

        >>> warnings.filterwarnings("ignore", message=".*torch.nn.utils.weight_norm")  # docs: hide
        >>> m = weight_norm(nn.Linear(20, 40), name='weight')
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_g.size()
        torch.Size([40, 1])
        >>> m.weight_v.size()
        torch.Size([40, 20])

    """
    WeightNorm.apply(module, name, dim)
    return module


def remove_weight_norm(module: T_module, name: str = "weight") -> T_module:
    # [FUNCTION DOC] Remove weight normalization from the named parameter and return the module.
    # Finds the matching forward pre-hook, restores a plain parameter and deletes the hook.
    # Raises ValueError if no weight norm hook exists for the given name.
    r"""Remove the weight normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> warnings.filterwarnings(
        ...     "ignore", message=".*torch.nn.utils.weight_norm"
        ... )  # docs: hide
        >>> m = weight_norm(nn.Linear(20, 40))
        >>> remove_weight_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, WeightNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(f"weight_norm of '{name}' not found in {module}")
