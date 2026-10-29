# mypy: allow-untyped-defs
# [MODULE DOC] Legacy hook-based spectral normalization (arxiv 1802.05957) for module weights.
# Defines SpectralNorm (a forward pre-hook that rescales a weight by its estimated largest
# singular value using power iteration on persistent u and v buffers), two state_dict
# hook classes that version the checkpoint format and upgrade old checkpoints, and the
# user-facing spectral_norm / remove_spectral_norm functions. Superseded by
# torch.nn.utils.parametrizations.spectral_norm. No import-time side effects.
"""Spectral Normalization from https://arxiv.org/abs/1802.05957."""

from typing import Any, TypeVar

import torch
import torch.nn.functional as F
from torch.nn.modules import Module


__all__ = [
    "SpectralNorm",
    "SpectralNormLoadStateDictPreHook",
    "SpectralNormStateDictHook",
    "spectral_norm",
    "remove_spectral_norm",
]


class SpectralNorm:
    # [CLASS DOC] Forward pre-hook object implementing spectral normalization of one parameter.
    # Attributes: name (parameter name), dim (dimension treated as output rows when flattening
    # the weight to a matrix), n_power_iterations (positive int), eps (normalization epsilon),
    # class constant _version (checkpoint format version, currently 1). Applying it moves the
    # original parameter to name_orig and registers buffers name_u and name_v; the normalized
    # weight is a plain attribute recomputed before each forward. Invariant after a forward:
    # u is the normalized product of the weight matrix and v (not enforced at init).
    # Invariant before and after each forward call:
    #   u = F.normalize(W @ v)
    # NB: At initialization, this invariant is not enforced

    _version: int = 1
    # At version 1:
    #   made  `W` not a buffer,
    #   added `v` as a buffer, and
    #   made eval mode use `W = u @ W_orig @ v` rather than the stored `W`.
    name: str
    dim: int
    n_power_iterations: int
    eps: float

    def __init__(
        self,
        name: str = "weight",
        n_power_iterations: int = 1,
        dim: int = 0,
        eps: float = 1e-12,
    ) -> None:
        # [FUNCTION DOC] Store configuration. Raises ValueError if n_power_iterations <= 0.
        self.name = name
        self.dim = dim
        if n_power_iterations <= 0:
            raise ValueError(
                "Expected n_power_iterations to be positive, but "
                f"got n_power_iterations={n_power_iterations}"
            )
        self.n_power_iterations = n_power_iterations
        self.eps = eps

    def reshape_weight_to_matrix(self, weight: torch.Tensor) -> torch.Tensor:
        # [FUNCTION DOC] Flatten a weight tensor into a 2-D matrix whose rows follow dimension dim.
        # Moves dim to the front when it is not 0 and reshapes the rest into columns.
        weight_mat = weight
        if self.dim != 0:
            # permute dim to front
            weight_mat = weight_mat.permute(
                self.dim, *[d for d in range(weight_mat.dim()) if d != self.dim]
            )
        height = weight_mat.size(0)
        return weight_mat.reshape(height, -1)

    def compute_weight(self, module: Module, do_power_iteration: bool) -> torch.Tensor:
        # [FUNCTION DOC] Compute the spectrally normalized weight for a module.
        # Reads name_orig, name_u and name_v from the module. If do_power_iteration is true,
        # runs the configured number of power iterations without gradient, updating u and v
        # in place (needed so DataParallel replicas propagate updates), then clones them so
        # autograd can backpropagate through repeated forwards. Returns the original weight
        # divided by sigma, the dot of u with the weight matrix times v (differentiable).
        # NB: If `do_power_iteration` is set, the `u` and `v` vectors are
        #     updated in power iteration **in-place**. This is very important
        #     because in `DataParallel` forward, the vectors (being buffers) are
        #     broadcast from the parallelized module to each module replica,
        #     which is a new module object created on the fly. And each replica
        #     runs its own spectral norm power iteration. So simply assigning
        #     the updated vectors to the module this function runs on will cause
        #     the update to be lost forever. And the next time the parallelized
        #     module is replicated, the same randomly initialized vectors are
        #     broadcast and used!
        #
        #     Therefore, to make the change propagate back, we rely on two
        #     important behaviors (also enforced via tests):
        #       1. `DataParallel` doesn't clone storage if the broadcast tensor
        #          is already on correct device; and it makes sure that the
        #          parallelized module is already on `device[0]`.
        #       2. If the out tensor in `out=` kwarg has correct shape, it will
        #          just fill in the values.
        #     Therefore, since the same power iteration is performed on all
        #     devices, simply updating the tensors in-place will make sure that
        #     the module replica on `device[0]` will update the _u vector on the
        #     parallelized module (by shared storage).
        #
        #    However, after we update `u` and `v` in-place, we need to **clone**
        #    them before using them to normalize the weight. This is to support
        #    backpropagating through two forward passes, e.g., the common pattern in
        #    GAN training: loss = D(real) - D(fake). Otherwise, engine will
        #    complain that variables needed to do backward for the first forward
        #    (i.e., the `u` and `v` vectors) are changed in the second forward.
        weight = getattr(module, self.name + "_orig")
        u = getattr(module, self.name + "_u")
        v = getattr(module, self.name + "_v")
        weight_mat = self.reshape_weight_to_matrix(weight)

        if do_power_iteration:
            with torch.no_grad():
                for _ in range(self.n_power_iterations):
                    # Spectral norm of weight equals to `u^T W v`, where `u` and `v`
                    # are the first left and right singular vectors.
                    # This power iteration produces approximations of `u` and `v`.
                    v = F.normalize(
                        torch.mv(weight_mat.t(), u), dim=0, eps=self.eps, out=v
                    )
                    u = F.normalize(torch.mv(weight_mat, v), dim=0, eps=self.eps, out=u)
                if self.n_power_iterations > 0:
                    # See above on why we need to clone
                    u = u.clone(memory_format=torch.contiguous_format)
                    v = v.clone(memory_format=torch.contiguous_format)

        sigma = torch.dot(u, torch.mv(weight_mat, v))
        weight = weight / sigma
        return weight

    def remove(self, module: Module) -> None:
        # [FUNCTION DOC] Undo the reparametrization on a module.
        # Computes the normalized weight without power iteration, deletes the weight attribute and
        # the u, v and orig entries, and registers a fresh detached Parameter under the original
        # name. Does not itself unregister the hooks (remove_spectral_norm does that).
        with torch.no_grad():
            weight = self.compute_weight(module, do_power_iteration=False)
        delattr(module, self.name)
        delattr(module, self.name + "_u")
        delattr(module, self.name + "_v")
        delattr(module, self.name + "_orig")
        module.register_parameter(self.name, torch.nn.Parameter(weight.detach()))

    def __call__(self, module: Module, inputs: Any) -> None:
        # [FUNCTION DOC] Forward pre-hook entry point: recompute and assign the normalized weight.
        # Runs power iteration only when the module is in training mode. Returns None.
        setattr(
            module,
            self.name,
            self.compute_weight(module, do_power_iteration=module.training),
        )

    def _solve_v_and_rescale(self, weight_mat, u, target_sigma):
        # [FUNCTION DOC] HELPER: Solve for a v consistent with a given u and target singular value.
        # Uses a pseudo-inverse of the Gram matrix so it works when that matrix is singular, then
        # rescales (in place) so that u times the matrix times v equals target_sigma. Used when
        # upgrading version-less checkpoints.
        # Tries to return a vector `v` s.t. `u = F.normalize(W @ v)`
        # (the invariant at top of this class) and `u @ W @ v = sigma`.
        # This uses pinverse in case W^T W is not invertible.
        v = torch.linalg.multi_dot(
            [weight_mat.t().mm(weight_mat).pinverse(), weight_mat.t(), u.unsqueeze(1)]
        ).squeeze(1)
        return v.mul_(target_sigma / torch.dot(u, torch.mv(weight_mat, v)))

    @staticmethod
    def apply(
        module: Module, name: str, n_power_iterations: int, dim: int, eps: float
    ) -> "SpectralNorm":
        # [FUNCTION DOC] Install spectral normalization on a module parameter and return the hook.
        # Raises RuntimeError if the same parameter already has a SpectralNorm hook; raises
        # ValueError if the parameter is None or an uninitialized lazy parameter. Side effects:
        # removes the parameter, registers name_orig as a parameter, keeps the raw data under the
        # old name as a plain attribute, registers random normalized u and v buffers, and registers
        # the forward pre-hook plus the state_dict save and load-pre hooks.
        for hook in module._forward_pre_hooks.values():
            if isinstance(hook, SpectralNorm) and hook.name == name:
                raise RuntimeError(
                    f"Cannot register two spectral_norm hooks on the same parameter {name}"
                )

        fn = SpectralNorm(name, n_power_iterations, dim, eps)
        weight = module._parameters[name]
        if weight is None:
            raise ValueError(
                f"`SpectralNorm` cannot be applied as parameter `{name}` is None"
            )
        if isinstance(weight, torch.nn.parameter.UninitializedParameter):
            raise ValueError(
                "The module passed to `SpectralNorm` can't have uninitialized parameters. "
                "Make sure to run the dummy forward before applying spectral normalization"
            )

        with torch.no_grad():
            weight_mat = fn.reshape_weight_to_matrix(weight)

            h, w = weight_mat.size()
            # randomly initialize `u` and `v`
            u = F.normalize(weight.new_empty(h).normal_(0, 1), dim=0, eps=fn.eps)
            v = F.normalize(weight.new_empty(w).normal_(0, 1), dim=0, eps=fn.eps)

        delattr(module, fn.name)
        module.register_parameter(fn.name + "_orig", weight)
        # We still need to assign weight back as fn.name because all sorts of
        # things may assume that it exists, e.g., when initializing weights.
        # However, we can't directly assign as it could be an nn.Parameter and
        # gets added as a parameter. Instead, we register weight.data as a plain
        # attribute.
        setattr(module, fn.name, weight.data)
        module.register_buffer(fn.name + "_u", u)
        module.register_buffer(fn.name + "_v", v)

        module.register_forward_pre_hook(fn)
        module._register_state_dict_hook(SpectralNormStateDictHook(fn))
        module._register_load_state_dict_pre_hook(SpectralNormLoadStateDictPreHook(fn))
        return fn


class SpectralNormLoadStateDictPreHook:
# [CLASS DOC] Load-state-dict pre-hook that upgrades pre-version-1 spectral norm checkpoints.
# Holds the SpectralNorm instance (fn). Version-1 checkpoints pass through untouched.
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
    # [FUNCTION DOC] Store the SpectralNorm hook this load hook belongs to.
        self.fn = fn

    # For state_dict with version None, (assuming that it has gone through at
    # least one training forward), we have
    #
    #    u = F.normalize(W_orig @ v)
    #    W = W_orig / sigma, where sigma = u @ W_orig @ v
    #
    # To compute `v`, we solve `W_orig @ x = u`, and let
    #    v = x / (u @ W_orig @ x) * (W / W_orig).
    def __call__(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
    # [FUNCTION DOC] Convert an old-format state dict entry set to the version-1 layout in place.
    # Reads the version from the local metadata. For an old or missing version it skips the
    # work if the dict already looks new (orig, u and v keys present and no plain weight key).
    # Otherwise, if the orig, plain weight and u keys are all present, it removes the plain
    # weight entry, estimates sigma as the mean ratio of orig to weight, and stores a solved v
    # (missing keys are appended to missing_keys only when strict, then it returns early).
        fn = self.fn
        version = local_metadata.get("spectral_norm", {}).get(
            fn.name + ".version", None
        )
        if version is None or version < 1:
            weight_key = prefix + fn.name
            if (
                version is None
                and all(weight_key + s in state_dict for s in ("_orig", "_u", "_v"))
                and weight_key not in state_dict
            ):
                # Detect if it is the updated state dict and just missing metadata.
                # This could happen if the users are crafting a state dict themselves,
                # so we just pretend that this is the newest.
                return
            has_missing_keys = False
            for suffix in ("_orig", "", "_u"):
                key = weight_key + suffix
                if key not in state_dict:
                    has_missing_keys = True
                    if strict:
                        missing_keys.append(key)
            if has_missing_keys:
                return
            with torch.no_grad():
                weight_orig = state_dict[weight_key + "_orig"]
                weight = state_dict.pop(weight_key)
                sigma = (weight_orig / weight).mean()
                weight_mat = fn.reshape_weight_to_matrix(weight_orig)
                u = state_dict[weight_key + "_u"]
                v = fn._solve_v_and_rescale(weight_mat, u, sigma)
                state_dict[weight_key + "_v"] = v


class SpectralNormStateDictHook:
# [CLASS DOC] State-dict hook that records the SpectralNorm format version in metadata.
# Holds the SpectralNorm instance (fn).
    # See docstring of SpectralNorm._version on the changes to spectral_norm.
    def __init__(self, fn) -> None:
    # [FUNCTION DOC] Store the SpectralNorm hook this state-dict hook belongs to.
        self.fn = fn

    def __call__(self, module, state_dict, prefix, local_metadata) -> None:
    # [FUNCTION DOC] Write the format version under the spectral_norm metadata for this name.
    # Creates the metadata mapping if needed. Raises RuntimeError if the key already exists.
        if "spectral_norm" not in local_metadata:
            local_metadata["spectral_norm"] = {}
        key = self.fn.name + ".version"
        if key in local_metadata["spectral_norm"]:
            raise RuntimeError(f"Unexpected key in metadata['spectral_norm']: {key}")
        local_metadata["spectral_norm"][key] = self.fn._version


T_module = TypeVar("T_module", bound=Module)


def spectral_norm(
    module: T_module,
    name: str = "weight",
    n_power_iterations: int = 1,
    eps: float = 1e-12,
    dim: int | None = None,
) -> T_module:
    # [FUNCTION DOC] Apply spectral normalization to the named parameter and return the module.
    # If dim is None it defaults to 1 for transposed convolutions and 0 otherwise. Mutates the
    # module in place by delegating to SpectralNorm.apply (errors propagate from there).
    r"""Apply spectral normalization to a parameter in the given module.

    .. math::
        \mathbf{W}_{SN} = \dfrac{\mathbf{W}}{\sigma(\mathbf{W})},
        \sigma(\mathbf{W}) = \max_{\mathbf{h}: \mathbf{h} \ne 0} \dfrac{\|\mathbf{W} \mathbf{h}\|_2}{\|\mathbf{h}\|_2}

    Spectral normalization stabilizes the training of discriminators (critics)
    in Generative Adversarial Networks (GANs) by rescaling the weight tensor
    with spectral norm :math:`\sigma` of the weight matrix calculated using
    power iteration method. If the dimension of the weight tensor is greater
    than 2, it is reshaped to 2D in power iteration method to get spectral
    norm. This is implemented via a hook that calculates spectral norm and
    rescales weight before every :meth:`~Module.forward` call.

    See `Spectral Normalization for Generative Adversarial Networks`_ .

    .. _`Spectral Normalization for Generative Adversarial Networks`: https://arxiv.org/abs/1802.05957

    Args:
        module (nn.Module): containing module
        name (str, optional): name of weight parameter
        n_power_iterations (int, optional): number of power iterations to
            calculate spectral norm
        eps (float, optional): epsilon for numerical stability in
            calculating norms
        dim (int, optional): dimension corresponding to number of outputs,
            the default is ``0``, except for modules that are instances of
            ConvTranspose{1,2,3}d, when it is ``1``

    Returns:
        The original module with the spectral norm hook

    .. note::
        This function has been reimplemented as
        :func:`torch.nn.utils.parametrizations.spectral_norm` using the new
        parametrization functionality in
        :func:`torch.nn.utils.parametrize.register_parametrization`. Please use
        the newer version. This function will be deprecated in a future version
        of PyTorch.

    Example::

        >>> m = spectral_norm(nn.Linear(20, 40))
        >>> m
        Linear(in_features=20, out_features=40, bias=True)
        >>> m.weight_u.size()
        torch.Size([40])

    """
    if dim is None:
        if isinstance(
            module,
            (
                torch.nn.ConvTranspose1d,
                torch.nn.ConvTranspose2d,
                torch.nn.ConvTranspose3d,
            ),
        ):
            dim = 1
        else:
            dim = 0
    SpectralNorm.apply(module, name, n_power_iterations, dim, eps)

    return module


def remove_spectral_norm(module: T_module, name: str = "weight") -> T_module:
    # [FUNCTION DOC] Remove spectral normalization from a module and return it.
    # Finds the matching forward pre-hook, restores a plain parameter, and deletes the hook
    # along with the matching state_dict hook and load-state-dict pre-hook. Raises ValueError
    # if no spectral norm hook exists for the given name.
    r"""Remove the spectral normalization reparameterization from a module.

    Args:
        module (Module): containing module
        name (str, optional): name of weight parameter

    Example:
        >>> m = spectral_norm(nn.Linear(40, 10))
        >>> remove_spectral_norm(m)
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, SpectralNorm) and hook.name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            break
    else:
        raise ValueError(f"spectral_norm of '{name}' not found in {module}")

    for k, hook in module._state_dict_hooks.items():
        if isinstance(hook, SpectralNormStateDictHook) and hook.fn.name == name:
            del module._state_dict_hooks[k]
            break

    for k, hook in module._load_state_dict_pre_hooks.items():
        if isinstance(hook, SpectralNormLoadStateDictPreHook) and hook.fn.name == name:
            del module._load_state_dict_pre_hooks[k]
            break

    return module
