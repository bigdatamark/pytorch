# mypy: allow-untyped-defs
# [MODULE DOC]
# Pruning utilities for torch.nn modules. Pruning a tensor named name on a module replaces
# the parameter with a pruned value computed as mask times name_orig (a new parameter holding
# the original), stores the mask in a buffer name_mask, and registers a forward pre-hook (the
# pruning method object) that recomputes the pruned value before every forward call.
# Contents: BasePruningMethod (abstract base), PruningContainer (iterative pruning), concrete
# methods (Identity, RandomUnstructured, L1Unstructured, RandomStructured, LnStructured,
# CustomFromMask), functional wrappers (identity, random_unstructured, l1_unstructured,
# random_structured, ln_structured, global_unstructured, custom_from_mask, remove, is_pruned)
# and private validation/norm helpers. No import-time side effects.
r"""Pruning methods."""

import numbers
from abc import ABC, abstractmethod
from collections.abc import Iterable

import torch


class BasePruningMethod(ABC):
    # [CLASS DOC] Abstract base for pruning techniques; instances double as forward pre-hooks.
    # State: _tensor_name, the name of the module tensor being pruned (set by apply; read by
    # the hook and by remove). Subclasses define a class attribute PRUNING_TYPE (unstructured,
    # structured or global) and implement compute_mask. Key methods: apply (install pruning on
    # a module), apply_mask (mask times original), prune (functional, no module), remove
    # (make pruning permanent and drop the reparametrization).
    r"""Abstract base class for creation of new pruning techniques.

    Provides a skeleton for customization requiring the overriding of methods
    such as :meth:`compute_mask` and :meth:`apply`.
    """

    _tensor_name: str

    def __call__(self, module, inputs):
        # [FUNCTION DOC] Forward pre-hook entry point: recompute the pruned tensor from the
        # module's
        # original tensor and mask and assign it to the module attribute. The inputs are
        # ignored; returns nothing (so the forward inputs are not altered).
        r"""Multiply the mask into original tensor and store the result.

        Multiplies the mask (stored in ``module[name + '_mask']``)
        into the original tensor (stored in ``module[name + '_orig']``)
        and stores the result into ``module[name]`` by using :meth:`apply_mask`.

        Args:
            module (nn.Module): module containing the tensor to prune
            inputs: not used.
        """
        setattr(module, self._tensor_name, self.apply_mask(module))

    @abstractmethod
    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Abstract: subclasses return a mask (same shape as t) that respects the
        # prior
        # default_mask and zeroes additional entries according to the pruning rule, using t
        # as the importance scores. Must not mutate the inputs.
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` according to the specific pruning
        method recipe.

        Args:
            t (torch.Tensor): tensor representing the importance scores of the
            parameter to prune.
            default_mask (torch.Tensor): Base mask from previous pruning
            iterations, that need to be respected after the new mask is
            applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``
        """

    def apply_mask(self, module):
        # [FUNCTION DOC] Return the pruned tensor: the module's mask buffer (cast to the original's
        # dtype) multiplied by its original parameter. Raises AssertionError if the method has
        # no tensor name; raises AttributeError if the mask or original are missing.
        # KNOWN ISSUE: the tensor name is only annotated as str and never defaulted, so an
        # unapplied method raises AttributeError rather than the intended AssertionError.
        r"""Simply handles the multiplication between the parameter being pruned and the generated mask.

        Fetches the mask and the original tensor from the module
        and returns the pruned version of the tensor.

        Args:
            module (nn.Module): module containing the tensor to prune

        Returns:
            pruned_tensor (torch.Tensor): pruned version of the input tensor
        """
        # to carry out the multiplication, the mask needs to have been computed,
        # so the pruning method must know what tensor it's operating on
        if self._tensor_name is None:
            raise AssertionError(
                f"Module {module} has to be pruned"
            )  # this gets set in apply()
        mask = getattr(module, self._tensor_name + "_mask")
        orig = getattr(module, self._tensor_name + "_orig")
        pruned_tensor = mask.to(dtype=orig.dtype) * orig
        return pruned_tensor

    @classmethod
    def apply(cls, module, name, *args, importance_scores=None, **kwargs):
        # [FUNCTION DOC] Class method that installs this pruning method on module's tensor name and
        # returns the method (or the container holding it). Steps: build the method from args and
        # kwargs, merging with any pruning hook already on that tensor into a PruningContainer
        # (existing hooks are removed from the module); validate importance_scores shape
        # (AssertionError on mismatch; defaults to the tensor itself); on first pruning move the
        # parameter to name_orig and start from an all-ones mask, else start from the current
        # mask; compute the mask, register it as a buffer name_mask, assign the pruned tensor
        # and register the method as a forward pre-hook. On any error during masking, a first
        # pruning is rolled back (parameter restored) and the error is re-raised.
        # KNOWN ISSUE: when merging into an existing container fails, the removed old hook is
        # not re-registered, so the module is left without its pruning hook.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            args: arguments passed on to a subclass of
                :class:`BasePruningMethod`
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the
                corresponding elements in the parameter being pruned.
                If unspecified or None, the parameter will be used in its place.
            kwargs: keyword arguments passed on to a subclass of a
                :class:`BasePruningMethod`
        """

        def _get_composite_method(cls, module, name, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Find and remove any existing pruning hook for name on module
            # (AssertionError if more than one), create the new method with its tensor name set,
            # and return it, or a PruningContainer combining the old and new methods when one
            # existed. Mutates the module's forward pre-hook mapping.
            # Check if a pruning method has already been applied to
            # `module[name]`. If so, store that in `old_method`.
            old_method = None
            found = 0
            # there should technically be only 1 hook with hook.name == name
            # assert this using `found`
            hooks_to_remove = []
            for k, hook in module._forward_pre_hooks.items():
                # if it exists, take existing thing, remove hook, then
                # go through normal thing
                if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
                    old_method = hook
                    hooks_to_remove.append(k)
                    found += 1
            if found > 1:
                raise AssertionError(
                    f"Avoid adding multiple pruning hooks to the "
                    f"same tensor {name} of module {module}. Use a PruningContainer."
                )

            for k in hooks_to_remove:
                del module._forward_pre_hooks[k]

            # Apply the new pruning method, either from scratch or on top of
            # the previous one.
            method = cls(*args, **kwargs)  # new pruning
            # Have the pruning method remember what tensor it's been applied to
            method._tensor_name = name

            # combine `methods` with `old_method`, if `old_method` exists
            if old_method is not None:  # meaning that there was a hook
                # if the hook is already a pruning container, just add the
                # new pruning method to the container
                if isinstance(old_method, PruningContainer):
                    old_method.add_pruning_method(method)
                    method = old_method  # rename old_method --> method

                # if the hook is simply a single pruning method, create a
                # container, add the old pruning method and the new one
                elif isinstance(old_method, BasePruningMethod):
                    container = PruningContainer(old_method)
                    # Have the pruning method remember the name of its tensor
                    # setattr(container, '_tensor_name', name)
                    container.add_pruning_method(method)
                    method = container  # rename container --> method
            return method

        method = _get_composite_method(cls, module, name, *args, **kwargs)
        # at this point we have no forward_pre_hooks but we could have an
        # active reparameterization of the tensor if another pruning method
        # had been applied (in which case `method` would be a PruningContainer
        # and not a simple pruning method).

        # Pruning is to be applied to the module's tensor named `name`,
        # starting from the state it is found in prior to this iteration of
        # pruning. The pruning mask is calculated based on importances scores.

        orig = getattr(module, name)
        if importance_scores is not None:
            if importance_scores.shape != orig.shape:
                raise AssertionError(
                    f"importance_scores should have the same shape as parameter "
                    f"{name} of {module}, got {importance_scores.shape} vs {orig.shape}"
                )
        else:
            importance_scores = orig

        # If this is the first time pruning is applied, take care of moving
        # the original tensor to a new parameter called name + '_orig'
        # and deleting the original parameter
        if not isinstance(method, PruningContainer):
            # copy `module[name]` to `module[name + '_orig']`
            module.register_parameter(name + "_orig", orig)
            # temporarily delete `module[name]`
            del module._parameters[name]
            default_mask = torch.ones_like(orig)  # temp
        # If this is not the first time pruning is applied, all of the above
        # has been done before in a previous pruning iteration, so we're good
        # to go
        else:
            default_mask = (
                getattr(module, name + "_mask")
                .detach()
                .clone(memory_format=torch.contiguous_format)
            )

        # Use try/except because if anything goes wrong with the mask
        # computation etc., you'd want to roll back.
        try:
            # get the final mask, computed according to the specific method
            mask = method.compute_mask(importance_scores, default_mask=default_mask)
            # reparameterize by saving mask to `module[name + '_mask']`...
            module.register_buffer(name + "_mask", mask)
            # ... and the new pruned tensor to `module[name]`
            setattr(module, name, method.apply_mask(module))
            # associate the pruning method to the module via a hook to
            # compute the function before every forward() (compile by run)
            module.register_forward_pre_hook(method)

        except Exception as e:
            if not isinstance(method, PruningContainer):
                orig = getattr(module, name + "_orig")
                module.register_parameter(name, orig)
                del module._parameters[name + "_orig"]
            raise e

        return method

    def prune(self, t, default_mask=None, importance_scores=None):
        # [FUNCTION DOC] Functional pruning with no module: return t multiplied by the mask from
        # compute_mask. default_mask defaults to ones and importance_scores to t itself;
        # raises AssertionError if importance_scores has a different shape than t.
        r"""Compute and returns a pruned version of input tensor ``t``.

        According to the pruning rule specified in :meth:`compute_mask`.

        Args:
            t (torch.Tensor): tensor to prune (of same dimensions as
                ``default_mask``).
            importance_scores (torch.Tensor): tensor of importance scores (of
                same shape as ``t``) used to compute mask for pruning ``t``.
                The values in this tensor indicate the importance of the
                corresponding elements in the ``t`` that is being pruned.
                If unspecified or None, the tensor ``t`` will be used in its place.
            default_mask (torch.Tensor, optional): mask from previous pruning
                iteration, if any. To be considered when determining what
                portion of the tensor that pruning should act on. If None,
                default to a mask of ones.

        Returns:
            pruned version of tensor ``t``.
        """
        if importance_scores is not None:
            if importance_scores.shape != t.shape:
                raise AssertionError(
                    f"importance_scores should have the same shape as tensor t, "
                    f"got {importance_scores.shape} vs {t.shape}"
                )
        else:
            importance_scores = t
        default_mask = default_mask if default_mask is not None else torch.ones_like(t)
        return t * self.compute_mask(importance_scores, default_mask=default_mask)

    def remove(self, module) -> None:
        # [FUNCTION DOC] Make the pruning permanent: compute the masked tensor, delete the module
        # attribute, write the masked values into the original parameter's data, drop the
        # name_orig parameter and name_mask buffer, and re-register the original parameter under
        # its plain name. Does not unregister the forward hook (the functional remove does).
        # Raises AssertionError if the method was never applied.
        r"""Remove the pruning reparameterization from a module.

        The pruned parameter named ``name`` remains permanently pruned,
        and the parameter named ``name+'_orig'`` is removed from the parameter list.
        Similarly, the buffer named ``name+'_mask'`` is removed from the buffers.

        Note:
            Pruning itself is NOT undone or reversed!
        """
        # before removing pruning from a tensor, it has to have been applied
        if self._tensor_name is None:
            raise AssertionError(
                f"Module {module} has to be pruned before pruning can be removed"
            )  # this gets set in apply()

        # to update module[name] to latest trained weights
        weight = self.apply_mask(module)  # masked weights

        # delete and reset
        if hasattr(module, self._tensor_name):
            delattr(module, self._tensor_name)
        orig = module._parameters[self._tensor_name + "_orig"]
        orig.data = weight.data
        del module._parameters[self._tensor_name + "_orig"]
        del module._buffers[self._tensor_name + "_mask"]
        setattr(module, self._tensor_name, orig)


class PruningContainer(BasePruningMethod):
    # [CLASS DOC] Ordered, iterable container of pruning methods applied successively to one
    # tensor (iterative pruning). State: _pruning_methods tuple and the shared _tensor_name
    # (all members must act on the same name). compute_mask applies only the latest method
    # on top of the mask left by earlier ones. Supports len, iteration and indexing.
    """Container holding a sequence of pruning methods for iterative pruning.

    Keeps track of the order in which pruning methods are applied and handles
    combining successive pruning calls.

    Accepts as argument an instance of a BasePruningMethod or an iterable of
    them.
    """

    def __init__(self, *args) -> None:
        # [FUNCTION DOC] Create a container from zero or more pruning methods. With exactly one
        # argument, adopt its tensor name and add it; with several (or none), add each in order.
        # KNOWN ISSUE: the argument pack is always a tuple, so the first (non-iterable)
        # branch is dead code; with several methods no tensor name is set on the container.
        self._pruning_methods: tuple[BasePruningMethod, ...] = ()
        if not isinstance(args, Iterable):  # only 1 item
            self._tensor_name = args._tensor_name
            self.add_pruning_method(args)

        elif len(args) == 1:  # only 1 item in a tuple
            self._tensor_name = args[0]._tensor_name

            self.add_pruning_method(args[0])
        else:  # manual construction from list or other iterable (or no args)
            for method in args:
                self.add_pruning_method(method)

    def add_pruning_method(self, method) -> None:
        # [FUNCTION DOC] Append a pruning method to the container. Raises TypeError if it is not a
        # BasePruningMethod (None is tolerated and appended), and ValueError if its tensor name
        # differs from the container's. Mutates the stored tuple.
        r"""Add a child pruning ``method`` to the container.

        Args:
            method (subclass of BasePruningMethod): child pruning method
                to be added to the container.
        """
        # check that we're adding a pruning method to the container
        if not isinstance(method, BasePruningMethod) and method is not None:
            raise TypeError(f"{type(method)} is not a BasePruningMethod subclass")
        elif method is not None and self._tensor_name != method._tensor_name:
            raise ValueError(
                "Can only add pruning methods acting on "
                f"the parameter named '{self._tensor_name}' to PruningContainer {self}."
                + f" Found '{method._tensor_name}'"
            )
        # if all checks passed, add to _pruning_methods tuple
        self._pruning_methods += (method,)  # type: ignore[operator]

    def __len__(self) -> int:
        # [FUNCTION DOC] Return the number of contained pruning methods.
        return len(self._pruning_methods)

    def __iter__(self):
        # [FUNCTION DOC] Iterate over the contained pruning methods in application order.
        return iter(self._pruning_methods)

    def __getitem__(self, idx):
        # [FUNCTION DOC] Return the contained method (or slice of methods) at the given index.
        return self._pruning_methods[idx]

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Compute the combined mask by running only the last contained method over
        # the still-unpruned portion of t (chosen by that method's PRUNING_TYPE) and writing
        # its partial mask back into default_mask's positions. Returns the new mask.
        # KNOWN ISSUE: the new mask is created by casting default_mask; when the dtype already
        # matches this is the same tensor, so default_mask is modified in place.
        r"""Apply the latest ``method`` by computing the new partial masks and returning its combination with the ``default_mask``.

        The new partial mask should be computed on the entries or channels
        that were not zeroed out by the ``default_mask``.
        Which portions of the tensor ``t`` the new mask will be calculated from
        depends on the ``PRUNING_TYPE`` (handled by the type handler):

        * for 'unstructured', the mask will be computed from the raveled
          list of nonmasked entries;

        * for 'structured', the mask will be computed from the nonmasked
          channels in the tensor;

        * for 'global', the mask will be computed across all entries.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
                (of same dimensions as ``default_mask``).
            default_mask (torch.Tensor): mask from previous pruning iteration.

        Returns:
            mask (torch.Tensor): new mask that combines the effects
            of the ``default_mask`` and the new mask from the current
            pruning ``method`` (of same dimensions as ``default_mask`` and
            ``t``).
        """

        def _combine_masks(method, t, mask):
            # [FUNCTION DOC] HELPER: Select the slice of t that method should act on (entries where
            # the mask is one for unstructured; channels not fully zeroed along the method's dim
            # for structured, raising AttributeError without a dim and IndexError for a bad dim;
            # everything for global; ValueError for an unknown type), ask the method for a
            # partial mask on that slice, write it into the mask slice and return the mask.
            r"""Combine the masks from all pruning methods and returns a new mask.

            Args:
                method (a BasePruningMethod subclass): pruning method
                    currently being applied.
                t (torch.Tensor): tensor representing the parameter to prune
                    (of same dimensions as mask).
                mask (torch.Tensor): mask from previous pruning iteration

            Returns:
                new_mask (torch.Tensor): new mask that combines the effects
                    of the old mask and the new mask from the current
                    pruning method (of same dimensions as mask and t).
            """
            new_mask = mask  # start off from existing mask
            new_mask = new_mask.to(dtype=t.dtype)

            # compute a slice of t onto which the new pruning method will operate
            if method.PRUNING_TYPE == "unstructured":
                # prune entries of t where the mask is 1
                slc = mask == 1

            # for struct pruning, exclude channels that have already been
            # entirely pruned
            elif method.PRUNING_TYPE == "structured":
                if not hasattr(method, "dim"):
                    raise AttributeError(
                        "Pruning methods of PRUNING_TYPE "
                        '"structured" need to have the attribute `dim` defined.'
                    )

                # find the channels to keep by removing the ones that have been
                # zeroed out already (i.e. where sum(entries) == 0)
                n_dims = t.dim()  # "is this a 2D tensor? 3D? ..."
                dim = method.dim
                # convert negative indexing
                if dim < 0:
                    dim = n_dims + dim
                # if dim is still negative after subtracting it from n_dims
                if dim < 0:
                    raise IndexError(
                        f"Index is out of bounds for tensor with dimensions {n_dims}"
                    )
                # find channels along dim = dim that aren't already tots 0ed out
                keep_channel = mask.sum(dim=[d for d in range(n_dims) if d != dim]) != 0
                # create slice to identify what to prune
                slc = [slice(None)] * n_dims
                slc[dim] = keep_channel

            elif method.PRUNING_TYPE == "global":
                n_dims = len(t.shape)  # "is this a 2D tensor? 3D? ..."
                slc = [slice(None)] * n_dims

            else:
                raise ValueError(f"Unrecognized PRUNING_TYPE {method.PRUNING_TYPE}")

            # compute the new mask on the unpruned slice of the tensor t
            if isinstance(slc, list):
                slc = tuple(slc)
            partial_mask = method.compute_mask(t[slc], default_mask=mask[slc])
            new_mask[slc] = partial_mask.to(dtype=new_mask.dtype)

            return new_mask

        method = self._pruning_methods[-1]
        mask = _combine_masks(method, t, default_mask)
        return mask


class Identity(BasePruningMethod):
    # [CLASS DOC] Unstructured pruning method that prunes nothing: its mask is the incoming
    # default mask (all ones on first use). Useful to set up the reparametrization only.
    r"""Utility pruning method that does not prune any units but generates the pruning parametrization with a mask of ones."""

    PRUNING_TYPE = "unstructured"

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Return default_mask unchanged (t is ignored).
        mask = default_mask
        return mask

    @classmethod
    def apply(cls, module, name):  # type: ignore[override]
        # [FUNCTION DOC] Install the identity pruning on a tensor of module via the base class and
        # return the method or container; same side effects as the base apply.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super().apply(module, name)


class RandomUnstructured(BasePruningMethod):
    # [CLASS DOC] Unstructured pruning that zeroes a random subset of entries. State: amount
    # (absolute count if integer, fraction of entries if float).
    r"""Prune (currently unpruned) units in a tensor at random.

    Args:
        name (str): parameter name within ``module`` on which pruning
            will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount) -> None:
        # [FUNCTION DOC] Validate amount (TypeError for non-numbers, ValueError for a float outside
        # zero to one or a negative integer) and store it.
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Convert amount to a count over the number of entries of t, validate it
        # (ValueError if too large), then clone default_mask and zero that many entries chosen
        # by the highest random scores (random choice over all entries, not only unpruned
        # ones). Returns the new mask.
        # KNOWN ISSUE: selection ignores default_mask, so already-pruned entries may be
        # picked again and fewer than the requested number of new entries get pruned.
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            prob = torch.rand_like(t)
            topk = torch.topk(prob.view(-1), k=nparams_toprune)
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount):  # type: ignore[override]
        # [FUNCTION DOC] Install random unstructured pruning with the given amount on module's
        # tensor name via the base class; returns the method or container.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
        """
        return super().apply(module, name, amount=amount)


class L1Unstructured(BasePruningMethod):
    # [CLASS DOC] Unstructured pruning that zeroes the entries with the smallest absolute value
    # (L1 magnitude). State: amount (absolute count if integer, fraction if float).
    r"""Prune (currently unpruned) units in a tensor by zeroing out the ones with the lowest L1-norm.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
    """

    PRUNING_TYPE = "unstructured"

    def __init__(self, amount) -> None:
        # [FUNCTION DOC] Validate amount (TypeError or ValueError when invalid) and store it.
        # Check range of validity of pruning amount
        _validate_pruning_amount_init(amount)
        self.amount = amount

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Convert amount to a count over the entries of t, validate it (ValueError
        # if
        # too large), clone default_mask and zero the entries of t with the smallest absolute
        # values. Returns the new mask.
        # KNOWN ISSUE: the smallest entries are chosen over all of t, including entries already
        # masked, so earlier-pruned (zero) entries use up the quota first when t is the
        # masked tensor.
        # Check that the amount of units to prune is not > than the number of
        # parameters in t
        tensor_size = t.nelement()
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        mask = default_mask.clone(memory_format=torch.contiguous_format)

        if nparams_toprune != 0:  # k=0 not supported by torch.kthvalue
            # largest=True --> top k; largest=False --> bottom k
            # Prune the smallest k
            topk = torch.topk(torch.abs(t).view(-1), k=nparams_toprune, largest=False)
            # topk will have .indices and .values
            mask.view(-1)[topk.indices] = 0

        return mask

    @classmethod
    def apply(cls, module, name, amount, importance_scores=None):  # type: ignore[override]
        # [FUNCTION DOC] Install L1 unstructured pruning with the given amount (and optional
        # importance scores) on module's tensor name via the base class; returns the method or
        # container.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super().apply(
            module, name, amount=amount, importance_scores=importance_scores
        )


class RandomStructured(BasePruningMethod):
    # [CLASS DOC] Structured pruning that zeroes randomly chosen whole channels along a dim.
    # State: amount (count or fraction of channels) and dim (default last dimension).
    r"""Prune entire (currently unpruned) channels in a tensor at random.

    Args:
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, dim=-1) -> None:
        # [FUNCTION DOC] Validate amount (TypeError or ValueError when invalid) and store amount
        # and
        # dim.
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.dim = dim

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Validate that t has at least two dims and that dim is in range
        # (ValueError,
        # IndexError), convert amount to a channel count along dim and validate it, then build
        # a mask that keeps random channels (all channels kept when the count is zero) and
        # multiply it by default_mask. Returns the mask.
        # KNOWN ISSUE: random channel scores are drawn on the default device regardless of t's
        # device, and the number of channels pruned ignores channels already zeroed.
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a random mask to
        apply on top of the ``default_mask`` by randomly zeroing out channels
        along the specified dim of the tensor.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied. Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)

        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, nchannels, nchannels_toprune):
            # [FUNCTION DOC] HELPER: Assign a random score to each channel, zero the channels whose
            # score is at or below the k-th smallest and return a mask shaped like t with ones in
            # the remaining channels along dim.
            # generate a random number in [0, 1] to associate to each channel
            prob = torch.rand(nchannels)
            # generate mask for each channel by 0ing out the channels that
            # got assigned the k = nchannels_toprune lowest values in prob
            threshold = torch.kthvalue(prob, k=nchannels_toprune).values
            channel_mask = prob > threshold

            mask = torch.zeros_like(t)
            slc = [slice(None)] * len(t.shape)
            slc[dim] = channel_mask
            slc = tuple(slc)
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            # apply the new structured mask on top of prior (potentially
            # unstructured) mask
            mask = make_mask(t, self.dim, tensor_size, nparams_toprune)
            mask *= default_mask.to(dtype=mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, amount, dim=-1):  # type: ignore[override]
        # [FUNCTION DOC] Install random structured pruning (amount, dim) on module's tensor name
        # via the base class; returns the method or container.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            dim (int, optional): index of the dim along which we define
                channels to prune. Default: -1.
        """
        return super().apply(module, name, amount=amount, dim=dim)


class LnStructured(BasePruningMethod):
    # [CLASS DOC] Structured pruning that zeroes the channels along a dim with the smallest Ln
    # norm. State: amount (count or fraction of channels), n (norm order passed to the norm
    # helper) and dim (default last dimension).
    r"""Prune entire (currently unpruned) channels in a tensor based on their L\ ``n``-norm.

    Args:
        amount (int or float): quantity of channels to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int, optional): index of the dim along which we define
            channels to prune. Default: -1.
    """

    PRUNING_TYPE = "structured"

    def __init__(self, amount, n, dim=-1) -> None:
        # [FUNCTION DOC] Validate amount (TypeError or ValueError when invalid) and store amount,
        # norm order n and dim.
        # Check range of validity of amount
        _validate_pruning_amount_init(amount)
        self.amount = amount
        self.n = n
        self.dim = dim

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Validate t (at least two dims) and dim (ValueError, IndexError), convert
        # amount to a channel count along dim and validate it, compute each channel's norm,
        # keep the channels with the largest norms (all kept if the count is zero) and multiply
        # the resulting mask by default_mask. Returns the mask.
        r"""Compute and returns a mask for the input tensor ``t``.

        Starting from a base ``default_mask`` (which should be a mask of ones
        if the tensor has not been pruned yet), generate a mask to apply on
        top of the ``default_mask`` by zeroing out the channels along the
        specified dim with the lowest L\ ``n``-norm.

        Args:
            t (torch.Tensor): tensor representing the parameter to prune
            default_mask (torch.Tensor): Base mask from previous pruning
                iterations, that need to be respected after the new mask is
                applied.  Same dims as ``t``.

        Returns:
            mask (torch.Tensor): mask to apply to ``t``, of same dims as ``t``

        Raises:
            IndexError: if ``self.dim >= len(t.shape)``
        """
        # Check that tensor has structure (i.e. more than 1 dimension) such
        # that the concept of "channels" makes sense
        _validate_structured_pruning(t)
        # Check that self.dim is a valid dim to index t, else raise IndexError
        _validate_pruning_dim(t, self.dim)

        # Check that the amount of channels to prune is not > than the number of
        # channels in t along the dim to prune
        tensor_size = t.shape[self.dim]
        # Compute number of units to prune: amount if int,
        # else amount * tensor_size
        nparams_toprune = _compute_nparams_toprune(self.amount, tensor_size)
        nparams_tokeep = tensor_size - nparams_toprune
        # This should raise an error if the number of units to prune is larger
        # than the number of units in the tensor
        _validate_pruning_amount(nparams_toprune, tensor_size)

        # Structured pruning prunes entire channels so we need to know the
        # L_n norm along each channel to then find the topk based on this
        # metric
        norm = _compute_norm(t, self.n, self.dim)
        # largest=True --> top k; largest=False --> bottom k
        # Keep the largest k channels along dim=self.dim
        topk = torch.topk(norm, k=nparams_tokeep, largest=True)
        # topk will have .indices and .values

        # Compute binary mask by initializing it to all 0s and then filling in
        # 1s wherever topk.indices indicates, along self.dim.
        # mask has the same shape as tensor t
        def make_mask(t, dim, indices):
            # [FUNCTION DOC] HELPER: Return a mask shaped like t that is one on the channels along
            # dim
            # listed in indices and zero elsewhere.
            # init mask to 0
            mask = torch.zeros_like(t)
            # e.g.: slc = [None, None, None], if len(t.shape) = 3
            slc = [slice(None)] * len(t.shape)
            # replace a None at position=dim with indices
            # e.g.: slc = [None, None, [0, 2, 3]] if dim=2 & indices=[0,2,3]
            slc[dim] = indices
            slc = tuple(slc)
            # use slc to slice mask and replace all its entries with 1s
            # e.g.: mask[:, :, [0, 2, 3]] = 1
            mask[slc] = 1
            return mask

        if nparams_toprune == 0:  # k=0 not supported by torch.kthvalue
            mask = default_mask
        else:
            mask = make_mask(t, self.dim, topk.indices)
            mask *= default_mask.to(dtype=mask.dtype)

        return mask

    @classmethod
    def apply(cls, module, name, amount, n, dim, importance_scores=None):  # type: ignore[override]
        # [FUNCTION DOC] Install Ln structured pruning (amount, n, dim, optional importance scores)
        # on module's tensor name via the base class; returns the method or container.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
            amount (int or float): quantity of parameters to prune.
                If ``float``, should be between 0.0 and 1.0 and represent the
                fraction of parameters to prune. If ``int``, it represents the
                absolute number of parameters to prune.
            n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
                entries for argument ``p`` in :func:`torch.norm`.
            dim (int): index of the dim along which we define channels to
                prune.
            importance_scores (torch.Tensor): tensor of importance scores (of same
                shape as module parameter) used to compute mask for pruning.
                The values in this tensor indicate the importance of the corresponding
                elements in the parameter being pruned.
                If unspecified or None, the module parameter will be used in its place.
        """
        return super().apply(
            module,
            name,
            amount=amount,
            n=n,
            dim=dim,
            importance_scores=importance_scores,
        )


class CustomFromMask(BasePruningMethod):
    # [CLASS DOC] Global-type pruning method that applies a user-supplied binary mask. State:
    # mask, a tensor with the same shape as the tensor being pruned.
    PRUNING_TYPE = "global"

    def __init__(self, mask) -> None:
        # [FUNCTION DOC] Store the user-provided mask without validation.
        self.mask = mask

    def compute_mask(self, t, default_mask):
        # [FUNCTION DOC] Return default_mask multiplied by the stored mask (cast to its dtype).
        # Raises AssertionError if the shapes differ. t is unused.
        if default_mask.shape != self.mask.shape:
            raise AssertionError(
                f"default_mask shape {default_mask.shape} must match "
                f"self.mask shape {self.mask.shape}"
            )
        mask = default_mask * self.mask.to(dtype=default_mask.dtype)
        return mask

    @classmethod
    def apply(cls, module, name, mask):  # type: ignore[override]
        # [FUNCTION DOC] Install pruning with the given precomputed mask on module's tensor name
        # via the base class; returns the method or container.
        r"""Add pruning on the fly and reparameterization of a tensor.

        Adds the forward pre-hook that enables pruning on the fly and
        the reparameterization of a tensor in terms of the original tensor
        and the pruning mask.

        Args:
            module (nn.Module): module containing the tensor to prune
            name (str): parameter name within ``module`` on which pruning
                will act.
        """
        return super().apply(module, name, mask=mask)


def identity(module, name):
    # [FUNCTION DOC] Apply the identity pruning (mask of ones) to module's tensor name, adding the
    # name_orig parameter, name_mask buffer and forward pre-hook. Returns the same module.
    r"""Apply pruning reparameterization without pruning any units.

    Applies pruning reparameterization to the tensor corresponding to the
    parameter called ``name`` in ``module`` without actually pruning any
    units. Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Note:
        The mask is a tensor of ones.

    Args:
        module (nn.Module): module containing the tensor to prune.
        name (str): parameter name within ``module`` on which pruning
                will act.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.identity(nn.Linear(2, 3), "bias")
        >>> print(m.bias_mask)
        tensor([1., 1., 1.])
    """
    Identity.apply(module, name)
    return module


def random_unstructured(module, name, amount):
    # [FUNCTION DOC] Randomly prune the given amount (count or fraction) of entries of module's
    # tensor name, with the standard reparametrization side effects. Returns the module.
    r"""Prune tensor by removing random (currently unpruned) units.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) units
    selected at random.
    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.random_unstructured(nn.Linear(2, 3), "weight", amount=1)
        >>> torch.sum(m.weight_mask == 0)
        tensor(1)

    """
    RandomUnstructured.apply(module, name, amount)
    return module


def l1_unstructured(module, name, amount, importance_scores=None):
    # [FUNCTION DOC] Prune the entries of module's tensor name with the smallest absolute
    # importance (optionally from importance_scores), with the standard reparametrization side
    # effects. Returns the module.
    r"""Prune tensor by removing units with the lowest L1-norm.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified `amount` of (currently unpruned) units with the
    lowest L1-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.l1_unstructured(nn.Linear(2, 3), "weight", amount=0.2)
        >>> m.state_dict().keys()
        odict_keys(['bias', 'weight_orig', 'weight_mask'])
    """
    L1Unstructured.apply(
        module, name, amount=amount, importance_scores=importance_scores
    )
    return module


def random_structured(module, name, amount, dim):
    # [FUNCTION DOC] Randomly prune whole channels along dim of module's tensor name, with the
    # standard reparametrization side effects. Returns the module.
    r"""Prune tensor by removing random channels along the specified dimension.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` selected at random.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        dim (int): index of the dim along which we define channels to prune.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> # xdoctest: +SKIP
        >>> m = prune.random_structured(nn.Linear(5, 3), "weight", amount=3, dim=1)
        >>> columns_pruned = int(sum(torch.sum(m.weight, dim=0) == 0))
        >>> print(columns_pruned)
        3
    """
    RandomStructured.apply(module, name, amount, dim)
    return module


def ln_structured(module, name, amount, n, dim, importance_scores=None):
    # [FUNCTION DOC] Prune the channels along dim of module's tensor name with the smallest norm
    # of order n (optionally from importance_scores), with the standard reparametrization side
    # effects. Returns the module.
    r"""Prune tensor by removing channels with the lowest L\ ``n``-norm along the specified dimension.

    Prunes tensor corresponding to parameter called ``name`` in ``module``
    by removing the specified ``amount`` of (currently unpruned) channels
    along the specified ``dim`` with the lowest L\ ``n``-norm.
    Modifies module in place (and also return the modified module)
    by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
                will act.
        amount (int or float): quantity of parameters to prune.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument ``p`` in :func:`torch.norm`.
        dim (int): index of the dim along which we define channels to prune.
        importance_scores (torch.Tensor): tensor of importance scores (of same
            shape as module parameter) used to compute mask for pruning.
            The values in this tensor indicate the importance of the corresponding
            elements in the parameter being pruned.
            If unspecified or None, the module parameter will be used in its place.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.ln_structured(
        ...     nn.Conv2d(5, 3, 2), "weight", amount=0.3, dim=1, n=float("-inf")
        ... )
    """
    LnStructured.apply(
        module, name, amount, n, dim, importance_scores=importance_scores
    )
    return module


def global_unstructured(
    parameters, pruning_method, importance_scores=None, **kwargs
) -> None:
    # [FUNCTION DOC] Globally prune several (module, name) tensors at once: flatten and
    # concatenate all importance scores (or the tensors) and existing masks, build the pruning
    # method from kwargs (must be unstructured, else TypeError), compute one combined mask,
    # then split it back per tensor and apply each via the custom-mask pruning (side effects on
    # every module). Raises TypeError if parameters is not iterable or importance_scores is
    # not a dict. Returns nothing. The parameters iterable is traversed several times, so it
    # must be re-iterable.
    r"""
    Globally prunes tensors corresponding to all parameters in ``parameters`` by applying the specified ``pruning_method``.

    Modifies modules in place by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        parameters (Iterable of (module, name) tuples): parameters of
            the model to prune in a global fashion, i.e. by aggregating all
            weights prior to deciding which ones to prune. module must be of
            type :class:`nn.Module`, and name must be a string.
        pruning_method (function): a valid pruning function from this module,
            or a custom one implemented by the user that satisfies the
            implementation guidelines and has ``PRUNING_TYPE='unstructured'``.
        importance_scores (dict): a dictionary mapping (module, name) tuples to
            the corresponding parameter's importance scores tensor. The tensor
            should be the same shape as the parameter, and is used for computing
            mask for pruning.
            If unspecified or None, the parameter will be used in place of its
            importance scores.
        kwargs: other keyword arguments such as:
            amount (int or float): quantity of parameters to prune across the
            specified parameters.
            If ``float``, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If ``int``, it represents the
            absolute number of parameters to prune.

    Raises:
        TypeError: if ``PRUNING_TYPE != 'unstructured'``

    Note:
        Since global structured pruning doesn't make much sense unless the
        norm is normalized by the size of the parameter, we now limit the
        scope of global pruning to unstructured methods.

    Examples:
        >>> from torch.nn.utils import prune
        >>> from collections import OrderedDict
        >>> net = nn.Sequential(
        ...     OrderedDict(
        ...         [
        ...             ("first", nn.Linear(10, 4)),
        ...             ("second", nn.Linear(4, 1)),
        ...         ]
        ...     )
        ... )
        >>> parameters_to_prune = (
        ...     (net.first, "weight"),
        ...     (net.second, "weight"),
        ... )
        >>> prune.global_unstructured(
        ...     parameters_to_prune,
        ...     pruning_method=prune.L1Unstructured,
        ...     amount=10,
        ... )
        >>> print(sum(torch.nn.utils.parameters_to_vector(net.buffers()) == 0))
        tensor(10)

    """
    # ensure parameters is a list or generator of tuples
    if not isinstance(parameters, Iterable):
        raise TypeError("global_unstructured(): parameters is not an Iterable")

    importance_scores = importance_scores if importance_scores is not None else {}
    if not isinstance(importance_scores, dict):
        raise TypeError("global_unstructured(): importance_scores must be of type dict")

    # flatten importance scores to consider them all at once in global pruning
    relevant_importance_scores = torch.nn.utils.parameters_to_vector(
        # pyrefly: ignore [bad-argument-type]
        [
            importance_scores.get((module, name), getattr(module, name))
            for (module, name) in parameters
        ]
    )
    # similarly, flatten the masks (if they exist), or use a flattened vector
    # of 1s of the same dimensions as t
    default_mask = torch.nn.utils.parameters_to_vector(
        [
            getattr(module, name + "_mask", torch.ones_like(getattr(module, name)))
            for (module, name) in parameters
        ]
    )

    # use the canonical pruning methods to compute the new mask, even if the
    # parameter is now a flattened out version of `parameters`
    container = PruningContainer()
    container._tensor_name = "temp"  # to make it match that of `method`
    method = pruning_method(**kwargs)
    method._tensor_name = "temp"  # to make it match that of `container`
    if method.PRUNING_TYPE != "unstructured":
        raise TypeError(
            'Only "unstructured" PRUNING_TYPE supported for '
            f"the `pruning_method`. Found method {pruning_method} of type {method.PRUNING_TYPE}"
        )

    container.add_pruning_method(method)

    # use the `compute_mask` method from `PruningContainer` to combine the
    # mask computed by the new method with the pre-existing mask
    final_mask = container.compute_mask(relevant_importance_scores, default_mask)

    # Pointer for slicing the mask to match the shape of each parameter
    pointer = 0
    for module, name in parameters:
        param = getattr(module, name)
        # The length of the parameter
        num_param = param.numel()
        # Slice the mask, reshape it
        param_mask = final_mask[pointer : pointer + num_param].view_as(param)
        # Assign the correct pre-computed mask to each parameter and add it
        # to the forward_pre_hooks like any other pruning method
        custom_from_mask(module, name, mask=param_mask)

        # Increment the pointer to continue slicing the final_mask
        pointer += num_param


def custom_from_mask(module, name, mask):
    # [FUNCTION DOC] Prune module's tensor name with the given precomputed mask, with the standard
    # reparametrization side effects. Returns the module.
    r"""Prune tensor corresponding to parameter called ``name`` in ``module`` by applying the pre-computed mask in ``mask``.

    Modifies module in place (and also return the modified module) by:

    1) adding a named buffer called ``name+'_mask'`` corresponding to the
       binary mask applied to the parameter ``name`` by the pruning method.
    2) replacing the parameter ``name`` by its pruned version, while the
       original (unpruned) parameter is stored in a new parameter named
       ``name+'_orig'``.

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.
        mask (Tensor): binary mask to be applied to the parameter.

    Returns:
        module (nn.Module): modified (i.e. pruned) version of the input module

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = prune.custom_from_mask(
        ...     nn.Linear(5, 3), name="bias", mask=torch.tensor([0, 1, 0])
        ... )
        >>> print(m.bias_mask)
        tensor([0., 1., 0.])

    """
    CustomFromMask.apply(module, name, mask)
    return module


def remove(module, name):
    # [FUNCTION DOC] Make the pruning of tensor name on module permanent: find the pruning hook for
    # that name, call its remove (restoring a plain parameter with the pruned values and
    # deleting the orig parameter and mask buffer), unregister the hook, and return the module.
    # Raises ValueError if the tensor is not pruned.
    r"""Remove the pruning reparameterization from a module and the pruning method from the forward hook.

    The pruned parameter named ``name`` remains permanently pruned, and the parameter
    named ``name+'_orig'`` is removed from the parameter list. Similarly,
    the buffer named ``name+'_mask'`` is removed from the buffers.

    Note:
        Pruning itself is NOT undone or reversed!

    Args:
        module (nn.Module): module containing the tensor to prune
        name (str): parameter name within ``module`` on which pruning
            will act.

    Examples:
        >>> m = random_unstructured(nn.Linear(5, 7), name="weight", amount=0.2)
        >>> m = remove(m, name="weight")
    """
    for k, hook in module._forward_pre_hooks.items():
        if isinstance(hook, BasePruningMethod) and hook._tensor_name == name:
            hook.remove(module)
            del module._forward_pre_hooks[k]
            return module

    raise ValueError(
        f"Parameter '{name}' of module {module} has to be pruned before pruning can be removed"
    )


def is_pruned(module) -> bool:
    # [FUNCTION DOC] Return True if module or any submodule has a pruning method registered as a
    # forward pre-hook, else False.
    r"""Check if a module is pruned by looking for pruning pre-hooks.

    Check whether ``module`` is pruned by looking for
    ``forward_pre_hooks`` in its modules that inherit from the
    :class:`BasePruningMethod`.

    Args:
        module (nn.Module): object that is either pruned or unpruned

    Returns:
        binary answer to whether ``module`` is pruned.

    Examples:
        >>> from torch.nn.utils import prune
        >>> m = nn.Linear(5, 7)
        >>> print(prune.is_pruned(m))
        False
        >>> prune.random_unstructured(m, name="weight", amount=0.2)
        >>> print(prune.is_pruned(m))
        True
    """
    for _, submodule in module.named_modules():
        for hook in submodule._forward_pre_hooks.values():
            if isinstance(hook, BasePruningMethod):
                return True
    return False


def _validate_pruning_amount_init(amount) -> None:
    # [FUNCTION DOC] HELPER: Raise TypeError if amount is not a real number; raise ValueError
    # if it is
    # a negative integer or a non-integer outside zero to one.
    r"""Validate helper to check the range of amount at init.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.

    Raises:
        ValueError: if amount is a float not in [0, 1], or if it's a negative
            integer.
        TypeError: if amount is neither a float nor an integer.

    Note:
        This does not take into account the number of parameters in the
        tensor to be pruned, which is known only at prune.
    """
    if not isinstance(amount, numbers.Real):
        raise TypeError(f"Invalid type for amount: {amount}. Must be int or float.")

    if (isinstance(amount, numbers.Integral) and amount < 0) or (
        not isinstance(amount, numbers.Integral)  # so it's a float
        and (float(amount) > 1.0 or float(amount) < 0.0)
    ):
        raise ValueError(
            f"amount={amount} should either be a float in the range [0, 1] or a non-negative integer"
        )


def _validate_pruning_amount(amount, tensor_size) -> None:
    # [FUNCTION DOC] HELPER: Raise ValueError if an integer amount exceeds tensor_size;
    # other values
    # pass unchecked.
    r"""Validate that the pruning amount is meaningful wrt to the size of the data.

    Validation helper to check that the amount of parameters to prune
    is meaningful wrt to the size of the data (`tensor_size`).

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.
    """
    # TODO: consider removing this check and allowing users to specify
    # a number of units to prune that is greater than the number of units
    # left to prune. In this case, the tensor will just be fully pruned.

    if isinstance(amount, numbers.Integral) and amount > tensor_size:
        raise ValueError(
            f"amount={amount} should be smaller than the number of parameters to prune={tensor_size}"
        )


def _validate_structured_pruning(t) -> None:
    # [FUNCTION DOC] HELPER: Raise ValueError unless t has at least two dimensions.
    r"""Validate that the tensor to be pruned is at least 2-Dimensional.

    Validation helper to check that the tensor to be pruned is multi-
    dimensional, such that the concept of "channels" is well-defined.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune

    Raises:
        ValueError: if the tensor `t` is not at least 2D.
    """
    shape = t.shape
    if len(shape) <= 1:
        raise ValueError(
            "Structured pruning can only be applied to "
            "multidimensional tensors. Found tensor of shape "
            f"{shape} with {len(shape)} dims"
        )


def _compute_nparams_toprune(amount, tensor_size):
    # [FUNCTION DOC] HELPER: Return amount unchanged if it is an integer, else the rounded
    # product of
    # the fraction and tensor_size.
    r"""Convert the pruning amount from a percentage to absolute value.

    Since amount can be expressed either in absolute value or as a
    percentage of the number of units/channels in a tensor, this utility
    function converts the percentage to absolute value to standardize
    the handling of pruning.

    Args:
        amount (int or float): quantity of parameters to prune.
            If float, should be between 0.0 and 1.0 and represent the
            fraction of parameters to prune. If int, it represents the
            absolute number of parameters to prune.
        tensor_size (int): absolute number of parameters in the tensor
            to prune.

    Returns:
        int: the number of units to prune in the tensor
    """
    # incorrect type already checked in _validate_pruning_amount_init
    if isinstance(amount, numbers.Integral):
        return amount
    else:
        return round(amount * tensor_size)


def _validate_pruning_dim(t, dim) -> None:
    # [FUNCTION DOC] HELPER: Raise IndexError if dim is not smaller than the number of dims of t.
    # KNOWN ISSUE: negative dims below minus the number of dims are not rejected here.
    r"""Validate that the pruning dimension is within the bounds of the tensor dimension.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        dim (int): index of the dim along which we define channels to prune
    """
    if dim >= t.dim():
        raise IndexError(f"Invalid index {dim} for tensor of size {t.shape}")


def _compute_norm(t, n, dim):
    # [FUNCTION DOC] HELPER: Return a vector with the norm of order n of each channel along dim,
    # reducing over all other dims. Nuclear (and Frobenius over exactly two reduced dims) use
    # matrix norms, everything else a vector p-norm (Frobenius meaning order two).
    r"""Compute the L_n-norm of a tensor along all dimensions except for the specified dimension.

    The L_n-norm will be computed across all entries in tensor `t` along all dimension
    except for the one identified by dim.
    Example: if `t` is of shape, say, 3x2x4 and dim=2 (the last dim),
    then norm will have Size [4], and each entry will represent the
    `L_n`-norm computed using the 3x2=6 entries for each of the 4 channels.

    Args:
        t (torch.Tensor): tensor representing the parameter to prune
        n (int, float, inf, -inf, 'fro', 'nuc'): See documentation of valid
            entries for argument p in torch.norm
        dim (int): dim identifying the channels to prune

    Returns:
        norm (torch.Tensor): L_n norm computed across all dimensions except
            for `dim`. By construction, `norm.shape = t.shape[-1]`.
    """
    # dims = all axes, except for the one identified by `dim`
    dims = list(range(t.dim()))
    # convert negative indexing
    if dim < 0:
        dim = dims[dim]
    dims.remove(dim)

    # torch.norm: 'nuc' and 'fro' over a matrix dim are matrix norms;
    # everything else (numeric n, 'fro' elsewhere) is the flat p-norm.
    is_matrix_dim = len(dims) == 2
    if n == "nuc" or (n == "fro" and is_matrix_dim):
        return torch.linalg.matrix_norm(
            t, ord=n, dim=dims if is_matrix_dim else (-2, -1)
        )
    return torch.linalg.vector_norm(t, ord=2 if n == "fro" else n, dim=dims)
