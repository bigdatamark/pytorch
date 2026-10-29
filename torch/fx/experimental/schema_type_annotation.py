# [MODULE DOC]
# Provides AnnotateTypesWithSchema, an FX Transformer that fills in the type field of
# graph nodes using Python signature return annotations (torch.nn.functional calls and
# torch.nn module forwards) and TorchScript type inference for get_attr values.
# No import-time side effects beyond imports.
import inspect
from typing import Any

import torch
import torch.fx
from torch._jit_internal import boolean_dispatched
from torch.fx import Transformer
from torch.fx.graph_module import GraphModule
from torch.fx.node import Argument, Target
from torch.fx.operator_schemas import _torchscript_type_to_python_type
from torch.fx.proxy import Proxy


__all__ = ["AnnotateTypesWithSchema"]


class AnnotateTypesWithSchema(Transformer):
    # [CLASS DOC]
    # Transformer that re-emits a GraphModule while setting node.type where it is unset.
    # State: three booleans (annotate_functionals, annotate_modules, annotate_get_attrs)
    # that enable each kind of annotation. Existing node types are never overwritten.
    """
    Use Python function signatures to annotate types for `Nodes` within an FX graph.
    This pulls out Python function signatures for:

        1. Standard `torch.nn` Module calls
        2. `torch.nn.functional` calls
        3. Attribute fetches via `get_attr`

    Example usage:

        m = torchvision.models.resnet18()

        traced = torch.fx.symbolic_trace(m)

        traced = AnnotateTypesWithSchema(traced).transform()

    """

    def __init__(
        self,
        module: GraphModule,
        annotate_functionals: bool = True,
        annotate_modules: bool = True,
        annotate_get_attrs: bool = True,
    ) -> None:
        # [FUNCTION DOC] Initializes the transformer for module and stores the three flags
        # selecting functional calls, module calls and get_attr nodes for annotation.
        super().__init__(module)
        self.annotate_functionals = annotate_functionals
        self.annotate_modules = annotate_modules
        self.annotate_get_attrs = annotate_get_attrs

    def call_function(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # [FUNCTION DOC] Runs the normal call_function transform and, when annotate_functionals
        # is set and the target belongs to torch.nn.functional, sets the result node's type
        # (if unset) to the target's return annotation. For boolean-dispatched functions
        # the true branch signature is used only if both branches have the same return
        # annotation; otherwise no annotation is added. Returns the result proxy.
        # Raises AssertionError if a boolean-dispatched target is a string.
        python_ret_type = None
        if self.annotate_functionals and target.__module__ == "torch.nn.functional":
            target_for_analysis = target
            if target in boolean_dispatched:
                # HACK: `boolean_dispatch` as used in `torch.nn.functional` makes it so that we have
                # a 2-way dispatch based on a boolean value. Here we check that the `true` and `false`
                # branches of the dispatch have exactly the same signature. If they do, use the `true`
                # branch signature for analysis. Otherwise, leave this un-normalized
                if isinstance(target, str):
                    raise AssertionError("target should not be a string here")
                dispatched = boolean_dispatched[target]
                if_true, if_false = dispatched["if_true"], dispatched["if_false"]
                # TODO: can we emit the union of these? What are the implications on TorchScript
                # compilation?
                if (
                    inspect.signature(if_true).return_annotation
                    != inspect.signature(if_false).return_annotation
                ):
                    return super().call_function(target, args, kwargs)
                target_for_analysis = if_true

            python_ret_type = self._extract_python_return_type(target_for_analysis)

        return_proxy = super().call_function(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def call_module(
        self, target: Target, args: tuple[Argument, ...], kwargs: dict[str, Any]
    ) -> Proxy:
        # [FUNCTION DOC] Runs the normal call_module transform and, when annotate_modules is
        # set and the submodule's class is exactly the torch.nn class of the same name, sets
        # the result node's type (if unset) to the return annotation of its forward.
        # Returns the result proxy. Raises AssertionError if target is not a string.
        python_ret_type = None
        if not isinstance(target, str):
            raise AssertionError(f"Expected str target, got {type(target)}")
        submod = self.fetch_attr(target)
        if self.annotate_modules and hasattr(submod.__class__, "__name__"):
            classname = submod.__class__.__name__
            if getattr(torch.nn, classname, None) == submod.__class__:
                python_ret_type = self._extract_python_return_type(submod.forward)
        return_proxy = super().call_module(target, args, kwargs)
        return_proxy.node.type = (
            return_proxy.node.type if return_proxy.node.type else python_ret_type
        )
        return return_proxy

    def get_attr(
        self,
        target: Target,
        args: tuple[Argument, ...],
        kwargs: dict[str, Any],
    ) -> Proxy:
        # [FUNCTION DOC] Runs the normal get_attr transform and, when annotate_get_attrs is
        # set, resolves the dotted target path on the module, asks TorchScript to infer the
        # attribute's type, and if successful sets the node type (if unset) to the Python
        # equivalent. Raises AssertionError for a non-string target and RuntimeError if a
        # path component does not exist. Returns the proxy.
        attr_proxy = super().get_attr(target, args, kwargs)

        if self.annotate_get_attrs:
            module_itr = self.module
            if not isinstance(target, str):
                raise AssertionError(f"Expected str target, got {type(target)}")
            atoms = target.split(".")
            for i, atom in enumerate(atoms):
                if not hasattr(module_itr, atom):
                    raise RuntimeError(
                        f"Node referenced nonextent target {'.'.join(atoms[:i])}!"
                    )
                module_itr = getattr(module_itr, atom)

            maybe_inferred_ts_type = torch._C._jit_try_infer_type(module_itr)
            if maybe_inferred_ts_type.success():
                python_type = _torchscript_type_to_python_type(
                    maybe_inferred_ts_type.type()
                )
                attr_proxy.node.type = (
                    python_type if not attr_proxy.node.type else attr_proxy.node.type
                )

        return attr_proxy

    def _extract_python_return_type(self, target: Target) -> Any | None:
        # [FUNCTION DOC] HELPER: Returns the return annotation of the callable target, or
        # None if its signature cannot be inspected or has no annotation. Raises
        # AssertionError if target is not callable.
        """
        Given a Python call target, try to extract the Python return annotation
        if it is available, otherwise return None

        Args:

            target (Callable): Python callable to get return annotation for

        Returns:

            Optional[Any]: Return annotation from the `target`, or None if it was
                not available.
        """
        if not callable(target):
            raise AssertionError(f"Expected callable target, got {type(target)}")
        try:
            sig = inspect.signature(target)
        except (ValueError, TypeError):
            return None

        return (
            sig.return_annotation
            if sig.return_annotation is not inspect.Signature.empty
            else None
        )
