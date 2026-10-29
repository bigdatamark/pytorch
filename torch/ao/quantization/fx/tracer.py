# [MODULE DOC] FX tracer used by quantization prepare. Defines QuantizationTracer, a
# torch.fx Tracer that treats torch.nn / torch.ao.nn modules, fused modules and
# user-skipped modules as leaves, plus a scope context manager helper.
# No import-time side effects beyond imports.
from collections.abc import Callable

import torch
from torch.ao.nn.intrinsic import _FusedModule
from torch.fx._symbolic_trace import Tracer
from torch.fx.proxy import Scope


__all__ = [
    "QuantizationTracer",
]


class ScopeContextManager(torch.fx.proxy.ScopeContextManager):
    # [CLASS DOC] HELPER: FX scope context manager that builds the new Scope from a module
    # path and the module's type instead of taking a prebuilt Scope.
    def __init__(
        self, scope: Scope, current_module: torch.nn.Module, current_module_path: str
    ):
        # [FUNCTION DOC] HELPER: Calls the parent constructor with the given scope and a new
        # Scope made from current_module_path and the type of current_module.
        super().__init__(scope, Scope(current_module_path, type(current_module)))


class QuantizationTracer(Tracer):
    # [CLASS DOC] torch.fx Tracer for quantization. Holds lists of skipped module names and
    # skipped module classes (these are kept as leaf call_module nodes), a root Scope
    # with empty path and no module type, and enables stack trace recording unless
    # torch.fx.config disables stack trace emission. Key method: is_leaf_module.
    def __init__(
        self, skipped_module_names: list[str], skipped_module_classes: list[Callable]
    ):
        # [FUNCTION DOC] Initializes the base Tracer, stores the skipped module names and classes,
        # sets the top-level scope (empty path, module type None) and sets
        # record_stack_traces from the inverse of the fx do_not_emit_stack_traces config.
        super().__init__()
        self.skipped_module_names = skipped_module_names
        self.skipped_module_classes = skipped_module_classes
        # NB: initialized the module_type of top level module to None
        # we are assuming people won't configure the model with the type of top level
        # module here, since people can use "" for global config
        # We can change this if there is a use case that configures
        # qconfig using top level module type
        self.scope = Scope("", None)
        self.record_stack_traces = not torch.fx.config.do_not_emit_stack_traces

    def is_leaf_module(self, m: torch.nn.Module, module_qualified_name: str) -> bool:
        # [FUNCTION DOC] Returns True if module m must not be traced into. Leaf if: m is defined under
        # torch.nn or torch.ao.nn and is not a Sequential; or its qualified name is in the
        # skipped names; or its exact type is in the skipped classes; or it is a fused module.
        return (
            (
                (
                    m.__module__.startswith("torch.nn")
                    or m.__module__.startswith("torch.ao.nn")
                )
                and not isinstance(m, torch.nn.Sequential)
            )
            or module_qualified_name in self.skipped_module_names
            or type(m) in self.skipped_module_classes
            or isinstance(m, _FusedModule)
        )
