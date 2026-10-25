# [MODULE DOC]
# Dynamo variable trackers for torch.distributed objects: the base
# DistributedVariable (availability check plus wrapped value), the world
# metaclass tracker, and BackwardHookVariable for module-level backward hooks
# under compiled autograd. Also helpers recognizing DTensor.from_local and
# constant process-group functions. No import-time side effects.
"""
Distributed computing variable tracking classes for PyTorch Dynamo.

This module implements variable tracking for distributed computing components:
- Process Groups (for collective communication)
- Device Meshes (for distributed tensor sharding)
- Placement Types (for specifying distribution strategies)
- Distributed Tensors and their operations
- Backward hooks for distributed module operations

These classes are responsible for tracking distributed operations during graph
compilation while maintaining proper guards and handling distributed-specific
behaviors. They ensure correct handling of distributed components like process
groups, device meshes, and placement strategies while preserving proper semantics
for distributed tensor operations in the compiled code.

The implementation provides special handling for distributed package availability
checks and proper tracking of distributed state and operations across processes.
"""

import functools
import inspect
from typing import Any, TYPE_CHECKING

import torch
from torch.fx.experimental._backward_state import BackwardState

from .. import compiled_autograd
from .._trace_wrapped_higher_order_op import trace_wrapped
from ..exc import unimplemented
from ..external_utils import call_module_hooks_from_backward_state
from ..guards import GuardBuilder, install_guard
from ..source import AttrSource
from .base import GetSet, Method, VariableTracker


if TYPE_CHECKING:
    from torch._dynamo.symbolic_convert import InstructionTranslatorBase


class DistributedVariable(VariableTracker):
    # [CLASS DOC]
    # Base tracker for distributed objects. Holds value (the wrapped real
    # object); construction graph breaks if torch.distributed is unavailable.
    # Hash is the real object's hash; comparison is generic object comparison.
    """
    The base distributed variable that encapsulates common methods
    for the distributed objects (i.e. ProcessGroup, DeviceMesh, etc.).
    Concrete distributed objects could inherit this class and add object
    specific logic.

    i.e. It provides the check on the distributed package existence
    and hold the tracking value for the corresponding distributed object.
    """

    def __init__(self, value: Any, **kwargs: Any) -> None:
        # [FUNCTION DOC] Stores the wrapped object; graph breaks (unsupported)
        # with a hint to build with distributed support if the
        # torch.distributed package is unavailable.
        super().__init__(**kwargs)
        if not DistributedVariable.is_available():
            unimplemented(
                gb_type="torch.distributed package is not available!",
                context="",
                explanation="The PyTorch package doesn't include torch.distributed when building from source.",
                hints=[
                    "Set USE_DISTRIBUTED=1 to enable it when building PyTorch from source."
                ],
            )
        self.value = value

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the type of the wrapped object.
        return type(self.value)

    @staticmethod
    def is_available() -> bool:
        # [FUNCTION DOC] Returns whether torch.distributed is available.
        # check if the distributed package is available or not
        return torch.distributed.is_available()

    def hash_impl(self, tx: "InstructionTranslatorBase") -> tuple[int, bool]:
        # [FUNCTION DOC] Returns the real object's hash and False (second
        # element flags no side-effecting user hash).
        return hash(self.value), False

    def tp_richcompare_impl(self, tx, other, op):
        # [FUNCTION DOC] Delegates to the generic object rich comparison.
        from .object_protocol import object_richcompare

        return object_richcompare(self, tx, other, op)


def is_from_local(value: object) -> bool:
    # [FUNCTION DOC] True if value is exactly the DTensor from_local function;
    # False when distributed is unavailable.
    if not DistributedVariable.is_available():
        return False
    from torch.distributed.tensor import DTensor

    return inspect.isfunction(value) and value is DTensor.from_local


def is_constant_pg_functions(value: object) -> bool:
    # [FUNCTION DOC] True if value is a plain function in a fixed list of
    # process-group query helpers (group size by name, group tag, rank not in
    # group, group ranks, resolve group name) treated as constant-foldable;
    # False when distributed is unavailable.
    if not DistributedVariable.is_available():
        return False

    from torch.distributed.distributed_c10d import (
        _get_group_size_by_name,
        _get_group_tag,
        _rank_not_in_group,
        _resolve_group_name_by_ranks_and_tag,
        get_process_group_ranks,
    )

    constant_processgroup_functions = [
        _get_group_size_by_name,
        _get_group_tag,
        _rank_not_in_group,
        get_process_group_ranks,
        _resolve_group_name_by_ranks_and_tag,
    ]

    return inspect.isfunction(value) and value in constant_processgroup_functions


class WorldMetaClassVariable(DistributedVariable):
    # [CLASS DOC]
    # Tracker for GroupMember and group classes (instances of the world
    # metaclass). Exposes the WORLD and NON_GROUP_MEMBER attributes via
    # getset entries that install identity guards on them; requires a source.
    """
    Tracks torch.distributed.GroupMember and torch.distributed.group, which are
    instances of the metaclass _WorldMeta.
    """

    @classmethod
    def is_group_member_type(cls, value: object) -> bool:
        # [FUNCTION DOC] True if the exact type of value is the world
        # metaclass; False when distributed is unavailable.
        if not cls.is_available():
            return False

        from torch.distributed.distributed_c10d import _WorldMeta

        return type(value) is _WorldMeta

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the type of the wrapped object.
        return type(self.value)

    # dynamo-internal getset (not real CPython descriptors): _WorldMeta
    # exposes the WORLD and NON_GROUP_MEMBER process-group sentinels.
    def _world_getset(self, tx: "InstructionTranslatorBase") -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for WORLD: installs an identity guard
        # on the attribute source and returns a tracker for the default world
        # group. Raises AssertionError if this tracker has no source.
        if not self.source:
            raise AssertionError(
                "WorldMetaClassVariable requires a source for WORLD attribute"
            )
        source = AttrSource(base=self.source, member="WORLD")
        install_guard(source.make_guard(GuardBuilder.ID_MATCH))
        return VariableTracker.build(tx, self.value.WORLD, source)

    def _non_group_member_getset(
        self, tx: "InstructionTranslatorBase"
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Getter for NON_GROUP_MEMBER: installs an
        # identity guard on the attribute source and returns a tracker for the
        # sentinel. Raises AssertionError if this tracker has no source.
        if not self.source:
            raise AssertionError(
                "WorldMetaClassVariable requires a source for NON_GROUP_MEMBER attribute"
            )
        source = AttrSource(base=self.source, member="NON_GROUP_MEMBER")
        install_guard(source.make_guard(GuardBuilder.ID_MATCH))
        return VariableTracker.build(tx, self.value.NON_GROUP_MEMBER, source)

    tp_getset = {
        "WORLD": GetSet(_world_getset, None),
        "NON_GROUP_MEMBER": GetSet(_non_group_member_getset, None),
    }


class BackwardHookVariable(VariableTracker):
    # [CLASS DOC]
    # Tracker for a module backward-hook helper object. State: proxy (graph
    # node producing the hook object), module, user_hooks and user_pre_hooks
    # trackers. Its input/output hook setup methods become graph method calls.
    # Created via the create factory, which requires compiled autograd.
    """
    Handles torch.utils.hooks.BackwardHook for module-level backward
    hooks.
    """

    @staticmethod
    def create(
        tx: "InstructionTranslatorBase",
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
    ) -> "BackwardHookVariable":
        # [FUNCTION DOC]
        # Factory: graph breaks unless compiled autograd is enabled. Registers
        # the module and the two hook collections as backward-state hooks on
        # the output graph, emits a graph call to a function that builds the
        # hook object from the backward state, sets its example value, and
        # returns a tracker wrapping the proxy.
        if not compiled_autograd.compiled_autograd_enabled:
            unimplemented(
                gb_type="Module-level backwards hooks require compiled autograd.",
                context="",
                explanation="",
                hints=[
                    "Enable compiled autograd by setting torch._dynamo.config.compiled_autograd = True."
                ],
            )

        def _in_graph_bw_hooks(
            bw_state: BackwardState,
        ) -> torch.utils.hooks.BackwardHook:
            # [FUNCTION DOC] HELPER: Graph-embedded function building a
            # backward hook object whose hooks call a traced wrapper that
            # dispatches to the user hooks and module stored in the given
            # backward state, by registered names.
            """
            Rather than installing the user hooks in the graph (which
            don't survive AotAutograd), we install hooks that will call
            trace_wrapped in the backward pass that CompiledAutograd
            can turn into actual hook calls.
            """
            return torch.utils.hooks.BackwardHook(
                None,
                (
                    functools.partial(
                        trace_wrapped,
                        fn=call_module_hooks_from_backward_state,
                        bw_state=bw_state,
                        hooks_name=user_hooks_name,
                        module_name=module_name,
                    ),
                ),
                (
                    functools.partial(
                        trace_wrapped,
                        fn=call_module_hooks_from_backward_state,
                        bw_state=bw_state,
                        hooks_name=user_pre_hooks_name,
                        module_name=module_name,
                    ),
                ),
            )

        module_name, bw_state_proxy = tx.output.add_backward_state_hook(module, "mod")
        user_pre_hooks_name, _ = tx.output.add_backward_state_hook(user_pre_hooks)
        user_hooks_name, _ = tx.output.add_backward_state_hook(user_hooks)
        proxy = tx.output.create_proxy(
            "call_function",
            _in_graph_bw_hooks,
            (bw_state_proxy,),
            {},
        )
        proxy.node.meta["example_value"] = torch.utils.hooks.BackwardHook(None, (), ())
        return BackwardHookVariable(proxy, module, user_hooks, user_pre_hooks)

    def __init__(
        self,
        proxy: torch.fx.Proxy,
        module: VariableTracker,
        user_hooks: VariableTracker,
        user_pre_hooks: VariableTracker,
        **options: Any,
    ) -> None:
        # [FUNCTION DOC] Stores the proxy and the module and hook trackers.
        super().__init__(**options)
        self.proxy = proxy
        self.module = module
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks

    def as_proxy(self) -> torch.fx.Proxy:
        # [FUNCTION DOC] Returns the graph proxy of the hook object.
        return self.proxy

    def python_type(self) -> type:
        # [FUNCTION DOC] Returns the backward hook class.
        return torch.utils.hooks.BackwardHook

    def setup_input_hook(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Emits a graph method call to the hook's input setup
        # method with the given args; returns the wrapped result tracker.
        return self._setup_hook(tx, "setup_input_hook", *args, **kwargs)

    def setup_output_hook(
        self,
        tx: "InstructionTranslatorBase",
        args: list[VariableTracker],
        kwargs: dict[str, VariableTracker],
    ) -> VariableTracker:
        # [FUNCTION DOC] Emits a graph method call to the hook's output setup
        # method with the given args; returns the wrapped result tracker.
        return self._setup_hook(tx, "setup_output_hook", *args, **kwargs)

    tp_methods = {
        "setup_input_hook": Method(setup_input_hook),
        "setup_output_hook": Method(setup_output_hook),
    }

    def _setup_hook(
        self,
        tx: "InstructionTranslatorBase",
        hook_method_name: str,
        args: VariableTracker,
    ) -> VariableTracker:
        # [FUNCTION DOC] HELPER: Creates a graph method-call node invoking the
        # named hook method on this hook proxy with the argument's proxy, and
        # wraps the resulting proxy in a tracker.
        # KNOWN ISSUE: the callers splat positional args and keyword args into
        # this method, whose signature takes a single argument tracker, so it
        # works only for exactly one positional argument and no keywords.
        from .builder import wrap_fx_proxy

        return wrap_fx_proxy(
            tx,
            tx.output.create_proxy(
                "call_method",
                hook_method_name,
                (self.as_proxy(), args.as_proxy()),
                {},
            ),
        )
