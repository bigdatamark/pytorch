# mypy: allow-untyped-defs
# [MODULE DOC] Provides ModuleTracker, a context manager that tracks which nn.Module
# (by qualified name) is executing in forward or backward using global module hooks and
# multi-grad hooks. Exports only ModuleTracker; creates a module logger at import.
import logging
import weakref
from typing import TYPE_CHECKING

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten


if TYPE_CHECKING:
    from torch.utils.hooks import RemovableHandle


logger = logging.getLogger(__name__)


__all__ = ["ModuleTracker"]


class ModuleTracker:
    # [CLASS DOC] Context manager that maintains the set parents of names of modules currently
    # running (always containing "Global"). On enter it installs global forward pre and post
    # hooks; for tensors requiring grad it adds autograd hooks to track backward as well. State:
    # weak maps of module names, per-run hook handles, and a flag for a pending engine callback.
    # On exit it removes all hooks. Not safe for concurrent backward calls.
    """
    ``ModuleTracker`` is a context manager that tracks the nn.Module hierarchy during execution
    so that other system can query which Module is currently being executed (or its backward is being
    executed).

    You can access the ``parents`` attribute on this context manager to get the set of all the
    Modules currently being executed via their fqn (fully qualified name, also used as the key within
    the state_dict).
    You can access the ``is_bw`` attribute to know if you are currently running in backward or not.

    Note that ``parents`` is never empty and always contains the "Global" key. The ``is_bw`` flag
    will remain ``True`` after the forward until another Module is executed. If you need it to be
    more accurate, please submit an issue requesting this. Adding a map from fqn to the module instance
    is possible but not done yet, please submit an issue requesting this if you need it.

    Example usage

    .. code-block:: python

        mod = torch.nn.Linear(2, 2)

        with ModuleTracker() as tracker:
            # Access anything during the forward pass
            def my_linear(m1, m2, bias):
                print(f"Current modules: {tracker.parents}")
                return torch.mm(m1, m2.t()) + bias

            torch.nn.functional.linear = my_linear

            mod(torch.rand(2, 2))

    """

    parents: set[str]
    """
    A Set containing the fqn for each module currently running their forward
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize parents to contain Global and empty name caches and hook list.
        self.parents = {"Global"}
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()
        self._has_callback = False
        self._hooks: list[RemovableHandle] = []

    def _maybe_set_engine_callback(self) -> None:
        # [FUNCTION DOC] HELPER: Once per backward pass, queue an autograd engine callback that
        # resets parents to only Global when the pass ends. No-op if one is already queued.
        # This assumes no concurrent calls to backward
        if self._has_callback:
            return

        def callback() -> None:
            # [FUNCTION DOC] HELPER: Reset parents to Global and clear the pending callback flag.
            self.parents = {"Global"}
            self._has_callback = False

        torch.autograd.Variable._execution_engine.queue_callback(callback)
        self._has_callback = True

    @property
    def is_bw(self):
        # [FUNCTION DOC] Property: True when called while an autograd backward task is running.
        """
        A boolean marking if this is currently running during the backward pass or not
        """
        return torch._C._current_graph_task_id() != -1

    def _get_mod_name(self, mod):
        # [FUNCTION DOC] HELPER: Return the cached name of mod (its class name if first seen). On
        # first visit, record dotted child names (parent name, dot, attribute) for all submodules.
        if mod not in self._known_modules:
            self._known_modules[mod] = type(mod).__name__
        mod_name = self._known_modules[mod]
        if mod not in self._seen_modules:
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
                self._get_mod_name(submod)
            self._seen_modules.add(mod)
        return mod_name

    def _get_append_fn(self, name, is_bw):
        # [FUNCTION DOC] HELPER: Build a hook function that marks name as entered.
        def fn(*args) -> None:
            # [FUNCTION DOC] HELPER: In backward, ensure the reset callback is queued; log at info
            # level if name is already entered; add name to parents. Arguments are ignored.
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents:
                logger.info(
                    "The module hierarchy tracking seems to be broken as this Module was already entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )
            self.parents.add(name)

        return fn

    def _get_pop_fn(self, name, is_bw):
        # [FUNCTION DOC] HELPER: Build a hook function that marks name as exited.
        def fn(*args) -> None:
            # [FUNCTION DOC] HELPER: Remove name from parents, or log at info level if it is not
            # there. Arguments are ignored.
            if name in self.parents:
                self.parents.remove(name)
            else:
                logger.info(
                    "The Module hierarchy tracking is confused as we're exiting a Module that was never entered. %s during %s",
                    name,
                    "backward" if is_bw else "forward",
                )

        return fn

    def _fw_pre_hook(self, mod, input) -> None:
        # [FUNCTION DOC] HELPER: Global forward pre-hook. Marks mod entered for forward and
        # registers a multi-grad hook on its grad-requiring input tensors that marks it exited
        # for backward (handle stored for cleanup).
        name = self._get_mod_name(mod)
        self._get_append_fn(name, False)()

        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            self._hooks.append(
                register_multi_grad_hook(tensors, self._get_pop_fn(name, True))
            )

    def _fw_post_hook(self, mod, input, output) -> None:
        # [FUNCTION DOC] HELPER: Global forward hook. Marks mod exited for forward and registers
        # a multi-grad hook on its grad-requiring output tensors that marks it entered for
        # backward.
        name = self._get_mod_name(mod)
        self._get_pop_fn(name, False)()

        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if tensors:
            self._hooks.append(
                register_multi_grad_hook(tensors, self._get_append_fn(name, True))
            )

    def __enter__(self):
        # [FUNCTION DOC] Register the global forward pre and post hooks; returns self.
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(self._fw_post_hook)
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] Remove the global hooks and all autograd hooks added during the run.
        # Does not reset parents or suppress exceptions.
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
        for hook in self._hooks:
            hook.remove()
        self._hooks.clear()
