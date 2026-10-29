# mypy: allow-untyped-defs
# [MODULE DOC] Defines ModTracker, a context manager that tracks which nn.Module (by fully qualified name) is currently running forward or backward.
# It works by registering global module forward pre/post hooks plus autograd multi-grad hooks and engine callbacks.
# Import-time side effects: none; global hooks are only installed while a ModTracker is entered. Exports ModTracker.
import warnings
import weakref
from collections.abc import Callable

import torch
from torch.autograd.graph import register_multi_grad_hook
from torch.nn.modules.module import (
    register_module_forward_hook,
    register_module_forward_pre_hook,
)
from torch.utils._pytree import tree_flatten


__all__ = ["ModTracker"]


class ModTracker:
    # [CLASS DOC] Context manager exposing the set of currently executing module names (parents, always containing "Global") and whether execution is in backward (is_bw).
    # State: parents set, per-name active counts, weak maps of known/seen modules (name assignment), a flag for a queued engine callback, pending post-backward callbacks, and four optional user hooks (pre/post forward/backward).
    # Entering installs global forward hooks; exiting removes them. Hook handlers are skipped inside dynamo-optimized modules.
    """
    ``ModTracker`` is a context manager that tracks the nn.Module hierarchy during execution
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

        with ModTracker() as tracker:
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

    def __init__(self):
        # [FUNCTION DOC] Initializes tracking state: parents starts as the Global entry, empty counters, weak module-name maps, no engine callback, no pending callbacks, and no user hooks.
        self.parents = {"Global"}
        self._active_module_cnt = {}
        self._known_modules: weakref.WeakKeyDictionary = weakref.WeakKeyDictionary()
        self._seen_modules: weakref.WeakSet = weakref.WeakSet()
        self._has_callback = False
        self._post_bw_callbacks_to_enqueue: list[Callable] = []
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _maybe_set_engine_callback(self):
        # [FUNCTION DOC] HELPER: Ensures a backward-end cleanup is queued on the autograd engine for the current backward pass (assumes no concurrent backward).
        # If one is already queued, returns. Otherwise queues the pending post-backward callbacks in reverse order, clears that list, then queues a callback that resets parents to the Global entry and clears the flag, and sets the flag.
        # This assumes no concurrent calls to backward
        if self._has_callback:
            return

        for post_bw_callback in reversed(self._post_bw_callbacks_to_enqueue):
            torch.autograd.Variable._execution_engine.queue_callback(post_bw_callback)
        self._post_bw_callbacks_to_enqueue.clear()

        def callback():
            # [FUNCTION DOC] HELPER: Engine callback that resets parents to only Global and marks that no callback is pending.
            self.parents = {"Global"}
            self._has_callback = False

        torch.autograd.Variable._execution_engine.queue_callback(callback)
        self._has_callback = True

    @property
    def is_bw(self):
        # [FUNCTION DOC] Property: true when a backward graph task is currently executing (the current graph task id is not -1).
        """
        A boolean marking if this is currently running during the backward pass or not
        """
        return torch._C._current_graph_task_id() != -1

    def get_known_fqn(self, mod):
        # [FUNCTION DOC] Returns the recorded fully qualified name for a module, or None if the tracker has not seen it.
        """
        Return the fqn for the given module if it is known to the ``ModTracker``, otherwise ``None``.
        """
        return self._known_modules.get(mod, None)

    def register_user_hooks(
        self,
        pre_fw_hook: Callable | None = None,
        post_fw_hook: Callable | None = None,
        pre_bw_hook: Callable | None = None,
        post_bw_hook: Callable | None = None,
    ):
        # [FUNCTION DOC] Registers up to four user hooks (pre/post forward, pre/post backward) invoked around each tracked module; arguments may be None to leave a slot unchanged-empty.
        # Raises AssertionError if a non-None hook is supplied for a slot that already holds a hook (clear first).
        # Side effect: stores the hooks on the tracker. KNOWN ISSUE: passing None for a slot overwrites an existing hook in that slot with None, so registering only one hook clears the others.
        """
        Registers user-specified hooks to be called before/after the forward/backward pass for each
        module tracked by the ``ModTracker``. One or more can be ``None``.
        Args:
            pre_fw_hook (Callable, optional): A hook to be called before the forward pass for the
                module. It should have the following signature:
                pre_fw_hook (module, input) -> None
            post_fw_hook (Callable, optional): A hook to be called after the forward pass for the
                module. It should have the following signature:
                post_fw_hook (module, input, output) -> None
            pre_bw_hook (Callable, optional): A multi-grad hook to be called on all the outputs of
                the module that require gradients. It should have the following signature:
                pre_bw_hook (module, grad_output) -> None
            post_bw_hook (Callable, optional): A multi-grad hook to be called on all the inputs of
                the module that require gradients. It should have the following signature:
                post_bw_hook (module, grad_input) -> None
        Raises:
            AssertionError: If a new hook is provided when one is already registered.
        Note:
            If the module is not alive during the backward pass, the pre_bw_hook and post_bw_hook will
            receive None as the module argument.
            The module fqn will be present in the ``parents`` attribute when each of the hooks is called.
            Hooks are intended to be used as markers only not to modify the inputs/outputs.
        """

        def set_hook(hook, user_hook, hook_name):
            # [FUNCTION DOC] HELPER: Returns the new hook, or raises AssertionError if both the new and existing hooks are set.
            if hook is not None and user_hook is not None:
                raise AssertionError(
                    f"Only one {hook_name} can be registered at a time"
                    f" Clear the existing hook by calling ``clear_user_hooks`` before registering a new one"
                )
            return hook

        self._user_pre_fw_hook = set_hook(
            pre_fw_hook, self._user_pre_fw_hook, "pre_fw_hook"
        )
        self._user_post_fw_hook = set_hook(
            post_fw_hook, self._user_post_fw_hook, "post_fw_hook"
        )
        self._user_pre_bw_hook = set_hook(
            pre_bw_hook, self._user_pre_bw_hook, "pre_bw_hook"
        )
        self._user_post_bw_hook = set_hook(
            post_bw_hook, self._user_post_bw_hook, "post_bw_hook"
        )

    def clear_user_hooks(self):
        # [FUNCTION DOC] Resets all four user hooks to None.
        """
        Clears the user specified hooks registered with ``register_user_hooks``
        """
        self._user_pre_fw_hook = None
        self._user_post_fw_hook = None
        self._user_pre_bw_hook = None
        self._user_post_bw_hook = None

    def _get_mod_name(self, mod):
        # [FUNCTION DOC] HELPER: Returns the name for a module, defaulting to its class name, and on first visit recursively assigns dotted names (parent name plus child attribute) to all children.
        # Side effect: updates the known-module map and marks the module as seen.
        if mod not in self._known_modules:
            self._known_modules[mod] = type(mod).__name__
        mod_name = self._known_modules[mod]
        if mod not in self._seen_modules:
            for name, submod in mod.named_children():
                self._known_modules[submod] = f"{mod_name}.{name}"
                self._get_mod_name(submod)
            self._seen_modules.add(mod)
        return mod_name

    def _get_append_fn(self, w_mod, name, is_bw):
        # [FUNCTION DOC] HELPER: Builds a callback that marks a module as entered. In backward mode it also ensures the end-of-backward engine callback is queued.
        def fn(*args):
            # [FUNCTION DOC] HELPER: Pushes the module name: warns (and replaces the global warning formatter) if the name is already active outside backward; adds the name to parents or increments its count; in backward, then calls the user pre-backward hook with the live module (or None) and the arguments.
            if is_bw:
                self._maybe_set_engine_callback()
            if name in self.parents and not self.is_bw:

                def custom_formatwarning(msg, category, filename, lineno, line=None):
                    # [FUNCTION DOC] HELPER: Formats a warning as file, line, category name and message.
                    return f"{filename}:{lineno}: {category.__name__}: {msg} \n"

                # pyrefly: ignore [bad-assignment]
                warnings.formatwarning = custom_formatwarning
                warnings.warn(
                    "The module hierarchy tracking maybe be messed up."
                    " Please file a bug to PyTorch, if it is the case.",
                    stacklevel=2,
                )
            if name not in self.parents:
                self._active_module_cnt[name] = 1
                self.parents.add(name)
            else:
                self._active_module_cnt[name] += 1

            if self._user_pre_bw_hook is not None and is_bw:
                self._user_pre_bw_hook(w_mod(), args)

        return fn

    def _get_pop_fn(self, w_mod, name, is_bw):
        # [FUNCTION DOC] HELPER: Builds a callback that marks a module as exited (see inner function).
        def fn(*args):
            # [FUNCTION DOC] HELPER: In backward, first calls the user post-backward hook. Then decrements the module's active count and removes it from parents at zero. If the name is not active and not in backward, raises RuntimeError about broken hierarchy tracking.
            if self._user_post_bw_hook is not None and is_bw:
                self._user_post_bw_hook(w_mod(), args)
            if name in self.parents:
                self._active_module_cnt[name] -= 1
                if self._active_module_cnt[name] == 0:
                    self.parents.remove(name)
            elif not self.is_bw:
                # Due to some input/output not requiring gradients, we cannot enforce
                # proper nesting in backward
                raise RuntimeError(
                    "The Module hierarchy tracking is wrong. Report a bug to PyTorch"
                )

        return fn

    def _fw_pre_hook(self, mod, input):
        # [FUNCTION DOC] HELPER: Global forward pre-hook. Skips if inside a dynamo-optimized module. Pushes the module name, calls the user pre-forward hook, and (outside backward) arranges the matching backward pop: a multi-grad hook on input tensors that require grad, or else a queued post-backward callback.
        if torch._dynamo.eval_frame._is_in_optimized_module():
            return

        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        self._get_append_fn(w_mod, name, False)()
        if self._user_pre_fw_hook is not None:
            self._user_pre_fw_hook(mod, input)
        args, _ = tree_flatten(input)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw:
            if tensors:
                register_multi_grad_hook(tensors, self._get_pop_fn(w_mod, name, True))
            else:
                self._post_bw_callbacks_to_enqueue.append(
                    self._get_pop_fn(w_mod, name, True)
                )

    def _fw_post_hook(self, mod, input, output):
        # [FUNCTION DOC] HELPER: Global forward post-hook. Skips inside dynamo-optimized modules. Calls the user post-forward hook, pops the module name, and (outside backward) registers a multi-grad hook in any-mode on output tensors requiring grad that pushes the name when backward reaches the module.
        if torch._dynamo.eval_frame._is_in_optimized_module():
            return

        name = self._get_mod_name(mod)
        w_mod = weakref.ref(mod)
        if self._user_post_fw_hook is not None:
            self._user_post_fw_hook(mod, input, output)
        self._get_pop_fn(w_mod, name, False)()
        args, _ = tree_flatten(output)
        tensors = [a for a in args if isinstance(a, torch.Tensor) and a.requires_grad]
        if not self.is_bw and tensors:
            register_multi_grad_hook(
                tensors, self._get_append_fn(w_mod, name, True), mode="any"
            )

    def __enter__(self):
        # [FUNCTION DOC] Installs the global forward pre-hook and forward hook (always_call) and returns the tracker.
        self._fw_pre_handle = register_module_forward_pre_hook(self._fw_pre_hook)
        self._fw_post_handle = register_module_forward_hook(
            self._fw_post_hook, always_call=True
        )
        return self

    def __exit__(self, *args):
        # [FUNCTION DOC] Removes the global forward hooks installed on enter. Exceptions are not suppressed.
        self._fw_pre_handle.remove()
        self._fw_post_handle.remove()
