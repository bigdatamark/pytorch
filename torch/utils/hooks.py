# mypy: allow-untyped-defs
# [MODULE DOC] Hook infrastructure shared by tensors and nn.Module: RemovableHandle (weakly
# references hook dictionaries so a hook can be removed), unserializable_hook (marks hooks
# that should not trigger a pickling warning), warn_if_has_hooks, and BackwardHook (adapts
# module full backward hooks and pre-hooks onto autograd). No import-time side effects.
import torch
from collections import OrderedDict
import weakref
import warnings
from typing import Any

__all__ = ["RemovableHandle", "unserializable_hook", "warn_if_has_hooks", "BackwardHook"]

class RemovableHandle:
    # [CLASS DOC] Handle returned when registering a hook; calling remove (or leaving a with
    # block) deletes the hook entry by its unique integer id from the hook dictionary and
    # any extra dictionaries. State: weak references to those dictionaries and the id, taken
    # from a class-wide counter that only grows. Supports pickling via __getstate__/__setstate__.
    r"""
    A handle which provides the capability to remove a hook.

    Args:
        hooks_dict (dict): A dictionary of hooks, indexed by hook ``id``.
        extra_dict (Union[dict, List[dict]]): An additional dictionary or list of
            dictionaries whose keys will be deleted when the same keys are
            removed from ``hooks_dict``.
    """

    id: int
    next_id: int = 0

    def __init__(self, hooks_dict: Any, *, extra_dict: Any = None) -> None:
        # [FUNCTION DOC] Store a weak reference to hooks_dict, take the next global id (and
        # increment the class counter), and keep weak references to extra_dict, which may be
        # one dictionary or a list of dictionaries; any other value means no extra dictionaries.
        self.hooks_dict_ref = weakref.ref(hooks_dict)
        self.id = RemovableHandle.next_id
        RemovableHandle.next_id += 1

        self.extra_dict_ref: tuple = ()
        if isinstance(extra_dict, dict):
            self.extra_dict_ref = (weakref.ref(extra_dict),)
        elif isinstance(extra_dict, list):
            self.extra_dict_ref = tuple(weakref.ref(d) for d in extra_dict)

    def remove(self) -> None:
        # [FUNCTION DOC] Delete this handle's id from the hooks dictionary and from each extra
        # dictionary, skipping any that were garbage collected or lack the id. Safe to repeat.
        hooks_dict = self.hooks_dict_ref()
        if hooks_dict is not None and self.id in hooks_dict:
            del hooks_dict[self.id]

        for ref in self.extra_dict_ref:
            extra_dict = ref()
            if extra_dict is not None and self.id in extra_dict:
                del extra_dict[self.id]

    def __getstate__(self):
        # [FUNCTION DOC] Pickle support: return a tuple of the (dereferenced) hooks dictionary
        # and id, plus a tuple of the extra dictionaries when the reference tuple is not None.
        # KNOWN ISSUE: extra_dict_ref is always a tuple, so the None branch is unreachable.
        if self.extra_dict_ref is None:
            return (self.hooks_dict_ref(), self.id)
        else:
            return (self.hooks_dict_ref(), self.id, tuple(ref() for ref in self.extra_dict_ref))

    def __setstate__(self, state) -> None:
        # [FUNCTION DOC] Restore from the state tuple: weakly reference the hooks dictionary (a
        # dead reference if it was None), restore the id and raise the class counter above it,
        # and rebuild weak references to extra dictionaries when present in the state.
        if state[0] is None:
            # create a dead reference
            self.hooks_dict_ref = weakref.ref(OrderedDict())
        else:
            self.hooks_dict_ref = weakref.ref(state[0])
        self.id = state[1]
        RemovableHandle.next_id = max(RemovableHandle.next_id, self.id + 1)

        if len(state) < 3 or state[2] is None:
            self.extra_dict_ref = ()
        else:
            self.extra_dict_ref = tuple(weakref.ref(d) for d in state[2])

    def __enter__(self) -> "RemovableHandle":
        # [FUNCTION DOC] Context manager entry; returns the handle itself.
        return self

    def __exit__(self, type: Any, value: Any, tb: Any) -> None:
        # [FUNCTION DOC] Context manager exit; removes the hook. Exceptions are not suppressed.
        self.remove()


def unserializable_hook(f):
    # [FUNCTION DOC] Decorator that sets a marker attribute on function f (mutating it) so
    # warn_if_has_hooks does not warn about it; returns f itself.
    """
    Mark a function as an unserializable hook with this decorator.

    This suppresses warnings that would otherwise arise if you attempt
    to serialize a tensor that has a hook.
    """
    f.__torch_unserializable__ = True
    return f


def warn_if_has_hooks(tensor) -> None:
    # [FUNCTION DOC] Emit a warning for each backward hook on the tensor that is not marked
    # as unserializable, saying it will not be serialized. Returns nothing.
    if tensor._backward_hooks:
        for k in tensor._backward_hooks:
            hook = tensor._backward_hooks[k]
            if not hasattr(hook, "__torch_unserializable__"):
                warnings.warn(f"backward hook {repr(hook)} on tensor will not be "
                              "serialized.  If this is expected, you can "
                              "decorate the function with @torch.utils.hooks.unserializable_hook "
                              "to suppress this warning", stacklevel=2)

class BackwardHook:
    # [CLASS DOC] Per-module helper that implements full backward hooks and backward pre-hooks.
    # State: the module, lists of user hooks and pre-hooks, the grad_outputs captured at the
    # output node, and the count and tensor positions of inputs and outputs (-1/None until set
    # up). setup_input_hook and setup_output_hook wrap forward inputs/outputs with an identity
    # autograd function and register hooks on its node; the user hook fires once both
    # grad_output (captured at the output) and grad_input are known.
    """
    A wrapper class to implement nn.Module backward hooks.

    It handles:
      - Ignoring non-Tensor inputs and replacing them by None before calling the user hook
      - Generating the proper Node to capture a set of Tensor's gradients
      - Linking the gradients captures for the outputs with the gradients captured for the input
      - Calling the user hook once both output and input gradients are available
    """

    def __init__(self, module, user_hooks, user_pre_hooks) -> None:
        # [FUNCTION DOC] Store the module and the user hook and pre-hook collections and
        # initialize the captured gradient and index bookkeeping to empty values.
        self.user_hooks = user_hooks
        self.user_pre_hooks = user_pre_hooks
        self.module = module

        self.grad_outputs = None
        self.n_outputs = -1
        self.output_tensors_index = None
        self.n_inputs = -1
        self.input_tensors_index = None

    def _pack_with_none(self, indices, values, size):
        # [FUNCTION DOC] HELPER: Build a tuple of length size filled with None, placing each
        # value at its matching index. Raises ValueError if indices and values differ in length.
        res = [None] * size
        for idx, val in zip(indices, values, strict=True):
            res[idx] = val

        return tuple(res)

    def _unpack_none(self, indices, values):
        # [FUNCTION DOC] HELPER: Return a tuple of the entries of values at the given indices.
        res = [values[idx] for idx in indices]

        return tuple(res)

    def _set_user_hook(self, grad_fn) -> None:
        # [FUNCTION DOC] HELPER: Register on the input-side autograd node a hook that runs the
        # user hooks with the full grad_input and the saved grad_outputs.
        def hook(grad_input, _):
            # [FUNCTION DOC] HELPER: Autograd hook. Does nothing if no grad_outputs were saved
            # (e.g. double backward). Otherwise expands grad_input to the full input positions,
            # calls each user hook in order; a non-None result replaces the gradients, and raises
            # RuntimeError if its length differs. Clears saved grad_outputs and returns the
            # gradients restricted to tensor inputs.
            if self.grad_outputs is None:
                # This happens because the gradient in your nn.Module flows to
                # the Module's input without " passing through the Module's
                # output, e.g. when you're doing double backward.
                return
            res = self._pack_with_none(self.input_tensors_index, grad_input, self.n_inputs)

            for hook in self.user_hooks:
                out = hook(self.module, res, self.grad_outputs)

                if out is None:
                    continue

                if len(out) != len(res):
                    raise RuntimeError("Backward hook returned an invalid number of grad_input, "
                                       f"got {len(out)}, but expected {len(res)}")

                res = out

            self.grad_outputs = None

            return self._unpack_none(self.input_tensors_index, res)

        grad_fn.register_hook(hook)

    def _apply_on_tensors(self, fn, args):
        # [FUNCTION DOC] HELPER: Pass the tensors among args through the identity backward-hook
        # autograd function and call fn on the resulting node. Returns (new args, indices of the
        # tensor positions), or (args, None) if no tensor requires grad or grad mode is off.
        # Rebuilds args as the same tuple or namedtuple type. Raises RuntimeError if there are
        # no tensors or the expected node cannot be found.
        # Can be used to apply the given function to the tensors contained in the
        # args. Will return updated args and the tensors indices
        tensors_idx = []
        tensors = []

        requires_grad = False
        for i, arg in enumerate(args):
            if isinstance(arg, torch.Tensor):
                tensors_idx.append(i)
                tensors.append(arg)
                requires_grad |= arg.requires_grad

        if not (requires_grad and torch.is_grad_enabled()):
            return args, None

        new_tensors = torch.nn.modules._functions.BackwardHookFunction.apply(*tensors)
        if len(new_tensors) == 0:
            raise RuntimeError("Cannot set Module backward hook for a Module with no input Tensors.")

        grad_fns = [t.grad_fn for t in new_tensors if t.grad_fn is not None and t.grad_fn.name() == "BackwardHookFunctionBackward"]
        if len(grad_fns) == 0:
            raise RuntimeError("Error while setting up backward hooks. Please open "
                               "an issue with a code sample to reproduce this.")

        fn(grad_fns[0])

        arg_list = list(args)
        for idx, val in zip(tensors_idx, new_tensors, strict=True):
            arg_list[idx] = val

        if type(args) is tuple:
            out = tuple(arg_list)
        else:
            out = type(args)(*arg_list)
        return out, tensors_idx

    def setup_input_hook(self, args):
        # [FUNCTION DOC] Wrap the module's forward inputs so the user hook is registered on
        # their autograd node. Records the input count and tensor positions and returns the
        # wrapped args (same container type).
        def fn(grad_fn) -> None:
            # [FUNCTION DOC] HELPER: Register the user hook on the given node.
            self._set_user_hook(grad_fn)

        res, input_idx = self._apply_on_tensors(fn, args)
        self.n_inputs = len(args)
        self.input_tensors_index = input_idx
        return res

    def setup_output_hook(self, args):
        # [FUNCTION DOC] Wrap the module's forward outputs (a single value is treated as a
        # one-element tuple and unwrapped again) so a hook captures grad_output on their node.
        # Records the output count and tensor positions and returns the wrapped outputs.
        def fn(grad_fn) -> None:
            # [FUNCTION DOC] HELPER: Register the output gradient capture hook on the node.
            def hook(_, grad_output):
                # [FUNCTION DOC] HELPER: Autograd hook on the output node. Saves grad_output
                # expanded to full output positions, then runs pre-hooks (a non-None result
                # replaces it; raises RuntimeError on a length change). If no input requires
                # grad, warns and runs the full user hooks directly with empty grad_input (raises
                # RuntimeError if a hook returns anything but None) and clears the saved
                # grads. Returns the possibly modified grad_output for tensor positions.
                self.grad_outputs = self._pack_with_none(self.output_tensors_index,
                                                         grad_output,
                                                         self.n_outputs)

                if self.user_pre_hooks:
                    expected_len = len(self.grad_outputs)
                    for user_pre_hook in self.user_pre_hooks:
                        hook_grad_outputs = user_pre_hook(self.module, self.grad_outputs)
                        if hook_grad_outputs is None:
                            continue

                        actual_len = len(hook_grad_outputs)
                        if actual_len != expected_len:
                            raise RuntimeError("Backward pre hook returned an invalid number of grad_output, "
                                               f"got {actual_len}, but expected {expected_len}")
                        self.grad_outputs = hook_grad_outputs

                # We need to be able to clear self.grad_outputs but also return it
                local_grad_outputs = self.grad_outputs

                # Special case if no input required gradients, this hook should call the user
                # hook directly
                if self.input_tensors_index is None:
                    # Only full backward hooks (not pre-hooks) receive grad_input
                    # and are the subject of the warning; pre-hooks already ran
                    # above from grad_output. Skip both when none are registered.
                    if self.user_hooks:
                        warnings.warn("Full backward hook is firing when gradients are computed "
                                      "with respect to module outputs since no inputs require gradients. See "
                                      "https://docs.pytorch.org/docs/main/generated/torch.nn.Module.html#torch.nn.Module.register_full_backward_hook "
                                      "for more details.",
                                      stacklevel=5)
                        grad_inputs = self._pack_with_none([], [], self.n_inputs)
                        for user_hook in self.user_hooks:
                            res = user_hook(self.module, grad_inputs, self.grad_outputs)
                            if res is not None and not (isinstance(res, tuple) and all(el is None for el in res)):
                                raise RuntimeError("Backward hook for Modules where no input requires "
                                                   "gradient should always return None or None for all gradients.")
                    self.grad_outputs = None

                if local_grad_outputs is not None:
                    if self.output_tensors_index is None:
                        raise AssertionError("output_tensors_index should not be None when grad_outputs is not None")
                    return tuple(local_grad_outputs[i] for i in self.output_tensors_index)

            grad_fn.register_hook(hook)

        is_tuple = True
        if not isinstance(args, tuple):
            args = (args,)
            is_tuple = False

        res, output_idx = self._apply_on_tensors(fn, args)
        self.n_outputs = len(args)
        self.output_tensors_index = output_idx

        if not is_tuple:
            res = res[0]
        return res
