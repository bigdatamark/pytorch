# mypy: ignore-errors
# [MODULE DOC] opcheck check that a function run through AOTAutograd (functorch
# compiled_function with no-op compilers) matches eager mode in outputs and gradients.
# Contains aot_autograd_check, the forward/backward comparison helper, input cloning
# helpers, and an assert_raises_regex context manager. Defines the module level message
# outputs_msg. No import-time side effects.

import re

import torch
import torch.utils._pytree as pytree
from torch._dynamo.utils import copy_dynamo_tensor_attributes
from functorch.compile import compiled_function, min_cut_rematerialization_partition, default_partition, nop
from torch.testing._utils import wrapper_set_seed
from torch.utils._python_dispatch import is_traceable_wrapper_subclass

from .make_fx import randomize


def _clone_input_for_aot_autograd(x):
    # [FUNCTION DOC] HELPER: Returns a clone of tensor x; a pinned cpu tensor (not a
    # traceable wrapper subclass) gets its clone pinned too. Copies dynamo tensor
    # attributes from x onto the clone.
    result = torch.clone(x)
    if (
        x.device.type == "cpu"
        and not is_traceable_wrapper_subclass(x)
        and x.is_pinned()
    ):
        result = result.pin_memory()
    copy_dynamo_tensor_attributes(x, result)
    return result


def _maybe_copy_inputs_for_aot_autograd(args, copy_inputs):
    # [FUNCTION DOC] HELPER: If copy_inputs is true returns args with every tensor cloned
    # (pytree map); otherwise returns args unchanged.
    if copy_inputs:
        return pytree.tree_map_only(torch.Tensor, _clone_input_for_aot_autograd, args)
    return args


class assert_raises_regex:
    # [CLASS DOC] Context manager asserting that the body raises exactly the given
    # exception class with a message matching a regular expression. State: exception_cls
    # and regex. The expected exception is suppressed on exit.
    def __init__(self, exception_cls, regex):
        # [FUNCTION DOC] Stores the expected exception class and the regex pattern.
        self.exception_cls = exception_cls
        self.regex = regex

    def __enter__(self):
        # [FUNCTION DOC] Does nothing; returns None.
        pass

    def __exit__(self, exc_type, exc_val, traceback):
        # [FUNCTION DOC] Returns True (suppressing the exception) if the exception type
        # equals exception_cls and its message matches the regex. Raises AssertionError if
        # the message does not match, if a different exception type occurred, or if no
        # exception occurred.
        # KNOWN ISSUE: the type check is exact equality, so subclasses of exception_cls are
        # reported as a wrong exception.
        if exc_type == self.exception_cls:
            msg = str(exc_val)
            if not re.search(self.regex, msg):
                raise AssertionError(
                    f"Expected exception to match regex. regex: {self.regex}, exception: {msg}")
            return True  # Squashes the exception
        if exc_type is not None:
            raise AssertionError(
                f"Expected {self.exception_cls} to be raised, instead got exception {exc_type}")
        raise AssertionError("Expected exception to be raised but none was")


def aot_autograd_check(
        func,
        args,
        kwargs,
        dynamic,
        assert_raises_regex_fn=assert_raises_regex,
        assert_equals_fn=torch.testing.assert_close,
        check_gradients=True,
        try_check_data_specialization=False,
        skip_correctness_check=False,
        disable_functionalization=False,
        copy_inputs=False):
    # [FUNCTION DOC] Checks func called with args and kwargs behaves the same under
    # AOTAutograd as in eager mode. Tensors are pulled out of the flattened args and the
    # function is rewrapped to take only a list of tensors. Compiles with no-op compilers
    # (min-cut partitioner, or default partitioner when disable_functionalization is set).
    # check_gradients may be true, false, or "auto" (true only if some input and some
    # output require grad). Without gradient checking, outputs are compared via
    # assert_equals_fn unless skip_correctness_check; with it, delegates to the
    # forwards/backwards helper (try_check_data_specialization applies only there).
    # Raises on mismatch; both runs use a fixed seed. Returns nothing.
    """Compares func(*args, **kwargs) in eager-mode to under AOTAutograd.

    Compares outputs and (if check_gradients=True) gradients produced by
    AOTAutograd against eager-mode PyTorch.

    We assume that func(*args, **kwargs) succeeds in eager-mode PyTorch.

    """
    flat_args, args_spec = pytree.tree_flatten((args, kwargs))
    args = [arg for arg in flat_args if isinstance(arg, torch.Tensor)]

    # We construct a new function that only accepts Tensors as inputs
    def func_no_tensors(args):
        # [FUNCTION DOC] HELPER: Rebuilds the original args and kwargs by substituting the
        # given tensor list, in order, for the tensor positions of the flattened inputs,
        # then calls func and returns its result.
        reconstructed_flat_args = []
        args = iter(args)
        for v in flat_args:
            if isinstance(v, torch.Tensor):
                reconstructed_flat_args.append(next(args))
            else:
                reconstructed_flat_args.append(v)

        c_args, c_kwargs = pytree.tree_unflatten(reconstructed_flat_args, args_spec)
        return func(*c_args, **c_kwargs)

    # cannot use the min cut partitioner without functionalization
    if disable_functionalization:
        compiled_f = compiled_function(
            func_no_tensors,
            nop,
            nop,
            dynamic=dynamic,
            partition_fn=default_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=True
        )
    else:
        compiled_f = compiled_function(
            func_no_tensors,
            nop,
            nop,
            dynamic=dynamic,
            partition_fn=min_cut_rematerialization_partition,
            keep_inference_input_mutations=True,
            disable_functionalization=False
        )

    out = wrapper_set_seed(
        func_no_tensors, _maybe_copy_inputs_for_aot_autograd(args, copy_inputs)
    )
    if check_gradients == "auto":
        any_tensor_requires_grad = pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, args)
        any_output_requires_grad = pytree.tree_any_only(torch.Tensor, lambda x: x.requires_grad, out)
        check_gradients = any_tensor_requires_grad and any_output_requires_grad
    if not check_gradients:
        compiled_out = wrapper_set_seed(
            compiled_f, _maybe_copy_inputs_for_aot_autograd(args, copy_inputs)
        )
        if not skip_correctness_check:
            assert_equals_fn(compiled_out, out, msg=outputs_msg)
        return
    _test_aot_autograd_forwards_backwards_helper(
        func_no_tensors, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization, skip_correctness_check, copy_inputs)

outputs_msg = (
    "Outputs of the operator are different in eager-mode PyTorch vs "
    "AOTDispatcher tracing. This means the operator will have incorrect output "
    "underneath torch.compile. This could be because the operator's "
    "implementation not traceable."
)


def _test_aot_autograd_forwards_backwards_helper(
        f, compiled_f, args, assert_raises_regex_fn, assert_equals_fn,
        try_check_data_specialization, skip_correctness_check=False,
        copy_inputs=False):
    # [FUNCTION DOC] HELPER: Compares outputs and gradients of f and its compiled version
    # on args. If eager gradients are all None while some input is a non-leaf, expects the
    # compiled call to fail with a "does not require grad" runtime error instead. On
    # mismatch re-raises the assertion error type with an explanatory message. When
    # try_check_data_specialization is set, repeats on randomized data, ignoring failures.
    # Verify grads are equal between compiled and non-compiled versions of f.

    def call_forwards_backwards(f, args):
        # [FUNCTION DOC] HELPER: Runs f on args with a fixed seed, sums the absolute value
        # of the sums of all tensor outputs, and returns the outputs together with the
        # gradients of that scalar for the inputs requiring grad (unused allowed).
        # Raises AssertionError if no tensor output exists.
        flat_args = pytree.arg_tree_leaves(*args)
        diff_args = [arg for arg in flat_args if isinstance(arg, torch.Tensor) and
                     arg.requires_grad]
        out = wrapper_set_seed(f, args)
        flat_out = pytree.tree_leaves(out)

        sm = 0
        for i in flat_out:
            if isinstance(i, torch.Tensor):
                # We need to call .abs() because it is possible that the output of the
                # operator is a complex Tensor and autograd will yell at autograd.grad
                # on a complex Tensor unless we manually provide the grad_output flag.
                sm += i.sum().abs()
        if not isinstance(sm, torch.Tensor):
            raise AssertionError(f"Expected sm to be a Tensor, got {type(sm)}")
        return out, torch.autograd.grad(sm, diff_args, allow_unused=True)

    def check(args, ignore_failure=False):
        # [FUNCTION DOC] HELPER: Runs the eager and compiled forward/backward on (copies of)
        # args and compares outputs and gradients unless skip_correctness_check. If the
        # eager run fails and ignore_failure is set, returns silently; otherwise re-raises.
        try:
            orig_args = _maybe_copy_inputs_for_aot_autograd(args, copy_inputs)
            orig_out, orig_grad = call_forwards_backwards(f, orig_args)
        except Exception:
            if ignore_failure:
                return
            raise

        # See https://github.com/pytorch/pytorch/pull/98960#issuecomment-1505962215
        tensor_args = [x for x in pytree.tree_flatten(orig_args)[0] if isinstance(x, torch.Tensor)]
        any_non_leaves = any(x.grad_fn is not None for x in tensor_args)
        if all(x is None for x in orig_grad) and any_non_leaves:
            with assert_raises_regex_fn(RuntimeError, 'does not require grad and does not have a grad_fn'):
                call_forwards_backwards(
                    compiled_f, _maybe_copy_inputs_for_aot_autograd(args, copy_inputs)
                )
            return

        msg = (
            "Gradients of the operator are different in eager-mode PyTorch vs "
            "AOTDispatcher. This means the operator will have incorrect gradients "
            "underneath torch.compile. This could be because the operator's "
            "backward is incorrectly registered or not traceable."
        )

        compiled_out, compiled_grad = call_forwards_backwards(
            compiled_f, _maybe_copy_inputs_for_aot_autograd(args, copy_inputs)
        )
        if not skip_correctness_check:
            try:
                assert_equals_fn(compiled_out, orig_out)
            except Exception as e:
                raise type(e)(outputs_msg) from e
            try:
                assert_equals_fn(compiled_grad, orig_grad)
            except Exception as e:
                raise type(e)(msg) from e

    check(args, ignore_failure=False)

    # Randomize the data and run the traced graph with it, to catch bugs
    # where we may have baked in Tensor data into the trace.
    # This is not guaranteed to succeed, because `f` might have preconditions
    # on the values of the inputs, so we just ignore if this test fails.
    if try_check_data_specialization:
        args = randomize(args)
        check(args, ignore_failure=True)
