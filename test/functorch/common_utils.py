# [MODULE DOC] Shared helpers for the functorch test suite, mostly for vmap OpInfo tests.
# Provides reference loop implementations of one and two level batching, generators of batched
# inputs and batch dimension choices, comparison quantity generators between vmap and loops,
# small decorator factories (xfail, skip and conditional variants) that produce decoration
# records for operators, a decorator that attaches decorators to module infos, a tolerance
# override attacher, a context manager disabling the vmap fallback, helpers building graph
# modules for saved tensor hooks, and a context manager capturing generated code from
# structured trace logs. Import-time side effect: reads the environment variable that marks an
# fbcode run into the IS_FBCODE flag, and imports the additional op database module that must
# be importable from the test directory.
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import itertools
import logging
import os
import unittest
from collections import namedtuple
from contextlib import contextmanager

from functorch_additional_op_db import additional_op_db

import torch
import torch.utils._pytree as pytree
from functorch import vmap
from torch.testing._internal.common_device_type import toleranceOverride
from torch.testing._internal.common_methods_invocations import DecorateInfo, op_db
from torch.testing._internal.common_modules import module_db
from torch.testing._internal.opinfo.core import (
    sample_skips_and_xfails,
    SkipRule,
    XFailRule,
)


IS_FBCODE = os.getenv("FUNCTORCH_TEST_FBCODE") == "1"


def loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values):
    # [FUNCTION DOC] Reference implementation of vmap by explicit looping. Arguments: op,
    # in_dims structure matching the batched arguments, out_dim (an int or structure),
    # batch_size, batched arguments and keyword values. For each index, selects that slice
    # from every argument whose in dim is not None, calls the op, and flattens the output.
    # Then stacks per-output results along a new leading dimension, or when the out dim for an
    # output is None takes the first batch result. Returns outputs rebuilt into the original
    # output structure. Raises AssertionError if the arguments and in_dims structures differ
    # and ValueError if an output to be batched is not a tensor. With a batch size of zero the
    # output flattening variable is never set, so it fails.
    outs = []
    out_spec = None
    for idx in range(batch_size):
        flat_args, args_spec = pytree.tree_flatten(batched_args)
        flat_dims, dims_spec = pytree.tree_flatten(in_dims)
        if args_spec != dims_spec:
            raise AssertionError(f"args_spec {args_spec} != dims_spec {dims_spec}")
        new_args = [
            a.select(in_dim, idx) if in_dim is not None else a
            for a, in_dim in zip(flat_args, flat_dims)
        ]
        out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
        flat_out, out_spec = pytree.tree_flatten(out)
        outs.append(flat_out)

    # use the same out_dim for all outputs
    if isinstance(out_dim, int):
        flat_out_dim = [out_dim for _ in flat_out]
    else:
        flat_out_dim, _ = pytree.tree_flatten(out_dim)

    outs = zip(*outs)

    result = []
    for i, out_lst in enumerate(outs):
        if flat_out_dim[i] is not None:
            if not all(isinstance(x, torch.Tensor) for x in out_lst):
                raise ValueError(
                    f"vmap `{op}` must only return "
                    "Tensors. Did you mean to set out_dims= to None for output?"
                )
            result.append(torch.stack(out_lst))
        else:
            # not batched over, result should be the same for all batches
            result.append(out_lst[0])
    return pytree.tree_unflatten(result, out_spec)


# Like loop helper function but for 2 levels of vmap. If we need more levels than this, probably possible
# to generalize the loops function but it seemed too complicated for this
def loop2(
    op,
    in_dims1,
    in_dims2,
    out_dim1,
    out_dim2,
    batch_size1,
    batch_size2,
    *batched_args,
    **kwarg_values,
):
    # [FUNCTION DOC] Reference implementation of two nested levels of vmap by explicit loops.
    # Arguments: op, in dims for the outer and inner levels, out dims for both, both batch
    # sizes, plus batched arguments and keyword values. Raises AssertionError if the argument
    # structures disagree or the flattened dim lists have different lengths. Loops over outer
    # then inner indices selecting slices, calls the op, stacks inner results along the first
    # out dim, then stacks across the outer loop along the second out dim. Handles outputs
    # that are a single tensor or a sequence of tensors; returns the list of stacked outputs.
    # KNOWN ISSUE: the check for the loop output being a tensor is applied to a list, so it
    # can never be true; a single tensor output is handled in the first stage only.
    outs = []
    flat_args, args_spec = pytree.tree_flatten(batched_args)
    flat_dims1, dims_spec1 = pytree.tree_flatten(in_dims1)
    flat_dims2, dims_spec2 = pytree.tree_flatten(in_dims2)
    if args_spec != dims_spec1:
        raise AssertionError(f"args_spec {args_spec} != dims_spec1 {dims_spec1}")
    if args_spec != dims_spec2:
        raise AssertionError(f"args_spec {args_spec} != dims_spec2 {dims_spec2}")
    if len(flat_dims1) != len(flat_dims2):
        raise AssertionError(
            f"len(flat_dims1) {len(flat_dims1)} != len(flat_dims2) {len(flat_dims2)}"
        )
    for idx1 in range(batch_size1):
        out_split = []
        arg_split = [
            a.select(in_dim1, idx1) if in_dim1 is not None else a
            for a, in_dim1 in zip(flat_args, flat_dims1)
        ]
        for idx2 in range(batch_size2):
            new_args = [
                a.select(in_dim, idx2) if in_dim is not None else a
                for a, in_dim in zip(arg_split, flat_dims2)
            ]
            out = op(*pytree.tree_unflatten(new_args, args_spec), **kwarg_values)
            out_split.append(out)
        outs.append(out_split)

    loop_out = []
    for out_split in outs:
        if isinstance(out_split[0], torch.Tensor):
            loop_out.append(torch.stack(out_split, out_dim1))
        else:
            new_out = []
            for idx in range(len(out_split[0])):
                new_out.append(torch.stack([i[idx] for i in out_split], out_dim1))
            loop_out.append(new_out)

    new_out = []
    if isinstance(loop_out, torch.Tensor):
        new_out = torch.stack(loop_out, out_dim2)
    else:
        for idx in range(len(loop_out[0])):
            new_out.append(torch.stack([i[idx] for i in loop_out], out_dim2))
    return new_out


def is_valid_inplace_sample_input(sample_input, op, inplace_variant):
    # [FUNCTION DOC] Decides whether a sample input can be used to test an in-place variant.
    # Returns false if there is no in-place variant, if the sample broadcasts its input, or if
    # the input is not a tensor. Otherwise runs the out-of-place op on the sample and returns
    # true only when the output dtype equals the input dtype (an in-place op cannot change
    # dtype). Side effect: executes the op once.
    if inplace_variant is None:
        return False
    if sample_input.broadcasts_input:
        return False
    if not isinstance(sample_input.input, torch.Tensor):
        return False

    # Check if input's dtype matches the output's dtype
    args = (sample_input.input,) + sample_input.args
    kwargs = sample_input.kwargs
    output_dtype = op(*args, **kwargs).dtype
    return sample_input.input.dtype == output_dtype


# This is kind of dangerous, please think carefully before using it.
# Known risks:
# - the return better not be mutated so it's best to return immutable types
# (e.g. prefer tuples to list)
# - Don't hash tensors in a global context, that'll keep them around forever
def memoize(fn):
    # [FUNCTION DOC] Decorator that caches a function by its positional arguments in a private
    # mapping, which must be hashable. Returns a wrapper function; keyword arguments are not
    # supported. The cache lives as long as the wrapper, so cached values and any hashed
    # tensors stay alive. Returned values are shared between callers.
    memo = {}

    def wrapped(*args):
        # [FUNCTION DOC] HELPER: Wrapper that returns the cached result for the given
        # positional arguments, calling the wrapped function and storing its result on first
        # use.
        if args not in memo:
            memo[args] = fn(*args)
        return memo[args]

    return wrapped


# NB: This is O(2 ** num_tensors).
# num_tensors ranges from 1 to 10, with 2-4 being most common.
# Try not to extravagate it if you're modifying it.
@memoize
def get_bdim_choices(num_tensors):
    # [FUNCTION DOC] Returns the batch dimension choices to test for a given number of
    # tensors, as a tuple of tuples. Includes the all zeros choice and every combination of
    # last dimension or no batching, excluding the choice where nothing is batched. Result is
    # cached. Cost grows exponentially with the tensor count. Raises AssertionError if the
    # final generated choice is not the all None choice.
    choices = []

    # full of zeros
    choices.append((0,) * num_tensors)

    # All permutations of (-1, None)
    options = (-1, None)
    choices.extend(itertools.product(options, repeat=num_tensors))

    if choices[-1] != (None,) * num_tensors:
        raise AssertionError(
            f"Expected choices[-1] to be {(None,) * num_tensors}, got {choices[-1]}"
        )
    return tuple(choices[:-1])


# NB: This is O(2 ** num_tensors).
# num_tensors ranges from 1 to 10, with 2-4 being most common.
# Try not to extravagate it if you're modifying it.
def get_bdim_choices_batch_norm(
    num_tensors, _, running_mean=None, running_var=None, *args
):
    # [FUNCTION DOC] Same as the general batch dimension choices but specialised for batch
    # norm. Arguments: number of tensors, an ignored second parameter (the input), running
    # mean and running var, and extra arguments ignored. If either running statistic is
    # missing, the input is never batched and the remaining tensors take zero or last
    # dimension or None choices. Otherwise all zeros plus every combination in which the input
    # is not batched unless both statistics are batched. Drops the all None choice at the end
    # and raises AssertionError if the last choice is not all None.
    choices = []
    options = (-1, None)

    # instance norm turns these into unbatched 0 tensors, so we cannot batch the input if either is not specified
    if running_mean is None or running_var is None:
        choices.append((None,) + (0,) * (num_tensors - 1))
        for choice in itertools.product(options, repeat=num_tensors - 1):
            choices.append((None,) + choice)

    else:
        # running_mean and running_var are specified as tensors. Batch norm doesn't work if the input is batched but
        # running_mean/var are unbatched, so this tests all other cases
        choices.append((0,) * num_tensors)
        for choice in itertools.product(options, repeat=num_tensors):
            input_bdim = choice[0]
            running_mean_bdim = choice[1]
            running_var_bdim = choice[2]
            if input_bdim and (not running_mean_bdim or not running_var_bdim):
                continue
            choices.append(choice)

    if choices[-1] != (None,) * num_tensors:
        raise AssertionError(
            f"Expected choices[-1] to be {(None,) * num_tensors}, got {choices[-1]}"
        )
    return tuple(choices[:-1])


def add_batch_dim(arg, bdim, batch_size=3):
    # [FUNCTION DOC] Adds a batch dimension to a tensor. Arguments: arg, bdim (zero or
    # negative one) and batch_size. Raises AssertionError if bdim is anything else or arg is
    # not a tensor. For zero repeats the tensor along a new leading dimension; for negative
    # one expands a new trailing dimension and makes it contiguous. Returns a pair of the
    # batched tensor and the bdim.
    if bdim != 0 and bdim != -1:
        raise AssertionError(f"Expected bdim to be 0 or -1, got {bdim}")
    if not isinstance(arg, torch.Tensor):
        raise AssertionError(f"Expected arg to be a torch.Tensor, got {type(arg)}")
    if bdim == 0:
        shape = [1] * len(arg.shape)
        shape.insert(bdim, batch_size)
        return (arg.repeat(shape), bdim)
    if bdim == -1:
        arg = arg.unsqueeze(-1).expand(*arg.shape, batch_size).contiguous()
        return (arg, bdim)


def construct_in_dims(bdim_choice_for_tensors, is_tensors):
    # [FUNCTION DOC] Builds the in dims tuple for all flattened arguments: positional entries
    # for tensors are taken in order from the iterable of batch dimension choices, and
    # non-tensor arguments get None. Returns a tuple.
    result = []
    bdim = iter(bdim_choice_for_tensors)
    for is_tensor in is_tensors:
        if not is_tensor:
            result.append(None)
            continue
        result.append(next(bdim))
    return tuple(result)


def is_batch_norm_training(op_name, kwarg_values):
    # [FUNCTION DOC] Decides whether a batch norm style operator call runs in training mode.
    # Returns false for operators other than batch norm and instance norm. Looks for boolean
    # values among the keyword values: none means the default (true for instance norm, false
    # for batch norm); exactly one means that value; more than one raises AssertionError.
    batch_norm_fns = (
        "nn.functional.batch_norm",
        "nn.functional.instance_norm",
    )  # instance norm calls batch norm
    if op_name not in batch_norm_fns:
        return False

    # batch norm and instance norm require the value to be a plain bool
    default_training = (
        op_name == "nn.functional.instance_norm"
    )  # instance norm defaults to training, batch norm doesn't
    is_training = tuple(
        arg for arg in tuple(kwarg_values.values()) if isinstance(arg, bool)
    )
    if len(is_training) == 0:
        return default_training
    else:
        if len(is_training) != 1:
            raise AssertionError(
                f"Expected len(is_training) to be 1, got {len(is_training)}"
            )
        return is_training[0]


def generate_vmap_inputs(
    arg_values, kwarg_values, is_batch_norm_and_training=False, batch_size=2
):
    # [FUNCTION DOC] Generator of batched test inputs. Arguments: positional values, keyword
    # values, a flag for batch norm in training mode, and batch size. Flattens the arguments,
    # counts tensors, and returns immediately (yields nothing) if there is a single tensor
    # under batch norm training. For each batch dimension choice (batch norm specific when
    # flagged) builds batched arguments, in dims in the original structure, and yields a
    # triple of batched args, in dims and the unchanged keyword values.
    flat_args, arg_spec = pytree.tree_flatten(tuple(arg_values))
    is_tensors = [isinstance(a, torch.Tensor) for a in flat_args]
    num_tensors = sum(is_tensors)
    # For Batch Norm, if there's only an input, we can't
    # batch it since running_mean/var will be seen as unbatched tensors
    if num_tensors == 1 and is_batch_norm_and_training:
        return
    bdim_choices = (
        get_bdim_choices_batch_norm(num_tensors, *arg_values)
        if is_batch_norm_and_training
        else get_bdim_choices(num_tensors)
    )

    @memoize
    def get_batched_arg(arg, bdim):
        # [FUNCTION DOC] HELPER: Memoized function returning the tensor batched by
        # add_batch_dim with the enclosing batch size, so repeated use of one argument with
        # the same dimension reuses one tensor. Raises AssertionError if arg is not a tensor
        # or bdim is None.
        if not isinstance(arg, torch.Tensor):
            raise AssertionError(f"Expected arg to be a torch.Tensor, got {type(arg)}")
        if bdim is None:
            raise AssertionError("Expected bdim to not be None")
        result, _ = add_batch_dim(arg, bdim, batch_size)
        return result

    for bdim_choice in bdim_choices:
        flat_in_dims = construct_in_dims(bdim_choice, is_tensors)

        flat_batched_args = tuple(
            arg if in_dim is None else get_batched_arg(arg, in_dim)
            for arg, in_dim in zip(flat_args, flat_in_dims)
        )
        batched_args = pytree.tree_unflatten(flat_batched_args, arg_spec)
        in_dims = pytree.tree_unflatten(flat_in_dims, arg_spec)
        yield batched_args, in_dims, kwarg_values


def clone_if_tensor(x):
    # [FUNCTION DOC] Returns a clone of the value if it is a tensor, otherwise the value
    # itself.
    if isinstance(x, torch.Tensor):
        return x.clone()
    return x


# Helper function to compare output of `vmap` against the
# `for-loop` version.
def _compute_quantities_for_vmap_test(
    op,
    orig_batched_args,
    orig_kwarg_values,
    in_dims,
    out_dim,
    batch_size,
    compute_loop_out=True,
    clone_inputs=False,
):
    # [FUNCTION DOC] HELPER: Generator computing vmap results for an op. Optionally clones the
    # inputs before every evaluation. Computes the reference loop output unless disabled (then
    # None), then the vmap output, then a double vmap output with an extra dummy batched
    # argument and the op, together with the expected value built by adding a size one
    # dimension at position one to the single vmap output. Yields once a tuple of vmap output,
    # loop output, double vmap output and expected double vmap output.
    def maybe_clone_inputs():
        # [FUNCTION DOC] HELPER: Returns the batched arguments and keyword values, cloned
        # tensor by tensor if clone_inputs is set so that in-place ops do not corrupt the
        # originals, otherwise the originals.
        if clone_inputs:
            batched_args = pytree.tree_map(clone_if_tensor, orig_batched_args)
            kwarg_values = pytree.tree_map(clone_if_tensor, orig_kwarg_values)
            return batched_args, kwarg_values
        return orig_batched_args, orig_kwarg_values

    batched_args, kwarg_values = maybe_clone_inputs()

    if compute_loop_out:
        loop_out = loop(op, in_dims, out_dim, batch_size, *batched_args, **kwarg_values)
    else:
        loop_out = None

    # Used for debugging the resulting operations
    # from functorch import make_fx
    # def f(a):
    #     return op(a)
    # t = make_fx(vmap(f, in_dims=in_dims, out_dims=out_dim))(*batched_args, **kwarg_values)
    # print(in_dims, [arg.shape for arg in batched_args], kwarg_values)
    batched_args, kwarg_values = maybe_clone_inputs()
    batched_out = vmap(op, in_dims=in_dims, out_dims=out_dim)(
        *batched_args, **kwarg_values
    )

    # Tests case where we dispatch to a batching rule with no bdims
    # This should be handled by autogenerated plumbing. For vmap support
    # added via a manual plumbing you may need to handle this specially.
    def add_bdim_if_tensor(x):
        # [FUNCTION DOC] HELPER: Returns the tensor with a new size one dimension inserted at
        # position one; non-tensors pass through.
        if isinstance(x, torch.Tensor):
            return x.unsqueeze(1)
        return x

    def f(dummy, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Wrapper taking a dummy first argument that is ignored and
        # calling the op with the remaining arguments, used to exercise the nested vmap path
        # without batch dimensions.
        return op(*args, **kwargs)

    dummy = torch.ones(batch_size, 1)
    vmapvmap_expected = pytree.tree_map(add_bdim_if_tensor, batched_out)

    inner_in_dims = (0,) + pytree.tree_map(lambda x: None, in_dims)
    outer_in_dims = (0,) + in_dims
    batched_args, kwarg_values = maybe_clone_inputs()
    vmapvmap_output = vmap(
        vmap(f, inner_in_dims, out_dims=out_dim), outer_in_dims, out_dims=out_dim
    )(dummy, *batched_args, **kwarg_values)

    yield (batched_out, loop_out, vmapvmap_output, vmapvmap_expected)


# Function with more friendly return types
# compared to `_compute_quantities_for_vmap_test`
def compute_quantities_for_vmap_test(
    op,
    orig_batched_args,
    orig_kwarg_values,
    in_dims,
    out_dim=0,
    batch_size=2,
    compute_loop_out=True,
    clone_inputs=False,
):
    # [FUNCTION DOC] Friendlier generator wrapping the quantity helper. With the same inputs,
    # for each result tuple it yields two pairs: the vmap output with the loop output, and the
    # double vmap output with its expected value. Used by tests that compare each pair for
    # equality.
    for quantities in _compute_quantities_for_vmap_test(
        op,
        orig_batched_args,
        orig_kwarg_values,
        in_dims,
        out_dim,
        batch_size,
        compute_loop_out,
        clone_inputs,
    ):
        yield (quantities[0], quantities[1])
        yield (quantities[2], quantities[3])


def get_fallback_and_vmap_exhaustive(
    op,
    arg_values,
    kwarg_values,
    is_batch_norm_and_training=False,
    compute_loop_out=True,
):
    # [FUNCTION DOC] Generator for exhaustive testing against the vmap fallback. Computes the
    # unbatched op result once and expands it along a new leading dimension to size two. For
    # every generated batched input, yields pairs of vmap output with that expected value and
    # double vmap output with its expected value. Raises AssertionError if the helper
    # unexpectedly produced a loop output.
    out_dim = 0
    batch_size = 2

    def make_batched(t):
        # [FUNCTION DOC] HELPER: Expands a tensor with an extra leading dimension of the
        # enclosing batch size without copying; non-tensors pass through unchanged.
        if isinstance(t, torch.Tensor):
            shape = list(t.shape)
            shape.insert(out_dim, batch_size)
            return t.expand(*shape)
        return t

    # Inputs generated by `generate_vmap_inputs` just copy/expand the unbatched inputs
    # over the batched dimension. Thus we can compute the expected value once and just
    # expand it based on the `out_dim` and `batch_size`.
    expected_unbatched = op(*arg_values, **kwarg_values)
    expected_batched = pytree.tree_map(make_batched, expected_unbatched)
    generator = generate_vmap_inputs(
        arg_values, kwarg_values, is_batch_norm_and_training
    )
    for batched_args, in_dims, kwarg_values in generator:
        for quantities in _compute_quantities_for_vmap_test(
            op,
            batched_args,
            kwarg_values,
            in_dims,
            out_dim,
            batch_size,
            compute_loop_out=False,
        ):
            if quantities[1] is not None:
                raise AssertionError(
                    f"Expected quantities[1] to be None, got {quantities[1]}"
                )
            yield (quantities[0], expected_batched)
            yield (quantities[2], quantities[3])


def opinfo_in_dict(opinfo, d):
    # [FUNCTION DOC] Returns whether the operator info name, or the name joined with its
    # variant test name by a dot, is a key in the mapping.
    return (opinfo.name in d) or (f"{opinfo.name}.{opinfo.variant_test_name}" in d)


DecorateMeta = namedtuple(
    "DecorateMeta",
    [
        "op_name",
        "variant_name",
        "decorator",
        "device_type",
        "dtypes",
    ],
)


def decorate(
    op_name, variant_name="", *, decorator=None, device_type=None, dtypes=None
):
    # [FUNCTION DOC] Builds a decoration record for an operator. Arguments: op name, variant
    # name, keyword only decorator (required; AssertionError if missing), device type and
    # dtypes filters. Returns the named tuple holding these fields.
    if decorator is None:
        raise AssertionError("decorator must not be None")
    return DecorateMeta(
        op_name=op_name,
        variant_name=variant_name,
        decorator=decorator,
        device_type=device_type,
        dtypes=dtypes,
    )


def xfail(op_name, variant_name="", *, device_type=None, dtypes=None):
    # [FUNCTION DOC] Builds a decoration record that marks the matching operator tests as
    # expected failures using the unittest expected failure decorator, optionally limited to a
    # device type and dtypes.
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.expectedFailure,
        device_type=device_type,
        dtypes=dtypes,
    )


# fail_fn should be a callable that accepts a single SampleInput and returns True if failure
# is expected
def xfailIf(op_name, fail_fn, variant_name="", *, device_type=None, dtypes=None):
    # [FUNCTION DOC] Builds a decoration record that expects failure only for samples where
    # the user function returns true. Wraps the function in a sample level expected failure
    # rule that matches every op and device, since the record already filters those.
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=sample_skips_and_xfails(
            [
                XFailRule(
                    # op matching is already handled by DecorateMeta
                    op_match_fn=lambda device, op: True,
                    # device matching is already handled by DecorateMeta
                    sample_match_fn=lambda device, sample: fail_fn(sample),
                )
            ]
        ),
        device_type=device_type,
        dtypes=dtypes,
    )


def skip(op_name, variant_name="", *, device_type=None, dtypes=None):
    # [FUNCTION DOC] Builds a decoration record that skips the matching tests with the message
    # Skipped!, optionally limited by device type and dtypes.
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=unittest.skip("Skipped!"),
        device_type=device_type,
        dtypes=dtypes,
    )


def skipIf(op_name, fail_fn, variant_name="", *, device_type=None, dtypes=None):
    # [FUNCTION DOC] Builds a decoration record that skips only the samples for which the user
    # function returns true, using a sample level skip rule matching every op and device since
    # the record already filters those.
    return decorate(
        op_name=op_name,
        variant_name=variant_name,
        decorator=sample_skips_and_xfails(
            [
                SkipRule(
                    # op matching is already handled by DecorateMeta
                    op_match_fn=lambda device, op: True,
                    # device matching is already handled by DecorateMeta
                    sample_match_fn=lambda device, sample: fail_fn(sample),
                )
            ]
        ),
        device_type=device_type,
        dtypes=dtypes,
    )


def decorateForModules(decorator, module_classes, device_type=None, dtypes=None):
    # [FUNCTION DOC] Returns a decorator that registers a decorator on module infos.
    # Arguments: the decorator, the module classes, optional device type and dtypes. Applying
    # the returned function to a test method looks up the test case and test names from its
    # qualified name (AssertionError if it is not exactly class and method), finds exactly one
    # module info per class in the module database (AssertionError otherwise) and appends a
    # decoration entry to its decorators. Mutates global module info objects; the test
    # function is returned unchanged.
    # This decorator doesn't modify fn in any way
    def wrapped(
        fn,
        module_classes=module_classes,
        decorator=decorator,
        device_type=device_type,
        dtypes=dtypes,
    ):
        # [FUNCTION DOC] HELPER: The returned decorator body described above. Defaults capture
        # the outer arguments. Registers the decorators on matching module infos and returns
        # the function unchanged.
        name_parts = fn.__qualname__.split(".")
        if len(name_parts) != 2:
            raise AssertionError(
                "Decorator only applies to a test function of a test class"
            )
        test_case_name, base_test_name = name_parts
        for module_cls in module_classes:
            matching_module_infos = [m for m in module_db if m.module_cls == module_cls]
            if len(matching_module_infos) != 1:
                raise AssertionError(
                    f"Couldn't find single ModuleInfo for {module_cls}"
                )
            module_info = matching_module_infos[0]
            decorators = list(module_info.decorators)
            new_decorator = DecorateInfo(
                decorator,
                test_case_name,
                base_test_name,
                device_type=device_type,
                dtypes=dtypes,
            )
            decorators.append(new_decorator)
            module_info.decorators = tuple(decorators)
        return fn

    return wrapped


def expectedFailureIf(condition):
    # [FUNCTION DOC] Returns a decorator that makes the function an expected failure if the
    # condition is truthy and leaves it unchanged otherwise.
    def decorator(fn):
        # [FUNCTION DOC] HELPER: Applies the expected failure marker when the captured
        # condition holds, otherwise returns the function unchanged.
        if condition:
            return unittest.expectedFailure(fn)
        return fn

    return decorator


def tol2(op_name, variant_name, override_dct, *, device_type=None):
    # [FUNCTION DOC] Builds a tolerance override record, a tuple of op name, variant name,
    # override mapping and device type.
    return (op_name, variant_name, override_dct, device_type)


def tol1(op_name, override_dct, *, device_type=None):
    # [FUNCTION DOC] Builds a tolerance override record for an operator without a variant
    # name, by delegating to the two field builder.
    return tol2(op_name, "", override_dct, device_type=device_type)


def opsToleranceOverride(test_case_name, base_test_name, overrides):
    # [FUNCTION DOC] Attaches tolerance overrides to operator infos. Arguments: test case
    # name, base test name, and a list of override records. For each record, finds exactly one
    # operator info among the standard and additional databases with matching name and variant
    # (AssertionError otherwise) and appends a decoration with the tolerance override to it.
    # Mutates global operator info objects. Returns a decorator that leaves the function
    # unchanged.
    all_opinfos = op_db + additional_op_db
    for override in overrides:
        op_name, variant_name, override, device_type = override
        matching_opinfos = [
            o
            for o in all_opinfos
            if o.name == op_name and o.variant_test_name == variant_name
        ]
        if len(matching_opinfos) != 1:
            raise AssertionError(f"Couldn't find OpInfo for {override}")
        opinfo = matching_opinfos[0]
        decorators = list(opinfo.decorators)
        decorators.append(
            DecorateInfo(
                toleranceOverride(override),
                test_case_name,
                base_test_name,
                device_type=device_type,
            )
        )
        opinfo.decorators = tuple(decorators)

    # This decorator doesn't modify fn in any way
    def wrapped(fn):
        # [FUNCTION DOC] HELPER: Identity decorator returned for use on test functions.
        return fn

    return wrapped


class DisableVmapFallback:
    # [CLASS DOC] Context manager that turns the vmap fallback off in functorch for the
    # duration of the block, so tests detect operations that lack a batching rule. Stores the
    # previous setting on entry and restores it on exit.
    def __enter__(self):
        # [FUNCTION DOC] Records the current vmap fallback enabled state in an attribute and
        # disables the fallback.
        self.prev_state = torch._C._functorch._is_vmap_fallback_enabled()
        torch._C._functorch._set_vmap_fallback_enabled(False)

    def __exit__(self, *ignored):
        # [FUNCTION DOC] Restores the saved fallback enabled state. Ignores exception
        # information and never suppresses exceptions.
        torch._C._functorch._set_vmap_fallback_enabled(self.prev_state)


def check_vmap_fallback(test_case, thunk, opinfo, dry_run=False):
    # [FUNCTION DOC] Runs the thunk with the vmap fallback disabled. If it raises, the
    # exception propagates unless dry_run is set, in which case it prints a line suggesting an
    # expected failure entry for the operator name, with the variant name if present. The test
    # case argument is unused.
    try:
        with DisableVmapFallback():
            thunk()
    except Exception:
        if not dry_run:
            raise
        if opinfo.variant_test_name:
            print(f"xfail('{opinfo.name}', '{opinfo.variant_test_name}'),")
        else:
            print(f"xfail('{opinfo.name}'),")


def saved_tensors_hooks_to_gm(
    pack_fn, unpack_fn, pack_cache_hash, unpack_cache_hash, symbolic_tracing=True
):
    # [FUNCTION DOC] Builds graph modules for pack and unpack hooks of saved tensors. With
    # symbolic tracing, symbolically traces both functions. Otherwise creates a random two by
    # three tensor marked dynamic on both dimensions and makes graphs by tracing the pack
    # function on it and the unpack function on the pack output. Then attaches the given cache
    # hash strings to the first node meta data of each graph. Returns the pack and unpack
    # modules. Marking dynamic is a side effect on the temporary tensor only.
    if symbolic_tracing:
        pack_gm = torch.fx.symbolic_trace(pack_fn)
        unpack_gm = torch.fx.symbolic_trace(unpack_fn)
    else:
        from torch.functorch import make_fx

        inp = torch.randn(2, 3)
        torch._dynamo.mark_dynamic(inp, 0)
        torch._dynamo.mark_dynamic(inp, 1)
        pack_out = pack_fn(inp)
        pack_gm = make_fx(pack_fn)(inp)
        unpack_gm = make_fx(unpack_fn)(pack_out)

    def set_manual_hash(g, manual_hash):
        # [FUNCTION DOC] HELPER: Stores a user supplied cache hash string in the meta data of
        # the first node of a graph.
        node = next(iter(g.nodes))
        node.meta["user_cache_hash"] = manual_hash

    set_manual_hash(pack_gm.graph, pack_cache_hash)
    set_manual_hash(unpack_gm.graph, unpack_cache_hash)

    return pack_gm, unpack_gm


@contextmanager
def capture_codegen_source(artifact_name):
    # [FUNCTION DOC] Context manager capturing payloads of structured trace artifacts with the
    # given name. Temporarily attaches a handler to the trace logger at debug level, yields a
    # list that is filled with payloads of matching records during the block, and on exit
    # removes the handler and restores the previous level, even on error.
    trace_log = logging.getLogger("torch.__trace")
    captured: list[str] = []

    class _ArtifactHandler(logging.Handler):
        # [CLASS DOC] HELPER: Logging handler appending the payload of any record whose
        # artifact metadata name equals the requested artifact name to the shared list;
        # records without payload are ignored.
        def emit(self, record):
            # [FUNCTION DOC] HELPER: Reads metadata and payload attributes from the record,
            # defaulting to empty, and appends the payload to the captured list when the
            # artifact name matches and the payload exists.
            metadata = getattr(record, "metadata", {})
            if (
                "artifact" in metadata
                and metadata["artifact"].get("name") == artifact_name
            ):
                payload = getattr(record, "payload", None)
                if payload is not None:
                    captured.append(payload)

    handler = _ArtifactHandler()
    handler.setLevel(logging.DEBUG)
    old_level = trace_log.level
    trace_log.setLevel(logging.DEBUG)
    trace_log.addHandler(handler)
    try:
        yield captured
    finally:
        trace_log.removeHandler(handler)
        trace_log.setLevel(old_level)
