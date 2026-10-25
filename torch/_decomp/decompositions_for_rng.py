# mypy: allow-untyped-decorators
# mypy: allow-untyped-defs
# [MODULE DOC]
# Decompositions that functionalize CUDA RNG ops into Philox-based functional
# primitives for AOT Autograd tracing. Contains the rng_decompositions registry
# (keyed by name, mapping op overloads to functions), rand/rand_like
# decompositions, PhiloxState and PhiloxStateTracker (seed/offset tracking for
# forward and backward), and extra random decompositions (bernoulli, normal,
# dropout, ...) that reduce to rand_like.
# Import-time side effects: builds extra_random_decomps from the global
# decomposition table, registers bernoulli decompositions, and merges all of
# them into rng_decompositions.
import functools
from collections import defaultdict
from collections.abc import Callable

import torch
import torch._decomp as decomp
from torch._decomp import get_decompositions
from torch._ops import OpOverload


aten = torch.ops.aten

rng_decompositions: dict[str, dict[OpOverload, Callable]] = defaultdict(dict)


def register_rng_decomposition(aten_op):
    # [FUNCTION DOC]
    # Returns a decorator registering a decomposition for aten_op into the
    # rng_decompositions registry.
    return decomp.register_decomposition(aten_op, rng_decompositions)


def throw_on_non_cuda(device):
    # [FUNCTION DOC] HELPER: Always raises RuntimeError explaining that RNG
    # functionalization is only supported for Philox-based (CUDA) devices.
    raise RuntimeError(
        f"You are trying to functionalize a {device.type} RNG operator but {device.type} does not "
        f"use Philox/counter-based RNG. Therefore, functionalizing a {device.type} RNG operator is "
        "not supported. We are discussing the possibility of a Philox-based RNG implementation for CPU."
    )


# TODO - We have to register many more distributions here, and also higher level
# ops like dropout which have fused implementation and can hide the rand inside.
@register_rng_decomposition(aten.rand)
def rand(shape, dtype=None, layout=torch.strided, device=None, pin_memory=False):
    # [FUNCTION DOC]
    # Decomposition of aten.rand into the functional philox_rand primitive.
    # Reads (seed, offset) from the tracker, draws a tensor of the given shape
    # (dtype defaults to float32), and advances the tracker offset by the
    # amount consumed. Raises RuntimeError for a non-CUDA device. layout and
    # pin_memory are ignored.
    if device and device.type != "cuda":
        throw_on_non_cuda(device)
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    dtype = dtype or torch.float32
    out, offset_jump = torch.ops.rngprims.philox_rand(
        shape, seed, offset, None, device, dtype
    )
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


@register_rng_decomposition(aten.rand_like)
def rand_like(
    x: torch.Tensor,
    dtype=None,
    layout=None,
    device=None,
    pin_memory=False,
    memory_format=torch.preserve_format,
):
    # [FUNCTION DOC]
    # Decomposition of aten.rand_like via philox_rand: draws uniform values
    # with the shape of x, using x's device and dtype unless overridden, and
    # advances the tracker offset. Raises RuntimeError for non-CUDA devices.
    # layout, pin_memory and memory_format are ignored (output is contiguous).
    device = device or x.device
    if device.type != "cuda":
        throw_on_non_cuda(device)
    dtype = dtype or x.dtype
    seed, offset = PhiloxStateTracker.get_state_as_tuple()
    out, offset_jump = torch.ops.rngprims.philox_rand(
        x.shape, seed, offset, None, device, dtype
    )
    PhiloxStateTracker.advance_offset(offset_jump)
    return out


class PhiloxState:
    # [CLASS DOC]
    # Holds one Philox RNG state: seed and base_offset tensors (RNG state just
    # before tracing) plus an integer relative_offset (offset consumed during
    # tracing), and a flag offset_advanced_at_least_once. The effective offset
    # is base_offset + relative_offset. Seed/base_offset are empty tensors until
    # set; get_state_* validate that they were set.
    """
    Represents a PhiloxRngState - (seed, offset) where offset = base_offset +
    relative_offset. seed and base_offset basically point to the rng state just
    before tracing starts. relative offset tracks the totally consumed offset at
    trace time.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Initialize to the empty/reset state.
        self.reset()

    def reset(self):
        # [FUNCTION DOC] Reset seed and base_offset to empty tensors,
        # relative_offset to 0, and clear the advanced flag.
        self.seed = torch.tensor(())
        self.base_offset = torch.tensor(())
        self.relative_offset = 0
        self.offset_advanced_at_least_once = False

    def validate_state(self):
        # [FUNCTION DOC] Raise AssertionError if seed or base_offset is still
        # empty (state never set).
        if self.seed.numel() == 0 or self.base_offset.numel() == 0:
            raise AssertionError(
                f"seed and base_offset must not be empty, got "
                f"seed.numel()={self.seed.numel()}, base_offset.numel()={self.base_offset.numel()}"
            )

    def advance_offset(self, consumed_offset):
        # [FUNCTION DOC] Add consumed_offset to relative_offset and mark that
        # the offset has been advanced.
        self.offset_advanced_at_least_once = True
        self.relative_offset = self.relative_offset + consumed_offset

    def set_state(self, seed, base_offset, relative_offset=0):
        # [FUNCTION DOC] Store seed, base_offset and relative_offset (default
        # 0). Does not touch the advanced flag.
        self.seed = seed
        self.base_offset = base_offset
        self.relative_offset = relative_offset

    def get_state_as_tuple(self):
        # [FUNCTION DOC] Validate, then return (seed, base_offset +
        # relative_offset). Raises AssertionError if state is unset.
        self.validate_state()
        return (self.seed, self.base_offset + self.relative_offset)

    def get_state_as_tensor(self):
        # [FUNCTION DOC] Validate, then return seed and the effective offset
        # stacked into one tensor (used to emulate get_rng_state). Raises
        # AssertionError if state is unset.
        # Only needed because we override get_rng_state.
        self.validate_state()
        return torch.stack([self.seed, self.base_offset + self.relative_offset])

    def set_state_from_tensor(self, state):
        # [FUNCTION DOC] Unbind a stacked state tensor into seed and
        # base_offset and reset relative_offset to 0 (emulates set_rng_state).
        # Only needed because we override set_rng_state.
        self.seed, self.base_offset = torch.unbind(state)
        self.relative_offset = 0


class PhiloxStateTracker:
    # [CLASS DOC]
    # Process-wide singleton (class-level state, no instances needed) tracking
    # Philox RNG state during AOT Autograd tracing. Class attributes:
    # fwd_state, bwd_state (PhiloxState each) and running_state, which aliases
    # whichever is currently active. Usable as a context manager that resets
    # state on enter and exit. Provides helpers to record state, advance
    # offsets, and compute final offsets rounded up to a multiple of 4.
    """
    Singleton class to track the philox rng state during AOT Autograd tracing.
    For each aot tracing instance, AOT Autograd resets this tracker and keeps
    track of both forward and backward offsets. At runtime, we only care about
    the total consumed forward and backward offsets. For dynamic shapes, these
    offsets are a function of input shapes. Therefore, the AOT generated graphs
    have additional outputs that compute total consumed forward and backward
    offsets.
    """

    running_state: PhiloxState
    fwd_state: PhiloxState
    bwd_state: PhiloxState

    def __enter__(self):
        # [FUNCTION DOC] Reset the global tracker and return self.
        PhiloxStateTracker.reset()
        return self

    def __exit__(self, exc_type, exc_cal, exc_tb):
        # [FUNCTION DOC] Reset the global tracker; never suppresses exceptions.
        PhiloxStateTracker.reset()

    @classmethod
    def reset(cls):
        # [FUNCTION DOC] Replace running, forward and backward states with
        # fresh PhiloxState objects (running_state is NOT aliased to
        # fwd_state afterwards).
        cls.running_state = PhiloxState()
        cls.fwd_state = PhiloxState()
        cls.bwd_state = PhiloxState()

    @classmethod
    def mark_beginning_of_forward(cls):
        # [FUNCTION DOC] Make the forward state the running state.
        # Tells the tracker to use fwd_state as the running state
        cls.running_state = cls.fwd_state

    @classmethod
    def mark_beginning_of_backward(cls):
        # [FUNCTION DOC] Make the backward state the running state.
        # Tells the tracker to use bwd_state as the running state
        cls.running_state = cls.bwd_state

    @classmethod
    def record_state(cls, seed, offset, mode):
        # [FUNCTION DOC] Store seed and base offset tensors for the given mode.
        # For "forward" it sets fwd_state and also makes it the running state;
        # for "backward" it only sets bwd_state. Any other mode raises
        # AssertionError.
        # Records the seed and offset tensors. These tensors are used to invoke
        # the philox_rand functional primitives.
        if mode == "forward":
            cls.fwd_state.set_state(seed, offset)
            cls.mark_beginning_of_forward()
        else:
            if mode != "backward":
                raise AssertionError(f"mode must be 'backward', got {mode}")
            cls.bwd_state.set_state(seed, offset)

    @classmethod
    def get_state_as_tensor(cls):
        # [FUNCTION DOC] Return the running state as a single stacked tensor
        # (seed, effective offset); see PhiloxState.get_state_as_tensor.
        # The only reason this exists is because we override get_rng_state and
        # set_rng_state during tracing. get_rng_state expects a tensor output,
        # so return (seed, offset) tuple upset other parts of the program like
        # ctx.saved_tensors.

        # A bad consequence is that if user saves and restores rng state, we
        # have little bit of ugliness in the generated code, where we first
        # concat the (seed, offset) to create a tensor for get_rng_state, and
        # then split it back to get (seed, offset) tuple in set_rng_state.

        # TODO: Investigate if there is a better way to wrap the tuple in a
        # false Tensor object, and then desugar it later on.
        return cls.running_state.get_state_as_tensor()

    @classmethod
    def get_state_as_tuple(cls):
        # [FUNCTION DOC] Return (seed, effective offset) of the running state.
        return cls.running_state.get_state_as_tuple()

    @classmethod
    def set_state_from_tensor(cls, x):
        # [FUNCTION DOC] Set the running state from a stacked state tensor x
        # (seed, offset); relative offset resets to 0.
        # This is only needed because we override set_rng_state. Look at the
        # comment in get_state_from_tensor method.
        cls.running_state.set_state_from_tensor(x)

    @classmethod
    def advance_offset(cls, consumed_offset):
        # [FUNCTION DOC] Advance the running state's relative offset.
        cls.running_state.advance_offset(consumed_offset)

    @classmethod
    def get_current_relative_offset(cls):
        # [FUNCTION DOC] Return the running state's consumed relative offset.
        return cls.running_state.relative_offset

    @staticmethod
    def multiple_of_4(offset):
        # [FUNCTION DOC] HELPER: Round offset up to the next multiple of 4
        # (CUDA RNG offsets must be multiples of 4). Works on ints or tensors.
        # torch cuda rng state offset must be a multiple of 4. For inductor, as
        # we sum up all the numel, the result might not be a multiple of 4. This
        # method achieves that.
        return (offset + 3) // 4 * 4

    @classmethod
    def get_updated_fwd_offset(cls):
        # [FUNCTION DOC] Return the forward base offset advanced by the
        # consumed relative offset, rounded up to a multiple of 4. If no rand
        # op advanced the offset, returns the unchanged base offset.
        # Short circuit if no rand ops were observed
        if not cls.fwd_state.offset_advanced_at_least_once:
            return cls.fwd_state.base_offset
        return cls.multiple_of_4(
            cls.fwd_state.base_offset + cls.fwd_state.relative_offset
        )

    @classmethod
    def get_updated_bwd_offset(cls):
        # [FUNCTION DOC] Same as get_updated_fwd_offset but for the backward
        # state.
        # Short circuit if no rand ops were observed
        if not cls.bwd_state.offset_advanced_at_least_once:
            return cls.bwd_state.base_offset
        return cls.multiple_of_4(
            cls.bwd_state.base_offset + cls.bwd_state.relative_offset
        )


# Adding more decompositions which eventually use rand_like inside decomps.
# Adding these in rng_decompositions ensures the functionalization of rand_like
# ops used in these decomps. The list is copied from inductor codebase, which
# uses it for similar purpose.
#
# Caution - These decomps do not have same accuracy as that of eager. However,
# we can't just disable them with a config flag like fallback_random, because
# for functionalization of rng ops, we have to decompose these ops.
extra_random_decomps = get_decompositions(
    [
        aten.cauchy,
        aten.cauchy_,
        aten.exponential,
        aten.exponential_,
        aten.geometric,
        aten.geometric_,
        aten.native_dropout,
        aten.normal,
        aten.normal_,
        aten.normal_functional,
        aten.log_normal,
        aten.log_normal_,
        aten.rrelu_with_noise,
        aten.rrelu_with_noise_,
        aten.uniform,
        aten.uniform_,
    ]
)
register_extra_random_decomp = functools.partial(
    decomp.register_decomposition, registry=extra_random_decomps
)


@register_extra_random_decomp([aten.bernoulli_])
def bernoulli_(self, p=0.5):
    # [FUNCTION DOC]
    # In-place bernoulli decomposition: fills self with whether a uniform
    # float32 draw is below p. Returns NotImplemented on CPU so the
    # decomposition is skipped there.
    if self.device == torch.device("cpu"):
        return NotImplemented
    return self.copy_(torch.rand_like(self, dtype=torch.float32) < p)


@register_extra_random_decomp([aten.bernoulli.p])
def bernoulli_p(self, p=0.5, *, generator=None):
    # [FUNCTION DOC]
    # Decomposition of bernoulli.p: returns a boolean tensor of self's shape
    # that is true where a uniform float32 draw is below p. Returns
    # NotImplemented on CPU. Raises AssertionError if a generator is given.
    if self.device == torch.device("cpu"):
        return NotImplemented
    if generator is not None:
        raise AssertionError(f"generator must be None, got {generator}")
    return torch.rand_like(self, dtype=torch.float32) < p


rng_decompositions.update(extra_random_decomps)  # type: ignore[arg-type]
