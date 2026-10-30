# [MODULE DOC] Helper classes for the torch-function override benchmarks: a trivial tensor
# subclass, a standalone class implementing the torch-function protocol, and a tensor subclass
# that overrides the protocol and defers to the default. Also defines repeat-count constants
# that nothing in this module uses.
import torch


NUM_REPEATS = 1000
NUM_REPEAT_OF_REPEATS = 1000


class SubTensor(torch.Tensor):
    # [CLASS DOC] HELPER: Empty tensor subclass; measures the cost of subclass dispatch without
    # any override.
    pass


class WithTorchFunction:
    # [CLASS DOC] HELPER: Non-tensor wrapper holding one tensor in the attribute _tensor
    # and implementing the torch-function protocol so that torch functions applied to it
    # are intercepted.
    def __init__(self, data, requires_grad=False):
        # [FUNCTION DOC] HELPER: Wraps the given data. If data is already a tensor it is stored
        # as is (requires_grad is ignored in that case); otherwise a new tensor is built from it
        # with the given requires_grad flag.
        if isinstance(data, torch.Tensor):
            self._tensor = data
            return

        self._tensor = torch.tensor(data, requires_grad=requires_grad)

    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Protocol hook called for any torch function applied to this
        # class. Normalizes missing kwargs to an empty mapping but then ignores the function,
        # types and kwargs: it always returns a new wrapper around the sum of the wrapped
        # tensors of the first two positional args.
        # KNOWN ISSUE: the result is an addition regardless of which function was requested, so
        #     it is only correct for the add benchmark.
        if kwargs is None:
            kwargs = {}

        return WithTorchFunction(args[0]._tensor + args[1]._tensor)


class SubWithTorchFunction(torch.Tensor):
    # [CLASS DOC] HELPER: Tensor subclass that overrides the torch-function protocol with a
    # pass-through to the default tensor implementation, to measure override overhead.
    @classmethod
    def __torch_function__(cls, func, types, args=(), kwargs=None):
        # [FUNCTION DOC] HELPER: Protocol hook that normalizes missing kwargs to an empty
        # mapping and delegates to the parent implementation, returning its result unchanged.
        if kwargs is None:
            kwargs = {}

        return super().__torch_function__(func, types, args, kwargs)
