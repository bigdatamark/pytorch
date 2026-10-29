# [PACKAGE DOC] HELPER: Legacy package exposing foreach-enabled variants of the optimizers
# (Adam, AdamW, NAdam, SGD, RAdam, RMSprop, Rprop, ASGD, Adamax, Adadelta, Adagrad). Each name
# is a subclass of the torch.optim optimizer whose constructor defaults to foreach set to true.
# Built at import time.
"""
:mod:`torch.optim._multi_tensor` is a package implementing various optimization algorithms.

Most commonly used methods are already supported, and the interface is general
enough, so that more sophisticated ones can be also easily integrated in the
future.
"""

from functools import partialmethod

from torch import optim


def partialclass(cls, *args, **kwargs):
    # [FUNCTION DOC] HELPER: Returns a subclass of cls whose constructor has the given positional
    # and keyword arguments pre-bound (via a partial method). Does not modify cls itself.
    class NewCls(cls):
        # [CLASS DOC] HELPER: Dynamically created subclass of cls; only overrides the constructor
        # with a partially applied version of the parent constructor.
        __init__ = partialmethod(cls.__init__, *args, **kwargs)

    return NewCls


Adam = partialclass(optim.Adam, foreach=True)
AdamW = partialclass(optim.AdamW, foreach=True)
NAdam = partialclass(optim.NAdam, foreach=True)
SGD = partialclass(optim.SGD, foreach=True)
RAdam = partialclass(optim.RAdam, foreach=True)
RMSprop = partialclass(optim.RMSprop, foreach=True)
Rprop = partialclass(optim.Rprop, foreach=True)
ASGD = partialclass(optim.ASGD, foreach=True)
Adamax = partialclass(optim.Adamax, foreach=True)
Adadelta = partialclass(optim.Adadelta, foreach=True)
Adagrad = partialclass(optim.Adagrad, foreach=True)
