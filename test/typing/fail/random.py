# flake8: noqa
# [MODULE DOC] Negative typing test. Type-checker input expecting an argument-type error when
# a list of ints rather than a Tensor is given to the function that restores the global
# random number generator state.
import torch


torch.set_rng_state(
    [  # E: Argument 1 to "set_rng_state" has incompatible type "list[int]"; expected "Tensor"  [arg-type]
        1,
        2,
        3,
    ]
)
