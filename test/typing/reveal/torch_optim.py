# [MODULE DOC] Reveal-type sample for optimizers: builds an Adagrad and an Adam optimizer over a
# single scalar tensor, reveals their types as Adagrad and Adam, and passes each to a helper
# typed to accept the Optimizer base class, checking the subclasses are accepted. Constructs the
# optimizers at import time.
import torch


def foo(opt: torch.optim.Optimizer) -> None:
    # [FUNCTION DOC] HELPER: Takes any Optimizer and calls its zero_grad method; returns
    # nothing. It exists to check that concrete optimizer classes are accepted where the base
    # class is annotated.
    opt.zero_grad()


opt_adagrad = torch.optim.Adagrad([torch.tensor(0.0)])
reveal_type(opt_adagrad)  # E: {Adagrad}
foo(opt_adagrad)

opt_adam = torch.optim.Adam([torch.tensor(0.0)], lr=1e-2, eps=1e-6)
reveal_type(opt_adam)  # E: {Adam}
foo(opt_adam)
