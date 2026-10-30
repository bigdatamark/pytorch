# [MODULE DOC] Tiny shared helper for the microbenchmarks in this directory: a wrapper over
# torch.utils.benchmark Timer. No import-time side effects.
from torch.utils.benchmark import Timer


def time_with_torch_timer(fn, args, kwargs=None, iters=100):
    # [FUNCTION DOC] Time a call of fn with the given positional args and optional keyword args
    # (default empty) using a Timer that runs the call the given number of iterations (default
    # 100). Returns the Timer measurement object (with a mean attribute, in seconds). It accepts
    # no label argument.
    kwargs = kwargs or {}
    env = {"args": args, "kwargs": kwargs, "fn": fn}
    fn_call = "fn(*args, **kwargs)"

    # Measure end-to-end time
    timer = Timer(stmt=f"{fn_call}", globals=env)
    tt = timer.timeit(iters)

    return tt
