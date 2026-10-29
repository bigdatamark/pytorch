# [MODULE DOC] HELPER: Helper functions for nested graph break dynamo tests, defined in a separate
# module so that inlined frames cross a module boundary. Holds a module global tensor, a module
# constant, and a constant tensor referenced from a closure. Import creates the global tensors.
import torch


global1 = torch.ones(3)


def reset_state():
    # [FUNCTION DOC] HELPER: Rebind the module global tensor global1 to a fresh tensor of three
    # ones.
    global global1
    global1 = torch.ones(3)


def fn(val, call):
    # [FUNCTION DOC] HELPER: Increment the global tensor global1 in place by one (side effect on
    # module state), add it to val, pass the sum to the callable call, add one to its result and
    # return it.
    global global1
    global1 += 1
    val = val + global1
    val = call(val)
    val = val + 1
    return val


_MODULE_CONST = 1


def fn_with_module_global(x):
    # [FUNCTION DOC] HELPER: Add the module constant to x, force a dynamo graph break, then return
    # x plus the constant plus one.
    x = x + _MODULE_CONST
    torch._dynamo.graph_break()
    return x + _MODULE_CONST + 1


HELPER_CONSTANT = torch.tensor([100.0])


def closure_with_graph_break(x):
    # [FUNCTION DOC] HELPER: Compute captured as x plus one, then call a closure that forces a
    # graph break and returns the captured value plus HELPER_CONSTANT; returns that result.
    captured = x + 1

    def inner():
        # [FUNCTION DOC] HELPER: Closure over captured; triggers a graph break, then returns
        # captured plus the module constant tensor.
        torch._dynamo.graph_break()
        return captured + HELPER_CONSTANT

    return inner()
