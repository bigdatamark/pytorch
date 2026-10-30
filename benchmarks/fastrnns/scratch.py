# [MODULE DOC] Scratch script for manual experiments with TorchScript fusion on cuda. Import-time
# side effects (runs when executed or imported): defines scripted functions, creates random cuda
# tensors, runs a recurrent scripted function and requests its optimized graph, then runs a
# second recurrent scale and shift experiment, then checks backward through a mean of an empty
# tensor on cpu and cuda. Requires a cuda device; not used by other modules.
import torch


@torch.jit.script
def fn(x, scale, shift):
    # [FUNCTION DOC] HELPER: Scripted pointwise function returning scale times x divided by
    # shift.
    return scale * x / shift


@torch.jit.script
def recurrent(x, scale, shift):
    # [FUNCTION DOC] HELPER: Scripted function applying fn to the input a hundred times in a
    # loop, feeding each result back in, and returning the final tensor.
    y = x
    for i in range(100):
        y = fn(y, scale, shift)
    return y


x = torch.randn(2, 2, device="cuda")
scale = torch.randn(2, 2, device="cuda", requires_grad=True)
shift = torch.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]


out = recurrent(x, scale, shift)
recurrent.graph_for(x, scale, shift)


import torch


@torch.jit.script
def recurrent_scaleshift(x, scale, shift):
    # [FUNCTION DOC] HELPER: Scripted loop of 64 iterations computing scale times y plus shift,
    # starting from x; returns the final tensor.
    y = x
    for i in range(64):
        y = scale * y + shift
    return y


x = torch.randn(2, 2, device="cuda")
scale = torch.randn(2, 2, device="cuda", requires_grad=True)
shift = torch.randn(2, 2, device="cuda", requires_grad=True)
inputs = [x, scale, shift]
out = recurrent_scaleshift(x, scale, shift)
recurrent_scaleshift.graph_for(x, scale, shift)


import torch


x = torch.tensor([])
x.requires_grad = True
x.mean().backward()  # no error triggered
x = x.cuda()
x.mean().backward()
