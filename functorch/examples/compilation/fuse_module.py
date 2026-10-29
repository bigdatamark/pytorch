# [MODULE DOC]
# Example script comparing an eager module with a compiled-module wrapper (pass-through
# compilers by default). Importing it runs everything: builds the module, checks outputs and
# gradients match with assert_close (raises on mismatch), applies a manual update, and prints
# eager vs compiled timings for five rounds.
import timeit

import torch
import torch.nn as nn
from functorch.compile import compiled_module, tvm_compile


def nop(f, _):
    # [FUNCTION DOC] HELPER: Pass-through compiler; returns the given graph module unchanged.
    return f


fw_compiler = tvm_compile(target="llvm", tuning_logfile="fw_keops")
bw_compiler = tvm_compile(target="llvm", tuning_logfile="bw_keops")
fw_compiler = nop
bw_compiler = nop


def run(mod, input):
    # [FUNCTION DOC] HELPER: Runs the module on input, backpropagates the summed output, and
    # returns a tuple of the output followed by every parameter gradient. Side effect:
    # accumulates gradients on the parameters.
    out = mod(input)
    out.sum().backward()
    grads = [p.grad for p in mod.parameters()]
    return (out, *grads)


class Foo(nn.Module):
    # [CLASS DOC] Tiny test module with one scalar parameter and one buffer; computes a
    # scaled-and-shifted input summed over the first dimension.
    def __init__(self) -> None:
        # [FUNCTION DOC] Creates a random one-element parameter and a one-element buffer.
        super().__init__()
        self.param = nn.Parameter(torch.randn(1))
        self.register_buffer("buf", torch.randn(1))

    def forward(self, x):
        # [FUNCTION DOC] Returns the parameter times x plus the buffer, summed over dimension 0.
        return (self.param * x + self.buf).sum(dim=0)


input = torch.randn(1)
mod = Foo()
compiled_mod = compiled_module(mod, fw_compiler, bw_compiler)

for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_close(a, b)

out = mod(input)
out.sum().backward()
mod.param.data -= mod.param.grad
compiled_mod.orig_module.param.data -= compiled_mod.orig_module.param.grad
compiled_mod.orig_module.param.grad = None

for a, b in zip(run(mod, input), run(compiled_mod, input)):
    torch.testing.assert_close(a, b)

for _ in range(5):
    i = 10000
    t = timeit.Timer("mod(input)", globals=globals()).timeit(10000)
    print(f"eager {t / i * 1e6}")
    t = timeit.Timer("compiled_mod(input)", globals=globals()).timeit(10000)
    print(f"compiled {t / i * 1e6}")
