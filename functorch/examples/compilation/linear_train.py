# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.
# [MODULE DOC]
# Example script comparing three training loops for a small linear network: plain functional
# step, NNC-compiled functional step, and TorchScript module with SGD. Import-time side
# effects: enables CPU fusion in the JIT, builds the model and data, and runs and prints all
# three training runs.

import time

import torch
import torch.nn as nn
from functorch import make_functional
from functorch.compile import nnc_jit


torch._C._jit_override_can_fuse_on_cpu(True)


def bench(f, iters=100, warmup=10):
    # [FUNCTION DOC] Calls f warmup times, then times iters calls and prints elapsed seconds.
    # Unused in this script.
    for _ in range(warmup):
        f()
    begin = time.time()
    for _ in range(iters):
        f()
    print(time.time() - begin)


class Foo(nn.Module):
    # [CLASS DOC] Stack of bias-free linear layers; forward returns the sum of squares of the
    # output as a scalar loss.
    def __init__(self, num_layers=3, features=100):
        # [FUNCTION DOC] Builds a sequential stack of num_layers square linear layers of the
        # given width, without bias.
        super().__init__()
        mods = []
        for _ in range(num_layers):
            mods.append(nn.Linear(features, features, bias=False))
        self.mod = nn.Sequential(*mods)

    def forward(self, x):
        # [FUNCTION DOC] Applies the layer stack to x, squares the result and returns the sum.
        return (self.mod(x) ** 2).sum()


batch_size = 16
features = 64
num_layers = 8
inp = torch.randn((batch_size, features))

mod = Foo(num_layers, features)

jit_mod = torch.jit.script(mod)

func_model, weights = make_functional(mod)
lr = 1.0


def functional_step(x, weights):
    # [FUNCTION DOC] One SGD step in functional style: detaches weights, runs the functional
    # model, backpropagates, and returns the loss and the list of updated weights using the
    # global learning rate. Does not mutate the inputs.
    weights = [weight.detach().requires_grad_() for weight in weights]
    out = func_model(weights, x)
    out.backward()
    new_weights = [weight - lr * weight.grad for weight in weights]
    return out, new_weights


optim = torch.optim.SGD(
    jit_mod.parameters(), lr=lr, momentum=0, dampening=0, weight_decay=0
)


def jit_step(x, weights):
    # [FUNCTION DOC] One SGD step on the scripted module via the optimizer: zero gradients,
    # forward, backward, step. Returns the loss and None; weights is unused. Mutates the
    # module parameters.
    optim.zero_grad()
    loss = jit_mod(x)
    loss.backward()
    optim.step()
    return loss, None


def train(train_step, weights):
    # [FUNCTION DOC] Seeds the RNG, runs one warm-up step, then 1000 timed steps on random
    # batches, threading the returned weights through. Prints the loss every 200 steps and
    # the total time. Returns nothing.
    torch.manual_seed(16)
    train_step(inp, weights)
    begin = time.time()
    for itr in range(1000):
        loss, weights = train_step(torch.randn(batch_size, features), weights)
        if itr % 200 == 0:
            print(f"Loss at {itr}: {loss}")
    print("Time taken: ", time.time() - begin)
    print()


grad_pt = functional_step
grad_nnc = nnc_jit(functional_step)

print("Starting PT training")
train(grad_pt, weights)

print("Starting NNC training")
train(grad_nnc, weights)

print("Starting JIT training")
train(jit_step, None)
