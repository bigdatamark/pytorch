# Owner(s): ["module: onnx"]
# [MODULE DOC] Helper module for ONNX export tests defining one custom autograd function, kept in
# its own module so tests can check how the exported graph names the function's module. Importing
# it only defines the class.

import torch


# Autograd function that is a replica of the autograd function in
# test_utility_funs.py (test_autograd_module_name)
class CustomFunction(torch.autograd.Function):
    # [CLASS DOC] Custom autograd function implementing a clamp at zero (a ReLU) with a
    # hand-written backward pass; stateless, used as a replica of a function in the
    # utility-function tests.
    @staticmethod
    def forward(ctx, input):
        # [FUNCTION DOC] Forward pass: saves the input for backward and returns the input
        # clamped to a minimum of zero.
        ctx.save_for_backward(input)
        return input.clamp(min=0)

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] Backward pass: reads the saved input, copies the incoming gradient
        # and zeroes it wherever the saved input was negative; returns that gradient for the
        # single input.
        (input,) = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input[input < 0] = 0
        return grad_input
