# [MODULE DOC] Throwaway script (not a test case) that defines a linear module whose forward calls
# the internal linear function directly, traces it with a random 20 by 20 input, and prints the
# traced graph. Import-time side effect: the trace and print run immediately on import.
import torch


class LinearMod(torch.nn.Linear):
    # [CLASS DOC] Linear layer subclass whose forward bypasses the functional wrapper and calls the
    # internal linear op with the stored weight and bias. State is the same as the parent layer.
    def __init__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the parent linear layer constructor.
        super().__init__(*args, **kwargs)

    def forward(self, input):
        # [FUNCTION DOC] Returns the internal linear op applied to the input with this layer's
        # weight and bias.
        return torch._C._nn.linear(input, self.weight, self.bias)


print(torch.jit.trace(LinearMod(20, 20), torch.rand([20, 20])).graph)
