# [MODULE DOC] Reveal-type sample: creates two random 2x3 tensors, reveals each as Tensor,
# performs an in-place copy of one into the other and reveals the target again as Tensor, then
# compares them for equality and reduces with all and reveals the result as Tensor. Expected
# types are in the marker comments.
import torch


t = torch.randn(2, 3)
reveal_type(t)  # E: {Tensor}
u = torch.randn(2, 3)
reveal_type(u)  # E: {Tensor}
t.copy_(u)
reveal_type(t)  # E: {Tensor}
r = (t == u).all()
reveal_type(r)  # E: {Tensor}
