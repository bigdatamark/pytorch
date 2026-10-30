# [MODULE DOC] Reveal-type sample: builds a list holding three tensors of rank 1, 2 and 3, then
# reveals the type of one shape entry of each element, expecting int in all three cases
# (indexing the shape tuple of an element drawn from the list).
import torch


input = []
input.append(torch.tensor([1.0, 2.0, 3.0, 4.0]))
input.append(torch.tensor([[1.0, 2.0, 3.0, 4.0]]))
input.append(torch.tensor([[[1.0, 2.0, 3.0, 4.0]]]))
reveal_type(input[0].shape[0])  # E: int
reveal_type(input[1].shape[1])  # E: int
reveal_type(input[2].shape[2])  # E: int
