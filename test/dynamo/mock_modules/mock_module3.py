# [MODULE DOC] HELPER: Mock module used by dynamo import tests; defines one helper function that
# mutates a list.
import torch


def method1(x, y):
    # [FUNCTION DOC] HELPER: Create (and discard) a 1x1 tensor of ones, append y to the list x in
    # place and return x.
    torch.ones(1, 1)
    x.append(y)
    return x
