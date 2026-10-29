# [MODULE DOC] HELPER: Mock module used by dynamo import tests. Imports torch and the sibling
# mock_module3, and defines a small class plus a module function that appends to a list.
# from . import mock_module3
import torch

from . import mock_module3


class Class1:
    # [CLASS DOC] HELPER: Stores two values x and y and delegates a method to mock_module3.
    def __init__(self, x, y):
    # [FUNCTION DOC] HELPER: Store the arguments as attributes x and y.
        self.x = x
        self.y = y

    def method2(self, x):
    # [FUNCTION DOC] HELPER: Call mock_module3 method1 with a fresh empty list and x and return
    # the resulting list (which then contains x). Instance attributes are not used.
        return mock_module3.method1([], x)


def method1(x, y):
    # [FUNCTION DOC] HELPER: Create (and discard) a 1x1 tensor of ones, append y to the list x in
    # place and return x.
    torch.ones(1, 1)
    x.append(y)
    return x
