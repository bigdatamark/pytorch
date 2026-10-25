# [MODULE DOC] ExportDB example: a forward that reassigns a module attribute.
# Defines example_args, tags and model.
# mypy: allow-untyped-defs
import torch


class ModelAttrMutation(torch.nn.Module):
    # [CLASS DOC] Example module with an attribute holding a list of two random
    # 3x2 tensors that forward replaces on each call.
    """
    Attribute mutation raises a warning. Covered in the test_export.py test_detect_leak_strict test.
    """

    def __init__(self) -> None:
        # [FUNCTION DOC] Stores attr_list as a list of two random 3x2 tensors.
        super().__init__()
        self.attr_list = [torch.randn(3, 2), torch.randn(3, 2)]

    def recreate_list(self):
        # [FUNCTION DOC] Returns a new list of two 3x2 zero tensors.
        return [torch.zeros(3, 2), torch.zeros(3, 2)]

    def forward(self, x):
        # [FUNCTION DOC] Side effect: replaces the attribute attr_list with fresh
        # zero tensors. Returns the sum of x plus the sum of the new first element.
        self.attr_list = self.recreate_list()
        return x.sum() + self.attr_list[0].sum()


example_args = (torch.randn(3, 2),)
tags = {"python.object-model"}
model = ModelAttrMutation()
