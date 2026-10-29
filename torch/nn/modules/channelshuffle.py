# [MODULE DOC] Defines the ChannelShuffle module (exported via __all__), which permutes tensor
# channels across groups by delegating to the functional channel_shuffle. No import-time side effects.
import torch.nn.functional as F
from torch import Tensor

from .module import Module


__all__ = ["ChannelShuffle"]


class ChannelShuffle(Module):
    # [CLASS DOC] Module that splits the channel dimension into groups and interleaves them
    # (as in ShuffleNet) while preserving the tensor shape. State: groups (int, a TorchScript constant).
    # No learnable parameters.
    r"""Divides and rearranges the channels in a tensor.

    This operation divides the channels in a tensor of shape :math:`(N, C, *)`
    into g groups as :math:`(N, \frac{C}{g}, g, *)` and shuffles them,
    while retaining the original tensor shape in the final output.

    Args:
        groups (int): number of groups to divide channels in.

    Examples::

        >>> channel_shuffle = nn.ChannelShuffle(2)
        >>> input = torch.arange(1, 17, dtype=torch.float32).view(1, 4, 2, 2)
        >>> input
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[13., 14.],
                  [15., 16.]]]])
        >>> output = channel_shuffle(input)
        >>> output
        tensor([[[[ 1.,  2.],
                  [ 3.,  4.]],
                 [[ 9., 10.],
                  [11., 12.]],
                 [[ 5.,  6.],
                  [ 7.,  8.]],
                 [[13., 14.],
                  [15., 16.]]]])
    """

    __constants__ = ["groups"]
    groups: int

    def __init__(self, groups: int) -> None:
        # [FUNCTION DOC] Store the number of groups. Args: groups (int). Not validated here; the
        # functional call requires the channel count to be divisible by groups.
        super().__init__()
        self.groups = groups

    def forward(self, input: Tensor) -> Tensor:
        # [FUNCTION DOC] Shuffle the channels of the input (N, C, *) across the stored number of groups and
        # return a tensor of the same shape.
        """
        Runs the forward pass.
        """
        return F.channel_shuffle(input, self.groups)

    def extra_repr(self) -> str:
        # [FUNCTION DOC] Return the repr suffix showing groups.
        """
        Return the extra representation of the module.
        """
        return f"groups={self.groups}"
