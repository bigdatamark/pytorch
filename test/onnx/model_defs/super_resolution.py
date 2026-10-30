# [MODULE DOC] Efficient sub-pixel convolution super-resolution network defined as an
# ONNX test model.
import torch.nn as nn
import torch.nn.init as init


class SuperResolutionNet(nn.Module):
    # [CLASS DOC] Four convolutions with ReLU followed by a pixel shuffle that upscales a
    # single-channel image by upscale_factor.
    def __init__(self, upscale_factor):
        # [FUNCTION DOC] Creates ReLU, four convolutions (1 to 64, 64 to 64, 64 to
        # 32, 32 to the square of the upscale factor channels), a pixel shuffle,
        # and initializes weights.
        super().__init__()

        self.relu = nn.ReLU()
        self.conv1 = nn.Conv2d(1, 64, (5, 5), (1, 1), (2, 2))
        self.conv2 = nn.Conv2d(64, 64, (3, 3), (1, 1), (1, 1))
        self.conv3 = nn.Conv2d(64, 32, (3, 3), (1, 1), (1, 1))
        self.conv4 = nn.Conv2d(32, upscale_factor**2, (3, 3), (1, 1), (1, 1))
        self.pixel_shuffle = nn.PixelShuffle(upscale_factor)

        self._initialize_weights()

    def forward(self, x):
        # [FUNCTION DOC] Applies the first three convolutions each followed by ReLU, then
        # the fourth convolution and pixel shuffle, and returns the upscaled result.
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.relu(self.conv3(x))
        x = self.pixel_shuffle(self.conv4(x))
        return x

    def _initialize_weights(self):
        # [FUNCTION DOC] HELPER: Orthogonally initializes the four convolution weights in
        # place, using the ReLU gain for the first three and the default gain for the last.
        init.orthogonal_(self.conv1.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv2.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv3.weight, init.calculate_gain("relu"))
        init.orthogonal_(self.conv4.weight)
