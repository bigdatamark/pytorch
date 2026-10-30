# Owner(s): ["oncall: package/deploy"]
# [MODULE DOC] Fixture defining TestNnModule, a DCGAN-style generator network built from
# transposed convolutions, batch norms and activations, used for save/load and backwards
# compatibility package tests.

import torch


class TestNnModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Image generator network: a sequential stack of four
    # transposed-convolution blocks (each followed by batch norm and relu) and a final
    # transposed convolution with a tanh, stored in the attribute main.
    def __init__(self, nz=6, ngf=9, nc=3):
        # [FUNCTION DOC] HELPER: Builds the sequential stack. The argument nz is the input
        # channel count, ngf scales the hidden channel widths (8, 4, 2 and 1 times ngf) and nc
        # is the output channel count. All transposed convolutions have no bias.
        super().__init__()
        self.main = torch.nn.Sequential(
            # input is Z, going into a convolution
            torch.nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            torch.nn.BatchNorm2d(ngf * 8),
            torch.nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            torch.nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 4),
            torch.nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            torch.nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf * 2),
            torch.nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            torch.nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            torch.nn.BatchNorm2d(ngf),
            torch.nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            torch.nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            torch.nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # [FUNCTION DOC] HELPER: Runs the input through the sequential stack and returns
        # the result.
        return self.main(input)
