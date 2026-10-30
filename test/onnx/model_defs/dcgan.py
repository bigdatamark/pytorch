# [MODULE DOC] DCGAN generator and discriminator models plus a weight initializer, used as
# ONNX test models. Module-level constants (bsz, imgsz, nz, ngf, ndf, nc) set batch, image
# and channel sizes. Only defines classes and functions at import; no side effects.
import torch
import torch.nn as nn


# configurable
bsz = 64
imgsz = 64
nz = 100
ngf = 64
ndf = 64
nc = 3


# custom weights initialization called on netG and netD
def weights_init(m):
    # [FUNCTION DOC] Initializer meant for Module.apply: for modules whose class name
    # contains Conv, fills the weight with normal noise (mean 0, std 0.02); for names
    # containing BatchNorm, weight gets normal noise (mean 1, std 0.02) and bias is set to
    # zero. Mutates in place; returns None.
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find("BatchNorm") != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


class _netG(nn.Module):
    # [CLASS DOC] DCGAN generator: maps a latent tensor of shape (N, nz, 1, 1) to an image
    # of shape (N, nc, 64, 64) through transposed convolutions, batch norm and ReLU, ending
    # with tanh. Holds ngpu and the sequential stack main.
    def __init__(self, ngpu):
        # [FUNCTION DOC] Stores ngpu and builds main, five transposed convolutions
        # without bias that upsample from 1x1 to 64x64, with channel widths ngf*8, ngf*4,
        # ngf*2, ngf, nc.
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(nz, ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(ngf * 2, ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf) x 32 x 32
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh(),
            # state size. (nc) x 64 x 64
        )

    def forward(self, input):
        # [FUNCTION DOC] Applies main to input. When ngpu is above one and the input data
        # is a CUDA float tensor, uses data_parallel across device ids 0..ngpu-1; otherwise
        # a plain call.
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
        return output


class _netD(nn.Module):
    # [CLASS DOC] DCGAN discriminator: maps images of shape (N, nc, 64, 64) to one
    # probability per sample using strided convolutions with batch norm and leaky ReLU,
    # ending with a sigmoid.
    def __init__(self, ngpu):
        # [FUNCTION DOC] Stores ngpu and builds main, strided 4x4 convolutions without bias
        # widening from nc to ndf*8 channels, then a final 4x4 convolution to one channel
        # and a sigmoid.
        super().__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is (nc) x 64 x 64
            nn.Conv2d(nc, ndf, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf) x 32 x 32
            nn.Conv2d(ndf, ndf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 2),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*2) x 16 x 16
            nn.Conv2d(ndf * 2, ndf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 4),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*4) x 8 x 8
            nn.Conv2d(ndf * 4, ndf * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ndf * 8),
            nn.LeakyReLU(0.2, inplace=True),
            # state size. (ndf*8) x 4 x 4
            nn.Conv2d(ndf * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid(),
        )

    def forward(self, input):
        # [FUNCTION DOC] Runs main (data_parallel across ngpu devices for CUDA float input
        # when ngpu is above one, else a direct call) and returns the result reshaped to a
        # column of shape (N, 1).
        if self.ngpu > 1 and isinstance(input.data, torch.cuda.FloatTensor):
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output.view(-1, 1)
