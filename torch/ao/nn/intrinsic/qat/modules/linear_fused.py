# mypy: allow-untyped-defs
# [MODULE DOC] QAT fused Linear+BatchNorm1d module (LinearBn1d). No import-time side
# effects.
import torch
import torch.ao.nn.intrinsic as nni
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import init
from torch.nn.parameter import Parameter
from torch.nn.utils.fusion import fuse_linear_bn_weights


__all__ = [
    "LinearBn1d",
]


class LinearBn1d(nn.modules.linear.Linear, nni._FusedModule):
    # [CLASS DOC] QAT module fusing Linear and BatchNorm1d with a weight FakeQuantize.
    # State: bn (a BatchNorm1d over out_features), weight_fake_quant, qconfig, freeze_bn
    # flag. Forward scales the weight by the BN running statistics before fake-
    # quantizing, then undoes the scaling and applies BN. Provides freeze/update of BN
    # stats, from_float and to_float.
    r"""
    A LinearBn1d module is a module fused from Linear and BatchNorm1d, attached
    with FakeQuantize modules for weight, used in quantization aware training.

    We combined the interface of :class:`torch.nn.Linear` and
    :class:torch.nn.BatchNorm1d`.

    Similar to :class:`torch.nn.Linear`, with FakeQuantize modules initialized
    to default.

    Attributes:
        freeze_bn:
        weight_fake_quant: fake quant module for weight

    """

    def __init__(
        self,
        # Linear args
        in_features,
        out_features,
        bias=True,
        # BatchNorm1d args
        # num_features: out_features
        eps=1e-05,
        momentum=0.1,
        # affine: True
        # track_running_stats: True
        # Args for this module
        freeze_bn=False,
        qconfig=None,
    ):
        # [FUNCTION DOC] Builds the Linear, a BatchNorm1d(out_features) with affine and
        # running stats, and the weight fake-quant from qconfig. Raises AssertionError
        # if qconfig is falsy. freeze_bn is forced True when the module is not in
        # training mode. Re-creates the bias parameter, resets BN parameters, then
        # freezes or updates BN stats according to the training flag and freeze_bn.
        nn.modules.linear.Linear.__init__(self, in_features, out_features, bias)
        if not qconfig:
            raise AssertionError("qconfig must be provided for QAT module")
        self.qconfig = qconfig
        self.freeze_bn = freeze_bn if self.training else True
        self.bn = nn.BatchNorm1d(out_features, eps, momentum, True, True)
        self.weight_fake_quant = self.qconfig.weight()
        if bias:
            self.bias = Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_bn_parameters()

        # this needs to be called after reset_bn_parameters,
        # as they modify the same state
        if self.training:
            if freeze_bn:
                self.freeze_bn_stats()
            else:
                self.update_bn_stats()
        else:
            self.freeze_bn_stats()

    def reset_running_stats(self):
        # [FUNCTION DOC] Resets the BN running statistics.
        self.bn.reset_running_stats()

    def reset_bn_parameters(self):
        # [FUNCTION DOC] Resets BN running stats, initializes the BN weight uniformly
        # and zeroes the BN bias.
        self.bn.reset_running_stats()
        init.uniform_(self.bn.weight)
        init.zeros_(self.bn.bias)

    def update_bn_stats(self):
        # [FUNCTION DOC] Unfreezes BN: clears freeze_bn and puts the BN submodule in
        # training mode. Returns self.
        self.freeze_bn = False
        self.bn.training = True
        return self

    def freeze_bn_stats(self):
        # [FUNCTION DOC] Freezes BN: sets freeze_bn and puts the BN submodule in eval
        # mode. Returns self.
        self.freeze_bn = True
        self.bn.training = False
        return self

    def forward(self, input):
        # [FUNCTION DOC] Computes fused linear+BN for QAT. Scales the weight per output
        # feature by bn weight / running std, fake-quantizes it, applies the linear op
        # with zero bias, divides by the scale, adds the original bias, then applies BN.
        # Raises AssertionError if bn.running_var is None. Returns the BN output.
        if self.bn.running_var is None:
            raise AssertionError("self.bn.running_var must not be None")

        # Scale the linear weights by BN's running statistics to reduce
        # weight jitter, see https://arxiv.org/pdf/1806.08342.pdf, page 18
        # for motivation.
        #
        # Instead of
        #
        #   x1 = F.linear(x0, fq(w), b)
        #   x2 = self.bn(x1)
        #
        # We have
        #
        #   # scale the weight by previous batch's running statistics
        #   scale_factor = bn.w / bn.running_std_from_prev_batch
        #   # do the linear transformation without bias
        #   x1_scaled = F.linear(x0, fq(w * scale_factor), 0)
        #   # reverse the scaling and add original bias
        #   x1_orig = x1_scaled / scale_factor + b
        #   x2 = self.bn(x1_orig)

        running_std = torch.sqrt(self.bn.running_var + self.bn.eps)
        scale_factor = self.bn.weight / running_std
        weight_shape = [1] * len(self.weight.shape)
        weight_shape[0] = -1
        bias_shape = [1] * len(self.weight.shape)
        bias_shape[1] = -1
        scaled_weight = self.weight_fake_quant(
            self.weight * scale_factor.reshape(weight_shape)
        )
        if self.bias is not None:
            zero_bias = torch.zeros_like(self.bias)
        else:
            zero_bias = torch.zeros(self.out_features, device=scaled_weight.device)
        linear_out = F.linear(input, scaled_weight, zero_bias)
        linear_out_orig = linear_out / scale_factor.reshape(bias_shape)
        if self.bias is not None:
            linear_out_orig = linear_out_orig + self.bias.reshape(bias_shape)
        bn_out = self.bn(linear_out_orig)
        return bn_out

    def train(self, mode=True):
        # [FUNCTION DOC] Sets the training flag; propagates to children only when BN is
        # not frozen, so a frozen BN stays in eval mode. Returns self.
        """
        Batchnorm's training behavior is using the self.training flag. Prevent
        changing it if BN is frozen. This makes sure that calling `model.train()`
        on a model with a frozen BN will behave properly.
        """
        self.training = mode
        if not self.freeze_bn:
            for module in self.children():
                module.train(mode)
        return self

    @classmethod
    def from_float(cls, mod, use_precomputed_fake_quant=False):
        # [FUNCTION DOC] Classmethod: builds the QAT module from a float fused
        # LinearBn1d, sharing its weight, bias and BN parameters/buffers (no copy).
        # Raises AssertionError if the type is not LinearBn1d or qconfig is missing or
        # falsy. use_precomputed_fake_quant is unused.
        r"""Create a qat module from a float module or qparams_dict

        Args:
            mod: A float module, either produced by torch.ao.quantization
                utilities or directly from the user.
        """
        if type(mod) is not nni.LinearBn1d:
            raise AssertionError(
                "qat."
                + cls.__name__
                + ".from_float only works for "
                + nni.LinearBn1d.__name__
            )
        if not hasattr(mod, "qconfig"):
            raise AssertionError("Input float module must have qconfig defined")
        if not mod.qconfig:
            raise AssertionError("Input float module must have a valid config")
        qconfig = mod.qconfig
        linear, bn = mod[0], mod[1]
        qat_linearbn = cls(
            linear.in_features,
            linear.out_features,
            linear.bias is not None,
            bn.eps,
            bn.momentum,
            False,
            qconfig,
        )
        qat_linearbn.weight = linear.weight  # type: ignore[assignment]
        qat_linearbn.bias = linear.bias  # type: ignore[assignment]
        qat_linearbn.bn.weight = bn.weight  # type: ignore[assignment]
        qat_linearbn.bn.bias = bn.bias  # type: ignore[assignment]
        qat_linearbn.bn.running_mean = bn.running_mean  # type: ignore[assignment]
        qat_linearbn.bn.running_var = bn.running_var  # type: ignore[assignment]
        qat_linearbn.bn.num_batches_tracked = bn.num_batches_tracked  # type: ignore[assignment]
        return qat_linearbn

    def to_float(self):
        # [FUNCTION DOC] Converts to a plain float Linear by folding the BN running
        # stats into weight and bias. Raises AssertionError if BN running stats are
        # None. Returns the Linear.
        linear = torch.nn.Linear(self.in_features, self.out_features)
        if self.bn.running_var is None or self.bn.running_mean is None:
            raise AssertionError(
                "self.bn.running_var and self.bn.running_mean must not be None"
            )
        linear.weight, linear.bias = fuse_linear_bn_weights(
            self.weight,
            self.bias,
            self.bn.running_mean,
            self.bn.running_var,
            self.bn.eps,
            self.bn.weight,
            self.bn.bias,
        )
        return linear
