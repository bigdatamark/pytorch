# mypy: allow-untyped-defs
# [MODULE DOC] Converts dense nn.Module trees to MKLDNN (oneDNN) equivalents. Public entry is
# to_mkldnn; the Mkldnn* ScriptModule classes are created lazily on first use or attribute
# access (module level __getattr__) so importing this module triggers no TorchScript compilation.
import functools

import torch


# The Mkldnn* classes subclass torch.jit.ScriptModule, whose metaclass compiles
# @torch.jit.script_method members at class-definition time. Building them lazily
# keeps `import torch.utils.mkldnn` (and its transitive importers) free of any
# TorchScript compilation until MKLDNN conversion is actually used.
@functools.lru_cache(None)
def _get_mkldnn_classes():
    # [FUNCTION DOC] HELPER: Build once (cached) and return a mapping from class name to the
    # MKLDNN ScriptModule classes: linear, conv base, conv 1d/2d/3d, batch norm, prelu. Defined
    # lazily because creating ScriptModule subclasses compiles script methods.
    class MkldnnLinear(torch.jit.ScriptModule):
        # [CLASS DOC] MKLDNN version of a linear layer. Holds buffers weight (converted to
        # the given dtype) and bias (fp32, zeros if the dense layer has none). Forward converts
        # a dense input to MKLDNN layout and back; pickling stores dense tensors.
        def __init__(self, dense_module, dtype) -> None:
            # [FUNCTION DOC] Convert the dense module's weight and bias to MKLDNN buffers.
            super().__init__()
            self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))
            if dense_module.bias is not None:
                # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
                # we use fp32 dtype.
                self.register_buffer('bias', dense_module.bias.to_mkldnn())
            else:
                # TODO: Remove this once ScriptModule supports registering None buffer
                self.register_buffer(
                    'bias',
                    torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

        @torch.jit.script_method
        def __getstate__(self):
            # [FUNCTION DOC] Return (dense weight, dense bias, training flag) for pickling.
            return (self.weight.to_dense(), self.bias.to_dense(), self.training)

        @torch.jit.script_method
        def __setstate__(self, state):
            # [FUNCTION DOC] Restore weight and bias as MKLDNN tensors and the training flag.
            self.weight = state[0].to_mkldnn()
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

        @torch.jit.script_method
        def forward(self, x):
            # [FUNCTION DOC] Apply the MKLDNN linear op; a dense input is converted to MKLDNN
            # layout first and the result converted back to dense. Returns the output tensor.
            x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
            y_mkldnn = torch._C._nn.mkldnn_linear(x_mkldnn, self.weight, self.bias)
            y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
            return y

    class _MkldnnConvNd(torch.jit.ScriptModule):
        # [CLASS DOC] Shared base of the MKLDNN conv modules. Stores stride, padding, dilation
        # and groups as constants and a bias buffer (fp32, zeros if absent); subclasses
        # register the weight buffer. Forward runs an MKLDNN convolution.
        """Common base of MkldnnConv1d and MkldnnConv2d."""

        __constants__ = ['stride', 'padding', 'dilation', 'groups']

        def __init__(self, dense_module) -> None:
            # [FUNCTION DOC] Copy conv hyperparameters from the dense module and register bias.
            super().__init__()

            self.stride = dense_module.stride
            self.padding = dense_module.padding
            self.dilation = dense_module.dilation
            self.groups = dense_module.groups

            if dense_module.bias is not None:
                self.register_buffer('bias', dense_module.bias.to_mkldnn())
            else:
                # Bias can be fp32 or bf16 for OneDNN bf16 path, but for good accuracy,
                # we use fp32 dtype.
                # TODO: Remove this once ScriptModule supports registering None buffer
                self.register_buffer(
                    'bias',
                    torch.zeros([dense_module.weight.size(0)], dtype=torch.float).to_mkldnn())

        @torch.jit.script_method
        def __getstate__(self):
            # [FUNCTION DOC] Return (dense weight, dense bias, training flag) for pickling.
            return (self.weight.to_dense(), self.bias.to_dense(), self.training)

        @torch.jit.script_method
        def forward(self, x):
            # [FUNCTION DOC] Run the MKLDNN convolution of x with the stored weight and bias.
            return torch.mkldnn_convolution(
                x,
                self.weight,
                self.bias,
                self.padding,
                self.stride,
                self.dilation,
                self.groups)

    class MkldnnConv1d(_MkldnnConvNd):
        # [CLASS DOC] MKLDNN 1d convolution; weight is converted to the given dtype layout.
        def __init__(self, dense_module, dtype) -> None:
            # [FUNCTION DOC] Initialize the base, then register the converted weight buffer.
            super().__init__(dense_module)

            self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

        @torch.jit.script_method
        def __setstate__(self, state):
            # [FUNCTION DOC] Restore weight and bias as MKLDNN tensors and the training flag.
            self.weight = state[0].to_mkldnn()
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

    class MkldnnConv2d(_MkldnnConvNd):
        # [CLASS DOC] MKLDNN 2d convolution; weight is reordered into the blocked layout
        # that suits the conv parameters.
        def __init__(self, dense_module, dtype) -> None:
            # [FUNCTION DOC] Initialize the base, then register the reordered weight buffer.
            super().__init__(dense_module)

            self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv2d_weight(
                dense_module.weight.to_mkldnn(dtype),
                self.padding,
                self.stride,
                self.dilation,
                self.groups))

        @torch.jit.script_method
        def __setstate__(self, state):
            # [FUNCTION DOC] Restore the reordered weight, MKLDNN bias and training flag.
            self.weight = torch._C._nn.mkldnn_reorder_conv2d_weight(
                state[0].to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups)
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

    class MkldnnConv3d(_MkldnnConvNd):
        # [CLASS DOC] MKLDNN 3d convolution; weight is reordered into the blocked layout
        # that suits the conv parameters.
        def __init__(self, dense_module, dtype) -> None:
            # [FUNCTION DOC] Initialize the base, then register the reordered weight buffer.
            super().__init__(dense_module)

            self.register_buffer('weight', torch._C._nn.mkldnn_reorder_conv3d_weight(
                dense_module.weight.to_mkldnn(dtype),
                self.padding,
                self.stride,
                self.dilation,
                self.groups))

        @torch.jit.script_method
        def __setstate__(self, state):
            # [FUNCTION DOC] Restore the reordered weight, MKLDNN bias and training flag.
            self.weight = torch._C._nn.mkldnn_reorder_conv3d_weight(
                state[0].to_mkldnn(),
                self.padding,
                self.stride,
                self.dilation,
                self.groups)
            self.bias = state[1].to_mkldnn()
            self.training = state[2]

    class MkldnnBatchNorm(torch.jit.ScriptModule):
        # [CLASS DOC] Inference only MKLDNN batch norm. Holds weight, bias, running mean and
        # variance buffers (MKLDNN layout) and the momentum derived factor and eps constants.
        __constants__ = ['exponential_average_factor', 'eps']

        def __init__(self, dense_module) -> None:
            # [FUNCTION DOC] Convert a dense batch norm. Raises AssertionError unless the module
            # is in eval mode and tracks running stats and has affine parameters. A None
            # momentum becomes an averaging factor of 0.0.
            super().__init__()

            if dense_module.training:
                raise AssertionError("Only support eval mode batchnorm for mkldnn path now")
            if not dense_module.track_running_stats:
                raise AssertionError("Only support track_running_stats=True for mkldnn path now")
            if not dense_module.affine:
                raise AssertionError("Only support affine=True for mkldnn path now")

            if dense_module.momentum is None:
                self.exponential_average_factor = 0.0
            else:
                self.exponential_average_factor = dense_module.momentum
            self.eps = dense_module.eps

            self.register_buffer('weight', dense_module.weight.to_mkldnn())
            self.register_buffer('bias', dense_module.bias.to_mkldnn())
            self.register_buffer('running_mean', dense_module.running_mean.to_mkldnn())
            self.register_buffer('running_var', dense_module.running_var.to_mkldnn())

        @torch.jit.script_method
        def __getstate__(self):
            # [FUNCTION DOC] Return dense (weight, bias, running_mean, running_var, training).
            weight = self.weight.to_dense()
            bias = self.bias.to_dense()
            running_mean = self.running_mean.to_dense()
            running_var = self.running_var.to_dense()
            return (weight, bias, running_mean, running_var, self.training)

        @torch.jit.script_method
        def __setstate__(self, state):
            # [FUNCTION DOC] Restore the four buffers as MKLDNN tensors and the training flag.
            self.weight = state[0].to_mkldnn()
            self.bias = state[1].to_mkldnn()
            self.running_mean = state[2].to_mkldnn()
            self.running_var = state[3].to_mkldnn()
            self.training = state[4]

        @torch.jit.script_method
        def forward(self, x):
            # [FUNCTION DOC] Batch normalize x in inference mode with the stored statistics.
            return torch.batch_norm(
                x,
                self.weight,
                self.bias,
                self.running_mean,
                self.running_var,
                False,  # training
                self.exponential_average_factor,
                self.eps,
                False,  # cuda_enabled
            )

    class MkldnnPrelu(torch.jit.ScriptModule):
        # [CLASS DOC] MKLDNN PReLU holding a weight buffer converted to the given dtype.
        def __init__(self, dense_module, dtype) -> None:
            # [FUNCTION DOC] Register the dense module's weight converted to MKLDNN layout.
            super().__init__()
            self.register_buffer('weight', dense_module.weight.to_mkldnn(dtype))

        @torch.jit.script_method
        def __getstate__(self):
            # [FUNCTION DOC] Return (dense weight, training flag) for pickling.
            return (self.weight.to_dense(), self.training)

        @torch.jit.script_method
        def __setstate__(self, state):
            # [FUNCTION DOC] Restore the weight as an MKLDNN tensor and the training flag.
            self.weight = state[0].to_mkldnn()
            self.training = state[1]

        @torch.jit.script_method
        def forward(self, x):
            # [FUNCTION DOC] Apply PReLU in MKLDNN layout; dense input and output are converted.
            x_mkldnn = x if x.is_mkldnn else x.to_mkldnn()
            y_mkldnn = torch.prelu(x_mkldnn, self.weight)
            y = y_mkldnn if x.is_mkldnn else y_mkldnn.to_dense()
            return y

    return {
        'MkldnnLinear': MkldnnLinear,
        '_MkldnnConvNd': _MkldnnConvNd,
        'MkldnnConv1d': MkldnnConv1d,
        'MkldnnConv2d': MkldnnConv2d,
        'MkldnnConv3d': MkldnnConv3d,
        'MkldnnBatchNorm': MkldnnBatchNorm,
        'MkldnnPrelu': MkldnnPrelu,
    }


def __getattr__(name):
    # [FUNCTION DOC] Module level lazy attribute hook: returns the named Mkldnn class, building
    # the classes on first use. Raises AttributeError for unknown names.
    classes = _get_mkldnn_classes()
    if name in classes:
        return classes[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def to_mkldnn(module, dtype=torch.float):
    # [FUNCTION DOC] Recursively convert a module tree: Linear, Conv1d/2d/3d, BatchNorm2d/3d
    # and PReLU become MKLDNN modules, others are kept. Children are re-attached to the new
    # modules by name (mutating them). Returns the converted root. Raises AssertionError unless
    # dtype is float, bfloat16 or half.
    if dtype not in (torch.float, torch.bfloat16, torch.half):
        raise AssertionError("MKLDNN only support float, bfloat16, and half path now")

    classes = _get_mkldnn_classes()
    MkldnnLinear = classes['MkldnnLinear']
    MkldnnConv1d = classes['MkldnnConv1d']
    MkldnnConv2d = classes['MkldnnConv2d']
    MkldnnConv3d = classes['MkldnnConv3d']
    MkldnnBatchNorm = classes['MkldnnBatchNorm']
    MkldnnPrelu = classes['MkldnnPrelu']

    def m_fn(m, d):
        # [FUNCTION DOC] HELPER: Return the MKLDNN replacement for module m with dtype d,
        # or m itself when its type has no MKLDNN equivalent.
        if isinstance(m, torch.nn.Linear):
            return MkldnnLinear(m, d)
        elif isinstance(m, torch.nn.Conv1d):
            return MkldnnConv1d(m, d)
        elif isinstance(m, torch.nn.Conv2d):
            return MkldnnConv2d(m, d)
        elif isinstance(m, torch.nn.Conv3d):
            return MkldnnConv3d(m, d)
        elif isinstance(m, (torch.nn.BatchNorm2d, torch.nn.BatchNorm3d)):
            # For batchnorm bf16 path, OneDNN requires weight and bias need fp32 dtype.
            # so it doesn't need dtype argument.
            return MkldnnBatchNorm(m)
        elif isinstance(m, torch.nn.PReLU):
            return MkldnnPrelu(m, d)
        else:
            return m

    def m_fn_rec(m, d):
        # [FUNCTION DOC] HELPER: Convert m, then recursively convert and set each named child.
        new_m = m_fn(m, d)
        for name, sub_m in m.named_children():
            setattr(new_m, name, m_fn_rec(sub_m, d))
        return new_m

    return m_fn_rec(module, dtype)
