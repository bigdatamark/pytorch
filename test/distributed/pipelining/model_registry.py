# [MODULE DOC] Model zoo for the pipelining tests. Defines small nn.Modules with pipe_split
# markers or split_spec mappings (multi-layer MLPs, kwargs models, parameter aliasing,
# multi-output autograd ops, flex attention transformer, custom dX/dW backward, conditional
# grad blocks). Import-time side effect: compiles flex_attention into
# _compiled_flex_attention and defines _flex_attention_kernel_options (Triton backend).
# Copyright (c) Meta Platforms, Inc. and affiliates
# Owner(s): ["oncall: distributed"]
# This file is a model zoo for testing torch.distributed.pipelining.
import torch
from torch.autograd import Function
from torch.distributed.pipelining import pipe_split, SplitPoint
from torch.distributed.tensor import DTensor
from torch.nn.attention.flex_attention import BlockMask, flex_attention


_compiled_flex_attention = torch.compile(flex_attention)
_flex_attention_kernel_options = {
    "BACKEND": "TRITON",
    "PRESCALE_QK": False,
    "ROWS_GUARANTEED_SAFE": True,
    "BLOCKS_ARE_CONTIGUOUS": True,
}


class ExampleCode(torch.nn.Module):
    # [CLASS DOC] Test model: two parameter matmuls plus up to seven Linear layers, with a
    # pipe_split call between stages. The number of stages (splits, at most 8) selects which
    # later Linear/relu blocks run. Also holds a non-grad buffer cval to test passing
    # constants across stage boundaries.
    def __init__(self, d_hid, splits=2):
        # [FUNCTION DOC] Builds parameters mm_param0/mm_param1 (d_hid x d_hid, random),
        # buffer cval, and seven Linear(d_hid, d_hid) layers. Raises AssertionError if
        # splits is greater than 8; stores splits.
        if not (splits <= 8):
            raise AssertionError(f"Expected splits <= 8, got {splits}")
        super().__init__()
        self.splits = splits
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.cval = torch.nn.Buffer(torch.randn((d_hid,), requires_grad=False))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)
        self.lin3 = torch.nn.Linear(d_hid, d_hid)
        self.lin4 = torch.nn.Linear(d_hid, d_hid)
        self.lin5 = torch.nn.Linear(d_hid, d_hid)
        self.lin6 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        # [FUNCTION DOC] Runs matmul, relu, lin0, pipe_split, relu plus the cval constant,
        # matmul, then for each additional split (up to splits) a pipe_split, a Linear layer
        # and relu. Returns the output tensor.
        x = torch.mm(x, self.mm_param0)
        x = torch.relu(x)
        # try passing a value that doesn't require_grad across skip boundaries
        a_constant = self.cval.clone()
        x = self.lin0(x)
        pipe_split()
        x = torch.relu(x) + a_constant
        x = torch.mm(x, self.mm_param1)
        if self.splits > 2:
            pipe_split()
            x = self.lin1(x)
            x = torch.relu(x)
        if self.splits > 3:
            pipe_split()
            x = self.lin2(x)
            x = torch.relu(x)
        if self.splits > 4:
            pipe_split()
            x = self.lin3(x)
            x = torch.relu(x)
        if self.splits > 5:
            pipe_split()
            x = self.lin4(x)
            x = torch.relu(x)
        if self.splits > 6:
            pipe_split()
            x = self.lin5(x)
            x = torch.relu(x)
        if self.splits > 7:
            pipe_split()
            x = self.lin6(x)
            x = torch.relu(x)
        return x


class ModelWithKwargs(torch.nn.Module):
    # [CLASS DOC] Test model whose forward takes an extra keyword tensor y (default zeros of
    # DEFAULT_BATCH_SIZE x DEFAULT_DHID) added after the first matmul. Same split structure
    # as ExampleCode with up to eight Linear layers (splits at most 8).
    DEFAULT_DHID = 512
    DEFAULT_BATCH_SIZE = 256

    def __init__(self, d_hid: int = DEFAULT_DHID, splits=2):
        # [FUNCTION DOC] Creates two random d_hid x d_hid parameters and eight Linear layers.
        # Raises AssertionError if splits is greater than 8. Stores splits.
        if not (splits <= 8):
            raise AssertionError(f"Expected splits <= 8, got {splits}")
        super().__init__()
        self.splits = splits
        self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.mm_param1 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin0 = torch.nn.Linear(d_hid, d_hid)
        self.lin1 = torch.nn.Linear(d_hid, d_hid)
        self.lin2 = torch.nn.Linear(d_hid, d_hid)
        self.lin3 = torch.nn.Linear(d_hid, d_hid)
        self.lin4 = torch.nn.Linear(d_hid, d_hid)
        self.lin5 = torch.nn.Linear(d_hid, d_hid)
        self.lin6 = torch.nn.Linear(d_hid, d_hid)
        self.lin7 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y=torch.zeros(DEFAULT_BATCH_SIZE, DEFAULT_DHID)):
        # [FUNCTION DOC] Computes matmul with mm_param0, adds y, applies lin0 and relu, calls
        # pipe_split, then matmul with mm_param1, lin1, relu, and further split stages
        # depending on splits. Returns the output tensor.
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = self.lin0(x)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        if self.splits > 2:
            pipe_split()
            x = self.lin2(x)
            x = torch.relu(x)
        if self.splits > 3:
            pipe_split()
            x = self.lin3(x)
            x = torch.relu(x)
        if self.splits > 4:
            pipe_split()
            x = self.lin4(x)
            x = torch.relu(x)
        if self.splits > 5:
            pipe_split()
            x = self.lin5(x)
            x = torch.relu(x)
        if self.splits > 6:
            pipe_split()
            x = self.lin6(x)
            x = torch.relu(x)
        if self.splits > 7:
            pipe_split()
            x = self.lin7(x)
            x = torch.relu(x)
        return x


class ModelWithParamAlias(torch.nn.Module):
    # [CLASS DOC] Test model where mm_param0/mm_param1 are the same Parameter and lin0/lin1
    # are the same Linear, to test pipeline splitting with aliased parameters and modules.
    default_dhid = 512
    default_batch_size = 256

    def __init__(self, d_hid: int = default_dhid):
        # [FUNCTION DOC] Creates one shared Parameter bound to both mm_param names and one
        # shared Linear bound to both lin names.
        super().__init__()
        self.mm_param1 = self.mm_param0 = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.lin1 = self.lin0 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, y):
        # [FUNCTION DOC] Matmul with the shared parameter, add y, Linear, relu, pipe_split,
        # then matmul, Linear and relu again with the same shared weights. Returns output.
        x = torch.mm(x, self.mm_param0)
        x = x + y
        x = self.lin0(x)
        x = torch.relu(x)
        pipe_split()
        x = torch.mm(x, self.mm_param1)
        x = self.lin1(x)
        x = torch.relu(x)
        return x


# MLP Layer
class MLPModule(torch.nn.Module):
    # [CLASS DOC] Two-layer MLP block: Linear, ReLU, Linear, all d_hid wide.
    def __init__(self, d_hid: int):
        # [FUNCTION DOC] Creates net1, relu and net2 submodules of width d_hid.
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x):
        # [FUNCTION DOC] Applies net1, relu, net2 in order and returns the result.
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        return x


class MLPKWargModule(torch.nn.Module):
    # [CLASS DOC] MLP block like MLPModule that also records layer_num and accepts an unused
    # keyword argument, used to test kwargs passing through pipeline stages.
    def __init__(self, d_hid: int, layer_num):
        # [FUNCTION DOC] Creates net1, relu, net2 of width d_hid and stores layer_num.
        super().__init__()
        self.net1 = torch.nn.Linear(d_hid, d_hid)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(d_hid, d_hid)
        self.layer_num = layer_num

    def forward(self, x, unused_kwarg: torch.Tensor = torch.zeros(1)):
        # [FUNCTION DOC] Applies net1, relu, net2 and returns only the activation;
        # unused_kwarg is ignored (extra-output variants are commented out as a TODO).
        x = self.net1(x)
        x = self.relu(x)
        x = self.net2(x)
        # Test when only 1 module has extra outputs
        # TODO: handle this case later
        # if self.layer_num == 0:
        #     return x, unused_kwarg
        # else:
        #     return x
        return x


# Multi-MLP model
class MultiMLP(torch.nn.Module):
    # [CLASS DOC] Stack of n_layers MLPModule blocks. split_spec maps every layer from index
    # 1 on to SplitPoint.BEGINNING so each layer after the first starts a new stage.
    def __init__(self, d_hid: int, n_layers: int = 2):
        # [FUNCTION DOC] Builds the ModuleList of MLPModule layers and the split_spec mapping.
        super().__init__()
        self.layers = torch.nn.ModuleList([MLPModule(d_hid) for _ in range(n_layers)])
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x):
        # [FUNCTION DOC] Passes x through each layer sequentially and returns the result.
        for layer in self.layers:
            x = layer(x)
        return x


class TwoInputOutputOp(torch.autograd.Function):
    # [CLASS DOC] HELPER: Autograd Function that returns both of its inputs unchanged as two
    # outputs, with backward passing the two gradients straight through. Used to create
    # multi-output intermediates.
    @staticmethod
    def forward(ctx, input: torch.Tensor, weight: torch.Tensor):
        # [FUNCTION DOC] HELPER: Returns the input and weight tensors unchanged as a tuple.
        return input, weight

    @staticmethod
    def backward(ctx, grad_input, grad_weight):
        # [FUNCTION DOC] HELPER: Returns grad_input and grad_weight unchanged.
        return grad_input, grad_weight


# Model with multi-output intermediates
class MultiInterMediateModel(torch.nn.Module):
    # [CLASS DOC] Test model with a learnable weight w; forward splits the input into two
    # parts along dim 1 and passes one part and w through a two-output autograd op.
    def __init__(self, weight_shape: list[int]):
        # [FUNCTION DOC] Stores weight_shape (also used as split sizes) and creates the
        # random Parameter w of that shape.
        super().__init__()
        self.shape = weight_shape
        self.w = torch.nn.Parameter(torch.randn(*weight_shape))

    def forward(self, x):
        # [FUNCTION DOC] Splits x along dim 1 into sizes self.shape (a, b), passes a and w
        # through TwoInputOutputOp, matrix-multiplies a by w, and returns the product times b.
        a, b = torch.split(x, self.shape, dim=1)
        a, w = TwoInputOutputOp.apply(a, self.w)
        a = torch.matmul(a, w)
        return a * b


# Multi-MLP with kwargs model
class MultiMLPKwargs(torch.nn.Module):
    # [CLASS DOC] Stack of MLPKWargModule layers whose forward accepts an unused keyword
    # tensor. Has the same split_spec scheme as MultiMLP (split before each layer after 0).
    def __init__(self, d_hid: int, n_layers: int = 2):
        # [FUNCTION DOC] Builds n_layers MLPKWargModule layers and the split_spec mapping.
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MLPKWargModule(d_hid, i) for i in range(n_layers)]
        )
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x, unused_kwarg: torch.Tensor = torch.zeros(1)):
        # [FUNCTION DOC] Runs x through each layer sequentially (unused_kwarg is ignored)
        # and returns the result.
        for layer in self.layers:
            # TODO: handle this case later
            # if layer.layer_num == 0:
            #     x, _ = layer(x, unused_kwarg)
            # else:
            #     x = layer(x)
            x = layer(x)
        return x


class FlexAttentionTransformerBlock(torch.nn.Module):
    # [CLASS DOC] Transformer block using compiled flex attention with a block mask: layer
    # norm, bias-free q/k/v/output projections, then a LayerNorm + GELU feed-forward block,
    # with residual connections. Requires model_dim divisible by num_heads.
    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int):
        # [FUNCTION DOC] Creates norms, projections and feed-forward Sequential. Raises
        # ValueError if model_dim is not divisible by num_heads.
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError(
                f"model_dim ({model_dim}) must be divisible by num_heads ({num_heads})"
            )
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.attention_norm = torch.nn.LayerNorm(model_dim)
        self.q_proj = torch.nn.Linear(model_dim, model_dim, bias=False)
        self.k_proj = torch.nn.Linear(model_dim, model_dim, bias=False)
        self.v_proj = torch.nn.Linear(model_dim, model_dim, bias=False)
        self.output_proj = torch.nn.Linear(model_dim, model_dim, bias=False)
        self.ffn_norm = torch.nn.LayerNorm(model_dim)
        self.feed_forward = torch.nn.Sequential(
            torch.nn.Linear(model_dim, ffn_dim, bias=False),
            torch.nn.GELU(),
            torch.nn.Linear(ffn_dim, model_dim, bias=False),
        )

    def forward(self, x: torch.Tensor, *, block_mask: BlockMask) -> torch.Tensor:
        # [FUNCTION DOC] Applies attention (via the compiled flex_attention with the given
        # block_mask and Triton kernel options) and the feed-forward, each with a residual.
        # Input x is (batch, seq, model_dim); returns the same shape.
        batch_size, seq_len, model_dim = x.shape
        normalized = self.attention_norm(x)

        def project(projection: torch.nn.Linear) -> torch.Tensor:
            # [FUNCTION DOC] HELPER: Applies the projection to the normalized input and
            # reshapes to (batch, heads, seq, head_dim).
            return (
                projection(normalized)
                .view(batch_size, seq_len, self.num_heads, self.head_dim)
                .transpose(1, 2)
            )

        attention = _compiled_flex_attention(
            project(self.q_proj),
            project(self.k_proj),
            project(self.v_proj),
            block_mask=block_mask,
            kernel_options=_flex_attention_kernel_options,
        )
        attention = (
            attention.transpose(1, 2).contiguous().view(batch_size, seq_len, model_dim)
        )
        x = x + self.output_proj(attention)
        return x + self.feed_forward(self.ffn_norm(x))


class FlexAttentionTransformer(torch.nn.Module):
    # [CLASS DOC] Stack of n_layers FlexAttentionTransformerBlock modules with a split_spec
    # that starts a new pipeline stage at each layer after the first.
    def __init__(self, model_dim: int, num_heads: int, ffn_dim: int, n_layers: int):
        # [FUNCTION DOC] Builds the layers ModuleList and split_spec mapping.
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                FlexAttentionTransformerBlock(model_dim, num_heads, ffn_dim)
                for _ in range(n_layers)
            ]
        )
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }

    def forward(self, x: torch.Tensor, *, block_mask: BlockMask) -> torch.Tensor:
        # [FUNCTION DOC] Runs x through each block, passing block_mask by keyword. Returns x.
        for layer in self.layers:
            x = layer(x, block_mask=block_mask)
        return x


class CustomLinearDx(Function):
    # [CLASS DOC] HELPER: Custom autograd Function for a linear layer whose backward only
    # computes the input gradient (dX). It stashes grad_output and the layer input in the
    # module's cached_context so weight gradients can be computed later.
    @staticmethod
    def forward(ctx, input_val, weight, bias, module, layer_idx):
        # [FUNCTION DOC] HELPER: Saves tensors, stores module and layer_idx on ctx, and
        # returns the linear result of input times transposed weight plus bias.
        ctx.save_for_backward(input_val, weight, bias)
        ctx.module = module
        ctx.layer_idx = layer_idx
        return input_val.mm(weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Returns the input gradient (grad_output times weight) and
        # None for other inputs. Side effect: appends clones of grad_output and the layer
        # input to the module's cached_context lists for later dW computation.
        input_val, weight, _ = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        ctx.module.cached_context[ctx.layer_idx].append(grad_output.clone())
        ctx.module.cached_context[str(ctx.layer_idx) + "_input"].append(
            input_val.clone()
        )
        return grad_input, None, None, None, None


class CustomLinearDxDw(Function):
    # [CLASS DOC] HELPER: Custom autograd Function for a linear layer whose backward
    # computes input, weight and bias gradients in the usual way.
    @staticmethod
    def forward(ctx, input_val, weight, bias):
        # [FUNCTION DOC] HELPER: Saves tensors and returns input times transposed weight plus bias.
        ctx.save_for_backward(input_val, weight, bias)
        return input_val.mm(weight.t()) + bias

    @staticmethod
    def backward(ctx, grad_output):
        # [FUNCTION DOC] HELPER: Returns gradients for input, weight and bias (bias gradient
        # is grad_output summed over the batch dimension).
        input_val, weight, _ = ctx.saved_tensors
        grad_input = grad_output.mm(weight)
        grad_weight = grad_output.t().mm(input_val)
        grad_bias = grad_output.sum(0)
        return grad_input, grad_weight, grad_bias


class MLPModuleWithDw(torch.nn.Module):
    # [CLASS DOC] Two-layer MLP with explicit parameters, supporting two backward modes
    # selected by use_custom_logic: normal (full gradients) or split (dX only in backward,
    # weight/bias gradients deferred via cached_context and computed by compute_dW).
    def __init__(self, d_hid: int):
        # [FUNCTION DOC] Creates fc1/fc2 weight and bias parameters initialized uniformly in
        # (-0.001, 0.001), the cached_context lists for grad outputs and inputs, and sets
        # use_custom_logic to False.
        super().__init__()
        self.fc1_weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.fc1_bias = torch.nn.Parameter(torch.randn(d_hid))
        self.fc2_weight = torch.nn.Parameter(torch.randn(d_hid, d_hid))
        self.fc2_bias = torch.nn.Parameter(torch.randn(d_hid))

        torch.nn.init.uniform_(self.fc1_weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2_weight, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc1_bias, -0.001, 0.001)
        torch.nn.init.uniform_(self.fc2_bias, -0.001, 0.001)

        self.cached_context = {}
        self.cached_context["fc1"] = []
        self.cached_context["fc2"] = []
        self.cached_context["fc1_input"] = []
        self.cached_context["fc2_input"] = []

        self.use_custom_logic = False

    def forward(self, x):
        # [FUNCTION DOC] Runs two linear layers with relu between. Uses the full-gradient
        # function when use_custom_logic is False, otherwise the dX-only function that
        # caches data. Stores the hidden activation on self.hidden. Returns the output.
        if not self.use_custom_logic:
            self.hidden = CustomLinearDxDw.apply(x, self.fc1_weight, self.fc1_bias)
            self.hidden = torch.nn.functional.relu(self.hidden)
            output = CustomLinearDxDw.apply(self.hidden, self.fc2_weight, self.fc2_bias)
            return output

        self.hidden = CustomLinearDx.apply(
            x, self.fc1_weight, self.fc1_bias, self, "fc1"
        )
        self.hidden = torch.nn.functional.relu(self.hidden)
        output = CustomLinearDx.apply(
            self.hidden, self.fc2_weight, self.fc2_bias, self, "fc2"
        )
        return output

    def compute_dW(self):
        # [FUNCTION DOC] Pops the oldest cached grad outputs and inputs for both layers and
        # computes weight and bias gradients, then adds them into the existing .grad of each
        # parameter or assigns them if no grad exists yet. Raises IndexError if the caches
        # are empty.
        grad_output_fc1 = self.cached_context["fc1"].pop(0)
        grad_output_fc2 = self.cached_context["fc2"].pop(0)
        cached_input_fc1 = self.cached_context["fc1_input"].pop(0)
        cached_input_fc2 = self.cached_context["fc2_input"].pop(0)

        dW2 = grad_output_fc2.t().mm(cached_input_fc2)
        db2 = grad_output_fc2.sum(0)

        dW1 = grad_output_fc1.t().mm(cached_input_fc1)
        db1 = grad_output_fc1.sum(0)

        if self.fc1_weight.grad is not None:
            self.fc1_weight.grad += dW1
            self.fc1_bias.grad += db1
            self.fc2_weight.grad += dW2
            self.fc2_bias.grad += db2
        else:
            self.fc1_weight.grad = dW1
            self.fc1_bias.grad = db1
            self.fc2_weight.grad = dW2
            self.fc2_bias.grad = db2

    def toggle(self):
        # [FUNCTION DOC] Flips use_custom_logic between full and split backward modes.
        self.use_custom_logic = not self.use_custom_logic


# Multi-MLP model With Dw
class MultiMLPWithDw(torch.nn.Module):
    # [CLASS DOC] Stack of MLPModuleWithDw layers with a split_spec like MultiMLP. toggle
    # switches all layers to the dX-only backward mode; compute_dW runs deferred weight
    # gradients in reverse layer order.
    def __init__(self, d_hid: int, n_layers: int = 2):
        # [FUNCTION DOC] Builds the layers and split_spec; use_custom_logic starts False.
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [MLPModuleWithDw(d_hid) for _ in range(n_layers)]
        )
        # For testing purpose only, this should be defined by user
        self.split_spec = {
            f"layers.{i}": SplitPoint.BEGINNING for i in range(1, n_layers)
        }
        self.use_custom_logic = False

    def forward(self, x):
        # [FUNCTION DOC] Passes x through each layer in order and returns the result.
        for layer in self.layers:
            x = layer(x)
        return x

    def toggle(self):
        # [FUNCTION DOC] Flips use_custom_logic on the stack and toggles every layer.
        self.use_custom_logic = not self.use_custom_logic
        for layer in self.layers:
            layer.toggle()

    def compute_dW(self):
        # [FUNCTION DOC] Computes deferred weight gradients for layers in reverse order.
        # Raises RuntimeError if toggle() has not enabled custom logic.
        if not self.use_custom_logic:
            raise RuntimeError("Need to call toggle() to enable custom backward and dW")

        for i in reversed(range(len(self.layers))):
            self.layers[i].compute_dW()


class ConditionalGradBlock(torch.nn.Module):
    # [CLASS DOC] Block whose forward either applies a Linear layer or, when conditional is
    # set and the flag's first element is falsy, returns a detached input that requires grad
    # (no parameter use). The flag is passed through as a second output.
    def __init__(self, d_hid, conditional=True):
        # [FUNCTION DOC] Stores the conditional switch and creates a d_hid Linear.
        super().__init__()
        self.conditional = conditional
        self.lin = torch.nn.Linear(d_hid, d_hid)

    def forward(self, x, flag):
        # [FUNCTION DOC] Converts flag to a local tensor if it is a DTensor, reads its first
        # element on the host, and picks the branch described in the class. Returns (y, flag).
        local_flag = flag.to_local() if isinstance(flag, DTensor) else flag
        if self.conditional and not bool(local_flag.flatten()[0].item()):
            y = x.detach().requires_grad_(True)
        else:
            y = self.lin(x)
        return y, flag


class ConditionalGradStack(torch.nn.Module):
    # [CLASS DOC] Stack of ConditionalGradBlock layers where only the last block is
    # unconditional (conditional for every index below n_layers - 1).
    def __init__(self, d_hid, n_layers):
        # [FUNCTION DOC] Builds n_layers ConditionalGradBlock modules.
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                ConditionalGradBlock(d_hid, conditional=i < n_layers - 1)
                for i in range(n_layers)
            ]
        )

    def forward(self, x, flag):
        # [FUNCTION DOC] Threads x and flag through all layers; returns (x, flag).
        for layer in self.layers:
            x, flag = layer(x, flag)
        return x, flag
