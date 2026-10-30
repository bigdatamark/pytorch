# [MODULE DOC] Eager PyTorch backend for the tensor-expression benchmark harness. Defines one
# engine class whose methods wrap torch ops so benchmarks can call them through a common
# interface.
import torch


class TorchTensorEngine:
    # [CLASS DOC] Stateless engine exposing tensor creation, math, normalization, pooling,
    # convolution, matmul, reduction and autograd helpers implemented with eager torch. The
    # benchmark base class copies these methods onto each benchmark instance.
    def rand(self, shape, device=None, dtype=None, requires_grad=False):
        # [FUNCTION DOC] Creates a uniform random tensor of the given shape on the given
        # device and dtype, with the requested gradient tracking.
        return torch.rand(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def randn(self, shape, device=None, dtype=None, requires_grad=False):
        # [FUNCTION DOC] Creates a standard normal random tensor of the given shape on the
        # given device and dtype, with the requested gradient tracking.
        return torch.randn(
            shape, device=device, dtype=dtype, requires_grad=requires_grad
        )

    def nchw_rand(self, shape, device=None, requires_grad=False):
        # [FUNCTION DOC] Creates a uniform random tensor for an NCHW shape through the plain
        # random method; accepts no dtype, so the default dtype is used.
        return self.rand(shape, device=device, requires_grad=requires_grad)

    def reset(self, _):
        # [FUNCTION DOC] Device reset hook; does nothing for the eager engine.
        pass

    def rand_like(self, v):
        # [FUNCTION DOC] Returns a uniform random tensor with the same shape and dtype as the
        # argument.
        return torch.rand_like(v)

    def numpy(self, t):
        # [FUNCTION DOC] Moves the tensor to the cpu and returns it as a numpy array.
        return t.cpu().numpy()

    def mul(self, t1, t2):
        # [FUNCTION DOC] Returns the elementwise product of the two tensors.
        return t1 * t2

    def add(self, t1, t2):
        # [FUNCTION DOC] Returns the elementwise sum of the two operands.
        return t1 + t2

    def batch_norm(self, data, mean, var, training):
        # [FUNCTION DOC] Applies functional batch norm with the given running mean and
        # variance and training flag, and returns the result.
        return torch.nn.functional.batch_norm(data, mean, var, training=training)

    def instance_norm(self, data):
        # [FUNCTION DOC] Applies functional instance norm with default settings and returns
        # the result.
        return torch.nn.functional.instance_norm(data)

    def layer_norm(self, data, shape):
        # [FUNCTION DOC] Applies functional layer norm over the given trailing shape and
        # returns the result.
        return torch.nn.functional.layer_norm(data, shape)

    def sync_cuda(self):
        # [FUNCTION DOC] Blocks until all queued CUDA work finishes.
        torch.cuda.synchronize()

    def backward(self, tensors, grad_tensors, _):
        # [FUNCTION DOC] Runs autograd backward on the given tensors using the given gradient
        # tensors, accumulating gradients on leaves. The third argument is ignored.
        torch.autograd.backward(tensors, grad_tensors=grad_tensors)

    def sum(self, data, dims):
        # [FUNCTION DOC] Sums the tensor over the given dimensions and returns the result.
        return torch.sum(data, dims)

    def softmax(self, data, dim=None, dtype=None):
        # [FUNCTION DOC] Applies functional softmax over the given dimension with the given
        # output dtype and returns the result.
        return torch.nn.functional.softmax(data, dim, dtype)

    def cat(self, inputs, dim=0):
        # [FUNCTION DOC] Concatenates the given sequence of tensors along the dimension and
        # returns the result.
        return torch.cat(inputs, dim=dim)

    def clamp(self, data, min, max):
        # [FUNCTION DOC] Clamps the tensor to the given lower and upper bounds and returns the
        # result.
        return torch.clamp(data, min=min, max=max)

    def relu(self, data):
        # [FUNCTION DOC] Applies the functional relu and returns the result.
        return torch.nn.functional.relu(data)

    def tanh(self, data):
        # [FUNCTION DOC] Applies the hyperbolic tangent and returns the result.
        return torch.tanh(data)

    def max_pool2d(self, data, kernel_size, stride=1):
        # [FUNCTION DOC] Applies functional 2D max pooling with the kernel size and stride and
        # returns the result.
        return torch.nn.functional.max_pool2d(data, kernel_size, stride=stride)

    def avg_pool2d(self, data, kernel_size, stride=1):
        # [FUNCTION DOC] Applies functional 2D average pooling with the kernel size and stride
        # and returns the result.
        return torch.nn.functional.avg_pool2d(data, kernel_size, stride=stride)

    def conv2d_layer(self, ic, oc, kernel_size, groups=1):
        # [FUNCTION DOC] Creates and returns a new convolution layer module with the given
        # input and output channels, kernel size and groups.
        return torch.nn.Conv2d(ic, oc, kernel_size, groups=groups)

    def matmul(self, t1, t2):
        # [FUNCTION DOC] Returns the matrix product of the two tensors with torch broadcasting
        # and batching rules.
        return torch.matmul(t1, t2)

    def to_device(self, module, device):
        # [FUNCTION DOC] Moves the module to the device and returns it.
        return module.to(device)
