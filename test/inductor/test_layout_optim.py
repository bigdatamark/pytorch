# Owner(s): ["module: inductor"]
# [MODULE DOC] Tests for inductor convolution layout optimization (channels-last conversion).
# Checks accuracy against a float64 reference across graph breaks (manual or via DDP), training,
# mutation of views/bases of conv outputs, dynamic shapes and an nll_loss backward regression.
# Reads the USE_DDP_WRAPPER environment variable (default on) at import time. The test run only
# starts when a GPU is available.
import copy
import os
import random

import torch
from torch import nn
from torch._dynamo.utils import same
from torch._inductor import config
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_cuda import tf32_off
from torch.testing._internal.common_utils import skipIfXpu
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


USE_DDP_WRAPPER = os.environ.get("USE_DDP_WRAPPER", "1") == "1"


class Model2Conv(nn.Module):
    # [CLASS DOC] HELPER: Small two-convolution network (3 to dim channels, then dim to dim, both
    # 3x3 stride 2, no bias). Optionally calls a dynamo graph break between the convs.
    def __init__(self, dim=512, manual_graph_break=False):
        # [FUNCTION DOC] HELPER: Builds the two conv layers and stores the manual graph break
        # flag.
        super().__init__()
        self.conv1 = nn.Conv2d(3, dim, kernel_size=3, stride=2, bias=False)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size=3, stride=2, bias=False)
        self.manual_graph_break = manual_graph_break

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies conv1, optionally forces a dynamo graph break, applies
        # conv2 and returns the result.
        x = self.conv1(x)
        if self.manual_graph_break:
            torch._dynamo.graph_break()
        x = self.conv2(x)
        return x

    def get_example_inputs(self):
        # [FUNCTION DOC] HELPER: Returns a one-element tuple holding a random CPU tensor of shape
        # 2x3x16x16.
        return (torch.rand(2, 3, 16, 16),)


@skipIfXpu(msg="ccl doesn't currently work on the XPU stack")
class TestLayoutOptim(TestCase):
    # [CLASS DOC] Layout optimization tests; skipped on XPU. The class setup creates a single-rank
    # distributed process group (needed for the DDP wrapper), and each test compares compiled
    # output with eager and a float64 reference.
    @classmethod
    def setUpClass(cls):
        # [FUNCTION DOC] HELPER: Class-level setup. After the base setup, initializes a world size
        # one, rank zero process group over a TCP address on localhost with a random port, using
        # nccl for CUDA or ccl for XPU. Retries up to five times on RuntimeError, re-raising on
        # the last failure. KNOWN ISSUE: backend is unbound if GPU_TYPE is neither cuda nor xpu.
        super().setUpClass()

        import torch.distributed as dist

        # not use a fixed port for stress test
        tot_retry = 5
        for retry_no in range(tot_retry):
            try:
                port = random.randint(10000, 60000)
                if GPU_TYPE == "cuda":
                    backend = "nccl"
                elif GPU_TYPE == "xpu":
                    backend = "ccl"
                dist.init_process_group(
                    backend=backend,
                    init_method=f"tcp://localhost:{port}",
                    world_size=1,
                    rank=0,
                )
                break
            except RuntimeError:
                if retry_no == tot_retry - 1:
                    raise
                else:
                    continue

    def verify_accuracy(
        self, model_class, use_ddp_wrapper=USE_DDP_WRAPPER, is_train=False
    ):
        # [FUNCTION DOC] HELPER: Builds the model class on the GPU (with the manual graph break
        # flag on when DDP is not used), computes an eager result and a float64 reference result,
        # then compiles the model (wrapped in DDP if requested) and asserts the compiled output is
        # numerically close to the eager output judged against the float64 reference. In training
        # mode the compared outputs are lists of parameter gradients; otherwise a sum comparison
        # is printed first. Args: model_class, use_ddp_wrapper, is_train.
        # there are 2 potential ways to introduce graph breaks
        # 1. manually
        # 2. using DDP
        # if we are not using DDP to introduce graph breaks, do that manually
        def wrap_mod(m):
            # [FUNCTION DOC] HELPER: Returns the module unchanged in inference mode. In training
            # mode returns a function that runs the module, backpropagates the sum of its output
            # and returns a list of parameter gradients, substituting zeros for missing gradients.
            if is_train:

                def f(*inp):
                    # [FUNCTION DOC] HELPER: Runs forward on the given inputs, runs backward on
                    # the summed output and returns the list of gradients of all named parameters
                    # (zeros where absent).
                    x = m(*inp)
                    x.sum().backward()

                    grads = []
                    for _, param in m.named_parameters():
                        grad = param.grad
                        if param.grad is None:
                            grad = torch.zeros_like(param)
                        grads.append(grad)
                    return grads

                return f
            else:
                return m

        manual_graph_break = not use_ddp_wrapper
        mod = model_class(manual_graph_break=manual_graph_break).to(GPU_TYPE)
        inp = [t.to(GPU_TYPE) for t in mod.get_example_inputs()]
        expected_out = wrap_mod(mod)(*inp)

        fp64_mod = copy.deepcopy(mod).to(torch.float64)
        fp64_inp = [t.to(torch.float64) for t in copy.deepcopy(inp)]
        fp64_out = wrap_mod(fp64_mod)(*fp64_inp)

        if use_ddp_wrapper:
            from torch.nn.parallel import DistributedDataParallel as DDP

            ddp_wrapped_mod = DDP(mod)
            opt_mod = torch.compile(wrap_mod(ddp_wrapped_mod))
        else:
            opt_mod = torch.compile(wrap_mod(mod))
        actual_out = opt_mod(*inp)

        if is_train:
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))
        else:
            expected_sum = expected_out.sum()
            actual_sum = actual_out.sum()
            print(f"Expected sum {expected_sum}, actual sum {actual_sum}")
            self.assertTrue(same(expected_out, actual_out, fp64_ref=fp64_out))

    def verify_accuracy_for_infer(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the accuracy check with training off.
        self.verify_accuracy(*args, **kwargs, is_train=False)

    def verify_accuracy_for_train(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Forwards all arguments to the accuracy check with training on.
        self.verify_accuracy(*args, **kwargs, is_train=True)

    def test_2conv_with_graph_break(self):
        # [FUNCTION DOC] Inference accuracy check of the two-conv model with graph break handling.
        """
        Make sure graph break does not cause any accuracy issue.
        """
        self.verify_accuracy_for_infer(Model2Conv)

    def test_3conv_with_graph_break(self):
        # [FUNCTION DOC] Inference accuracy check of a three-conv model (patchify conv, depthwise
        # same-padded conv, pointwise conv) with graph break handling.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Three-conv model: a sequential of a patch-size strided conv and
            # a depthwise same-padding conv, then a 1x1 conv, with an optional dynamo graph break
            # between.
            def __init__(
                self, dim=512, patch_size=7, kernel_size=7, manual_graph_break=False
            ):
                # [FUNCTION DOC] HELPER: Builds the sequential block, the pointwise conv and
                # stores the manual graph break flag.
                super().__init__()
                self.seq = nn.Sequential(
                    nn.Conv2d(
                        3, dim, kernel_size=patch_size, stride=patch_size, bias=False
                    ),
                    nn.Conv2d(
                        dim, dim, kernel_size, groups=dim, padding="same", bias=False
                    ),
                )
                self.conv = nn.Conv2d(dim, dim, kernel_size=1, bias=False)
                self.manual_graph_break = manual_graph_break

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the sequential block, optionally a graph break, then
                # the 1x1 conv; returns the result.
                x = self.seq(x)
                if self.manual_graph_break:
                    torch._dynamo.graph_break()
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                # [FUNCTION DOC] HELPER: Returns a tuple with one random normal tensor of shape
                # 2x3x16x16.
                return (torch.randn(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    @torch.no_grad()
    def test_keep_output_layout_infer(self):
        # [FUNCTION DOC] Under no-grad, checks that compiled conv output keeps a layout allowing a
        # flattening view (a channels-last output would make the view fail). Also reused by the
        # freezing test.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Single 3 to 128 channel 3x3 conv model without bias.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds the conv layer.
                super().__init__()
                self.conv = nn.Conv2d(
                    3, 128, kernel_size=3, padding=1, stride=1, bias=False
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the conv and returns it.
                x = self.conv(x)
                return x

            def get_example_inputs(self):
                # [FUNCTION DOC] HELPER: Returns a tuple with one random normal tensor of shape
                # 2x3x5x5.
                return (torch.randn(2, 3, 5, 5),)

        mod = Model().to(GPU_TYPE)
        inp = [t.to(GPU_TYPE) for t in mod.get_example_inputs()]
        out = mod(*inp)

        opt_mod = torch.compile(mod)
        opt_out = opt_mod(*inp)

        # We should be able to do view on eager output
        out.view(5, -1)

        # We should be able to do view on the output of the optimized module
        # Note that if the output is channels last, the view op will fail.
        opt_out.view(5, -1)

    def test_keep_output_layout_with_freezing(self):
        # [FUNCTION DOC] Runs the output-layout test with the inductor freezing option enabled.
        with config.patch(
            {
                "freezing": True,
            }
        ):
            self.test_keep_output_layout_infer()

    def test_training_acc(self):
        # [FUNCTION DOC] Training-mode accuracy check of the two-conv model (compares gradients).
        self.verify_accuracy_for_train(Model2Conv)

    def test_mutate_view(self):
        # [FUNCTION DOC] Compiles a function that mutates a view of its input in place and checks
        # the base tensor sees the update (all elements doubled).
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/07228313fd017e2267101ff32edc6d64

        It shows that we will call copy_ to update the argument in the end. This
        guarantees the correctnesss.
        """

        @torch.compile
            # [FUNCTION DOC] HELPER: Creates a view of the input with a different shape and
            # multiplies it in place by two; returns nothing.
        def f(x):
            y = x.view(3, 2)
            y.mul_(2)

        x = torch.ones(2, 3).to(GPU_TYPE)
        f(x)
        self.assertTrue(torch.equal(x, torch.ones(2, 3).to(GPU_TYPE) * 2))

    def test_mutate_base(self):
        # [FUNCTION DOC] Compiles a function that makes a view, then mutates the base in place,
        # and checks the returned view reflects the update (all elements doubled).
        """
        The GraphModule passed to GraphLowering init method is like:
        https://gist.github.com/shunting314/fd60fe11d1f844c6db76aba7b06811bc

        It shows that the output of the graph is the mul node which contains
        the update we applied to the base tensor.
        """

        @torch.compile
            # [FUNCTION DOC] HELPER: Makes a reshaped view of the input, multiplies the input in
            # place by two and returns the view.
        def f(x):
            y = x.view(3, 2)
            x.mul_(2)
            return y

        x = torch.ones(2, 3).to(GPU_TYPE)
        y = f(x)
        self.assertTrue(torch.equal(y, torch.ones(3, 2).to(GPU_TYPE) * 2))

    @tf32_off()
    def test_mutate_base_for_conv_output(self):
        # [FUNCTION DOC] With TF32 disabled, checks accuracy when a view of a conv output is
        # returned after the conv output itself is mutated in place.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Conv model that flattens the conv output into a view, doubles
            # the conv output in place and returns the view.
            def __init__(self, manual_graph_break=False):
                # [FUNCTION DOC] HELPER: Builds the conv layer; the graph break argument is
                # ignored.
                super().__init__()
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, takes a flat view, mutates the conv output
                # in place and returns the view.
                x = self.conv(x)
                y = x.view(-1)
                x.mul_(2)
                return y

            def get_example_inputs(self):
                # [FUNCTION DOC] HELPER: Returns a tuple with one random tensor of shape
                # 2x3x16x16.
                return (torch.rand(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    @tf32_off()
    def test_mutate_view_for_conv_output(self):
        # [FUNCTION DOC] With TF32 disabled, checks accuracy when a flat view of a conv output is
        # mutated in place and the conv output itself is returned.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Conv model that mutates a flat view of the conv output in place
            # and returns the conv output.
            def __init__(self, manual_graph_break=False):
                # [FUNCTION DOC] HELPER: Builds the conv layer; the graph break argument is
                # ignored.
                super().__init__()
                self.conv = nn.Conv2d(3, 512, kernel_size=3, stride=2, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies conv, takes a flat view, doubles the view in
                # place and returns the conv output.
                x = self.conv(x)
                y = x.view(-1)
                y.mul_(2)
                return x

            def get_example_inputs(self):
                # [FUNCTION DOC] HELPER: Returns a tuple with one random tensor of shape
                # 2x3x16x16.
                return (torch.rand(2, 3, 16, 16),)

        self.verify_accuracy_for_infer(Model)

    def test_dynamic_shape_specialization(self):
        # [FUNCTION DOC] Compiles a function with dynamic shapes for several sizes, checks results
        # match eager and triggers the backward pass to ensure the backward graph compiles.
        """
        Previously in aot_autograd.py we compare strides of FakeTensor
        with real tensor. That cause dynamic dimensions of the FakeTensor
        being specialized to static shapes. This test protects against that.
        """

        def f(a, b):
            # [FUNCTION DOC] HELPER: Returns the sine of the first input plus the cosine of the
            # second.
            x = a.sin()
            y = b.cos()
            z = x + y
            return z

        for size in [4, 8, 16]:
            a = torch.randn(2, size, requires_grad=True).to(GPU_TYPE)
            b = torch.randn(2, size).to(GPU_TYPE)
            actual = torch.compile(f, dynamic=True)(a, b)
            self.assertTrue(torch.allclose(f(a, b), actual))

            # Trigger the compiling of the backward graph
            actual.sum().backward()

    def test_nll_loss_backward(self):
        # [FUNCTION DOC] Regression test for a layout issue in the 2D nll loss backward: builds a
        # conv plus linear model with cross entropy loss, runs forward and backward in eager and
        # checks that a second forward gives the same loss. KNOWN ISSUE: the compiled model is
        # created but never called, so compilation is not actually exercised.
        """
        Repro for issue https://github.com/pytorch/pytorch/issues/120759

        The CUDA implementation of aten.nll_loss2d_backward.default requires
        the self tensor (whose layout will be used to create grad_input)
        to be contiguous. Layout optimization may change the self tensor's layout
        and cause failure. We fix that by adding layout constraints to the
        fallback of aten.nll_loss2d_backward.default .
        """

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Model with a same-padded conv, a linear layer over the flattened
            # channel and frequency dims, and a cross entropy loss over transposed logits.
            def __init__(self, input_dim, num_classes):
                # [FUNCTION DOC] HELPER: Builds the conv layer and the linear output layer.
                super().__init__()
                self.conv = torch.nn.Conv2d(1, num_classes, 3, 1, padding="same")
                self.out = torch.nn.Linear(input_dim * num_classes, num_classes)

            def forward(self, x: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Runs conv, reshapes and applies the linear layer,
                # reshapes into class-logits form and returns the cross entropy loss against the
                # targets.
                x = self.conv(x)
                b, c, t, f = x.size()
                x = self.out(x.reshape(b, t, c * f))
                logits = x.reshape(x.size(0), x.size(2), x.size(1))
                loss = torch.nn.functional.cross_entropy(logits, targets)
                return loss

        device = GPU_TYPE
        batch_size = 48
        seq_len = 144
        input_dim = 39
        num_classes = 111

        model = MyModel(input_dim, num_classes)
        model.to(device)

        opt_model = torch.compile(model)  # noqa: F841

        x = torch.ones((batch_size, 1, seq_len, input_dim), device=device)
        targets = torch.randint(
            0, num_classes - 1, (batch_size, seq_len), device=device, dtype=torch.int64
        )

        loss = model(x, targets)
        loss.backward()

        ref = model(x, targets)
        self.assertTrue(torch.allclose(ref, loss))


if __name__ == "__main__":
    if HAS_GPU:
        run_tests()
