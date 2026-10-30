# Owner(s): ["module: mkldnn"]
# [MODULE DOC] Tests for the oneDNN Graph (LLGA) fuser in TorchScript on CPU. Traces small ops,
# fusion-pattern modules and TorchVision models, runs them in float32 and bfloat16, and checks
# both numerical equality with eager mode and the number and contents of fusion group nodes in the
# optimized graph. Import-time side effects: probes the CPU flags file on Linux for AVX512,
# detects TorchVision availability, builds one test per vision model and dtype on the model test
# class, and instantiates device-type tests for the op and fusion-pattern classes. Runs the test
# runner only on Python older than 3.14.
import sys
import torch
import unittest
import itertools
import torch.nn as nn
from functools import wraps
from concurrent import futures
import torch.nn.functional as F
import torch.fx.experimental.optimization as optimization
from torch.testing._internal.jit_utils import JitTestCase
from torch.testing._internal.common_utils import run_tests, TEST_SCIPY, IS_WINDOWS, IS_MACOS
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyCPU,
    dtypes
)

# We use this wrapper to run UTs of TorchVision models because of a memory-leak
# issue with JIT tracing that causes traced model objects to persist in the
# memory. Ref: https://github.com/pytorch/pytorch/issues/35600
# Memory requirement for running these UTs was thus increasing cumulatively, and
# invoked the Linux kernel OOM killer on linux.2xlarge PyTorch CI runners, which
# only have 16 GB RAM. Cumulatively, these UTs had been using more than 14 GB
# memory (as per psutils). So now we run each TorchVision model UTs in separate processes.
def separate_process(func):
    # [FUNCTION DOC] HELPER: Decorator that runs the wrapped test function in a
    # separate worker process via a process pool and waits for it to finish, to
    # avoid cumulative memory growth from traced models. The wrapper discards the
    # result and does not re-raise errors from the worker.
    # KNOWN ISSUE: the future's result is never retrieved, so exceptions raised in the worker
    # process are silently swallowed and the test passes.
    @wraps(func)
    def wrapper(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Submits the original function with the given arguments to a new
        # process pool, waits for completion, and returns nothing.
        with futures.ProcessPoolExecutor() as executor:
            future = executor.submit(func, *args, **kwargs)
            futures.wait([future])
    return wrapper

def is_avx512_supported():
    # [FUNCTION DOC] HELPER: Returns true only on Linux when the processor information file lists
    # avx512; returns false on other platforms. Reads that file as text.
    if sys.platform != 'linux':
        return False
    with open("/proc/cpuinfo", encoding="ascii") as f:
        lines = f.read()
    return "avx512" in lines

IS_AVX512_UNSUPPORTED = not is_avx512_supported()

LLGA_FUSION_GROUP = 'prim::oneDNNFusionGroup'
LLGA_NOT_ENABLED = not torch.backends.mkldnn.is_available() or IS_WINDOWS or IS_MACOS

def warmup_forward(f, *args, profiling_count=3):
    # [FUNCTION DOC] HELPER: Calls the function the given number of times (default three) with the
    # given arguments to let the profiling executor specialize it, and returns the result of the
    # last call.
    for _ in range(profiling_count):
        results = f(*args)

    return results

class JitLlgaTestCase(JitTestCase):
    # [CLASS DOC] Base test case for the oneDNN Graph fuser tests. Setup enables oneDNN fusion and
    # disables the JIT autocast mode; teardown restores them. Provides helpers to trace and check
    # a module, find fusion groups in a graph and assert on graph contents.

    def setUp(self):
        # [FUNCTION DOC] Records torch function mode stack length and state baselines, saves the
        # previous JIT autocast mode while turning it off, and enables oneDNN fusion. Deliberately
        # does not call the parent setup because its emit hooks cause crashes at process exit.
        # Don't call super().setUp() — JitTestCase.setUp installs JIT emit
        # hooks that cause segfaults during process cleanup. Record state
        # baselines that tearDown checks for.
        self._prev_torch_function_mode_stack_len = torch._C._len_torch_function_stack()
        self._prev_torch_function_state = torch._C._get_torch_function_state()
        # PyTorch has divergent op support for AMP in JIT & eager modes
        # so we disable AMP for JIT & leverage eager-mode AMP.
        # Ref: https://github.com/pytorch/pytorch/issues/75956
        self.original_autocast_mode = torch._C._jit_set_autocast_mode(False)
        torch.jit.enable_onednn_fusion(True)

    def tearDown(self):
        # [FUNCTION DOC] Disables oneDNN fusion and restores the saved JIT autocast mode.
        torch.jit.enable_onednn_fusion(False)
        torch._C._jit_set_autocast_mode(self.original_autocast_mode)

    def checkTrace(self, m, x, dtype=torch.float32, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Puts a module in eval mode, traces it (freezing it if it is a
        # module) with the example inputs, warms it up, runs the eager model for reference, and
        # gets the optimized graph. For bfloat16 the tracing runs under CPU autocast with caching
        # disabled. Asserts the traced output equals the eager output and returns the traced
        # object and its graph. Runs with gradients and emit hooks disabled.
        if isinstance(m, torch.nn.Module):
            m.eval()
        with torch.no_grad(), torch._jit_internal._disable_emit_hooks():
            if dtype == torch.bfloat16:
                # We rely upon eager-mode AMP support for BF16
                with torch.autocast(device_type="cpu", cache_enabled=False, dtype=torch.bfloat16):
                    traced = torch.jit.trace(m, x)
                    if isinstance(m, torch.nn.Module):
                        traced = torch.jit.freeze(traced)
                    warmup_forward(traced, *x)
                    ref_o = m(*x)
                    fwd_graph = traced.graph_for(*x)
            else:
                traced = torch.jit.trace(m, x)
                if isinstance(m, torch.nn.Module):
                    traced = torch.jit.freeze(traced)
                warmup_forward(traced, *x)
                ref_o = m(*x)
                fwd_graph = traced.graph_for(*x)

            jit_o = traced(*x)
            self.assertEqual(jit_o, ref_o)
            return traced, fwd_graph


    def assertFused(self, graph, fused_patterns):
        # [FUNCTION DOC] HELPER: Asserts that none of the given operator kind strings remain in
        # the graph (exact count zero), i.e. they were absorbed into fusion groups.
        for pat in fused_patterns:
            self.assertGraphContainsExactly(graph, pat, 0)

    def findFusionGroups(self, graph):
        # [FUNCTION DOC] HELPER: Recursively walks the graph nodes and nested blocks and returns a
        # list of the subgraphs of every oneDNN fusion group node found.
        result = []
        for n in graph.nodes():
            if n.kind() == LLGA_FUSION_GROUP:
                result.append(n.g('Subgraph'))
                continue
            for block in n.blocks():
                result += self.findFusionGroups(block)
        return result

    def checkPatterns(self, graph, patterns):
        # [FUNCTION DOC] HELPER: Collects the fusion groups of the graph, raises an assertion
        # error if their number differs from the number of pattern lists, then asserts each group
        # subgraph contains every operator in its corresponding pattern list.
        fusion_groups = self.findFusionGroups(graph)
        if len(fusion_groups) != len(patterns):
            raise AssertionError("length of subgraphs not equal to length of given patterns")

        for i in range(len(fusion_groups)):
            for pattern in patterns[i]:
                self.assertGraphContains(fusion_groups[i], pattern)

try:
    import torchvision
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False
except RuntimeError:
    HAS_TORCHVISION = False
skipIfNoTorchVision = unittest.skipIf(not HAS_TORCHVISION, 'no torchvision')

def get_eltwise_fn(name):
    # [FUNCTION DOC] HELPER: Returns an elementwise function by name, looking first in the torch
    # namespace, then the functional namespace, and returning an in-place hardswish module for the
    # name with trailing underscore. Raises a name error if none is found.
    if hasattr(torch, name):
        return getattr(torch, name)
    elif hasattr(F, name):
        return getattr(F, name)
    elif name == 'hardswish_':
        return torch.nn.Hardswish(inplace=True)
    else:
        raise NameError(f'Eltwise function {name} not found')


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestOp(JitLlgaTestCase):
    # [CLASS DOC] Tests of single operators under the oneDNN Graph fuser (convolution, pooling,
    # softmax, linear, binary ops and others), skipped without AVX512 or without a MKL-DNN build.
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d(self, dtype):
        # [FUNCTION DOC] Traces a 2D convolution over a grid of spatial size, channels, kernel,
        # padding, stride, dilation, groups and bias options, and asserts exactly one fusion group
        # each time.
        for [spatial, in_channels, out_channels, kernel, padding, stride, dilation, g, bias] in itertools.product(
                [7, 8],
                [8, 15],
                [7, 16],
                [3, 4],
                [0, 2],
                [1, 2],
                [1, 2],
                [1, 2],
                [True, False]):

            m = nn.Conv2d(in_channels=in_channels * g,
                          out_channels=out_channels * g,
                          kernel_size=kernel,
                          padding=padding,
                          stride=stride,
                          dilation=dilation,
                          groups=g,
                          bias=bias)

            x = torch.rand(1, in_channels * g, spatial, spatial)
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_bn2d(self, dtype):
        # [FUNCTION DOC] Traces an eval-mode 2D batch norm and asserts no fusion group is created
        # (single-op partitions are not formed).
        m = nn.BatchNorm2d(32).eval()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        # single-op partition shouldn't be created for softmax
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_eltwise(self, dtype):
        # [FUNCTION DOC] Checks that relu and gelu applied alone (wrapped in a module) produce no
        # fusion group.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module applying a given elementwise function to its input.
            def __init__(self, eltwise_fn):
                # [FUNCTION DOC] HELPER: Stores the elementwise function as an attribute.
                super().__init__()
                self.eltwise = eltwise_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the elementwise function applied to the input.
                return self.eltwise(x)

        for eltwise in ['relu', 'gelu']:
            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn)
            x = torch.rand(1, 32, 28, 28)
            _, graph = self.checkTrace(m, [x], dtype)
            # single-op partition shouldn't be created.
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_max_pool2d(self, dtype):
        # [FUNCTION DOC] Traces 2D max pooling over a grid of sizes, kernel, padding, stride (one
        # or two), dilation of one and ceil mode, asserting one fusion group each time.
        for [spatial, kernel, padding, stride, dilation, ceil_mode] in itertools.product(
                [15, 16, 17, 18, 19],
                [4, 5],
                [0, 1, 2],
                [1, 2],  # [1, 2, 4], TODO: fix issue in pad calculation
                [1],     # [1, 2], TODO: backend support for dilation
                [True, False]):

            m = nn.MaxPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             dilation=dilation,
                             ceil_mode=ceil_mode)

            x = torch.rand(1, 4, spatial, spatial)
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_avg_pool2d(self, dtype):
        # [FUNCTION DOC] Traces 2D average pooling over a grid of sizes, kernel,
        # padding, stride and count-include-pad, with ceil mode fixed to false,
        # asserting one fusion group each time.
        for [spatial, kernel, padding, stride, ceil_mode, count_include_pad] in itertools.product(
                [15, 16, 17, 18, 19],
                [4, 5],
                [0, 1, 2],
                [1, 2, 4],
                [False],  # TODO: oneDNN Graph does not fully support ceil_mode=True
                [True, False]):

            m = nn.AvgPool2d(kernel_size=kernel,
                             stride=stride,
                             padding=padding,
                             ceil_mode=ceil_mode,
                             count_include_pad=count_include_pad)

            x = torch.rand(1, 4, spatial, spatial)
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_variable_kernel_avg_pool2d(self, dtype):
        # [FUNCTION DOC] Traces average pooling whose kernel size comes from the input size and
        # asserts no fusion group, since the kernel is not constant.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module that average-pools its input with a kernel equal to its
            # full spatial size.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the average pool over the full spatial extent,
                # without padding and excluding padding from the count.
                x = F.avg_pool2d(x, kernel_size=(x.size(2), x.size(3)), padding=0, count_include_pad=False)
                return x

        x = torch.randn(1, 1000, 1, 1)
        m = M()
        _, graph = self.checkTrace(m, [x], dtype)
        # kernel_size is not Constant, shouldn't have any LLGA_FUSION_GROUP
        # TODO: with shape specialization, should have 1 LLGA_FUSION_GROUP
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_softmax(self, dtype):
        # [FUNCTION DOC] Traces softmax over each dimension of a 4D input and asserts no fusion
        # group is created.
        for dim in [-4, -3, -2, -1, 0, 1, 2, 3]:
            m = nn.Softmax(dim=dim)
            x = torch.rand(8, 12, 12, 12)
            _, graph = self.checkTrace(m, [x], dtype)
            # single-op partition shouldn't be created for softmax
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_linear(self, dtype):
        # [FUNCTION DOC] Traces a linear layer with and without bias, asserting one fusion group
        # and that the linear operator no longer appears outside it.
        for bias in [True, False]:
            x = torch.rand(32, 28)
            m = torch.nn.Linear(in_features=28, out_features=64, bias=bias)
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::linear'])


    def _gen_binary_inputs(self, gen_permute=True):
        # [FUNCTION DOC] HELPER: Generator yielding pairs of random tensors for binary op tests: a
        # 4D tensor with itself shape, with a broadcast shape, with a vector and with a single
        # element, and optionally the swapped order for differing shapes.
        for xshape, yshape in [
            [[1, 32, 28, 28], [1, 32, 28, 28]],
            [[1, 32, 28, 28], [1, 1, 28, 28]],
            [[1, 32, 28, 28], [28]],
            [[1, 32, 28, 28], [1]],

        ]:
            yield torch.rand(xshape), torch.rand(yshape)
            if gen_permute and xshape != yshape:
                yield torch.rand(yshape), torch.rand(xshape)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_add(self, dtype):
        # [FUNCTION DOC] Traces an add with an alpha factor over the binary inputs and asserts one
        # fusion group each time.
        def forward_add(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the two inputs with an
            # alpha of two on the second.
            return torch.add(x, y, alpha=2)

        for x, y in self._gen_binary_inputs():
            _, graph = self.checkTrace(forward_add, [x, y], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_add_scalar(self, dtype):
        # [FUNCTION DOC] Traces adding two scalars to a tensor and asserts one fusion group.
        def add_scalar(x):
            # [FUNCTION DOC] HELPER: Returns the input plus an integer plus a float constant.
            return 42 + x + 3.14

        x = torch.rand(32, 32)
        _, graph = self.checkTrace(add_scalar, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_addmm(self, dtype):
        # [FUNCTION DOC] Traces a matrix multiply with bias add and asserts one fusion group (a
        # single-op partition is allowed for matmul with bias).
        # Just a sidenote - comparison of eager-mode & oneDNN Graph JIT outputs of
        # addmm (which entails matmul-bias-add fusion) might require higher tolerance
        # bounds for BF16. This is subject to change in the near future.
        def addmm(x, y, z):
            # [FUNCTION DOC] HELPER: Returns the fused matrix multiply of two inputs added to a
            # third input.
            # alpha and beta are 1, by default
            return torch.addmm(z, x, y)

        x = torch.rand(64, 32)
        y = torch.rand(32, 32)
        z = torch.rand(64, 32)
        _, graph = self.checkTrace(addmm, [x, y, z], dtype)
        # single-op partition should be created for matmul with bias.
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_mul(self, dtype):
        # [FUNCTION DOC] Traces multiplication followed by a scalar multiply over the binary
        # inputs and asserts one fusion group each time.
        def forward_mul(x, y):
            # [FUNCTION DOC] HELPER: Returns the product of two inputs times three.
            return torch.mul(x, y) * 3

        for x, y in self._gen_binary_inputs():
            _, graph = self.checkTrace(forward_mul, [x, y], dtype)
            # single-op partitions shouldn't be created
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_identity_binary(self, dtype):
        # [FUNCTION DOC] Traces an expression multiplying by one and adding zero and asserts no
        # mul or add operator remains outside fusion groups.
        def forward(x):
            # [FUNCTION DOC] HELPER: Returns the input times one plus zero.
            return x * 1 + 0.0

        x = torch.rand(32)
        _, graph = self.checkTrace(forward, [x], dtype)
        self.assertFused(graph, ['aten::add', 'aten::mul'])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_layer_norm(self, dtype):
        # [FUNCTION DOC] Traces a layer norm over the last dimension of ten and
        # asserts one fusion group.
        # TODO: support more normalized_shape
        m = torch.nn.LayerNorm(10)
        x = torch.randn(2, 5, 10, 10)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_cat(self, dtype):
        # [FUNCTION DOC] Traces concatenation of three copies of a tensor along each dimension,
        # for several shapes, asserting one fusion group.
        def cat_along_dim(d):
            # [FUNCTION DOC] HELPER: Returns a function that concatenates its arguments along the
            # given dimension.
            def forward_cat(*inputs):
                # [FUNCTION DOC] HELPER: Concatenates all positional inputs along
                # the captured dimension.
                return torch.cat(inputs, d)
            return forward_cat

        for xshape in [
            [8, 8, 8, 8],
            [64, 8, 32],
            [2048, 64],
        ]:
            for d in range(len(xshape)):
                x = torch.rand(xshape)
                _, graph = self.checkTrace(cat_along_dim(d), [x, x, x], dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_typecheck(self, dtype):
        # [FUNCTION DOC] Traces a linear layer created in the test dtype with one fusion group,
        # then runs the traced module on a smaller batch and checks the result still equals eager,
        # exercising the fallback path on shape change.
        x = torch.rand(32, 28, dtype=dtype)
        m = torch.nn.Linear(in_features=28, out_features=64, bias=True, dtype=dtype)
        traced, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::linear'])
        # change the shape of the input, we should enter fallback graph
        x = torch.rand(5, 28, dtype=dtype)
        self.assertEqual(m(x), traced(x))


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestFusionPattern(JitLlgaTestCase):
    # [CLASS DOC] Tests of fusion patterns (convolution with elementwise ops, batch norm, linear
    # with elementwise ops, sums, wildcards) under the oneDNN Graph fuser, skipped without AVX512
    # or MKL-DNN.
    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_eltwise(self, dtype):
        # [FUNCTION DOC] For several elementwise functions in in-place and out-of-place form,
        # traces two convolutions with the function between and after, asserts two fusion groups
        # and that neither the in-place nor out-of-place op remains outside.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with two convolutions applying a given elementwise
            # function after each.
            def __init__(self, eltwise_fn):
                # [FUNCTION DOC] HELPER: Creates two 3x3 convolutions with 32 channels (the first
                # with bias, the second without) and stores the elementwise function.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=False)
                self.eltwise = eltwise_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs convolution, elementwise, convolution, elementwise
                # on the input.
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                return x

        for eltwise in ['relu', 'leaky_relu', 'sigmoid', 'square',
                        'abs', 'exp', 'hardswish', 'tanh', 'hardtanh']:
            for inplace in [True, False]:
                eltwise_fn_name = eltwise + '_' if inplace else eltwise
                eltwise_fn = get_eltwise_fn(eltwise_fn_name)

                m = M(eltwise_fn)
                x = torch.rand(1, 32, 28, 28)
                _, graph = self.checkTrace(m, [x], dtype=dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # test if relu_ is replace with relu by mutation removal pass
                self.assertFused(graph, ['aten::' + eltwise_fn_name])
                # test if relu is fused into the fusion group
                self.assertFused(graph, ['aten::' + eltwise])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_silu(self, dtype):
        # [FUNCTION DOC] For in-place and out-of-place silu and both contiguous and channels-last
        # memory formats, traces two convolutions with silu between them, asserts two fusion
        # groups, that convolution and silu are fused, and that the first group contains
        # convolution, sigmoid and multiply while the second contains only convolution.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with two convolutions and a silu between them, in-place
            # if requested.
            def __init__(self, inplace):
                # [FUNCTION DOC] HELPER: Creates two biased 32 channel convolutions and the silu
                # module with the in-place option.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = nn.SiLU(inplace=inplace)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the first convolution, silu and
                # the second convolution.
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                return x
        for inplace in [False, True]:
            for memory_format in [torch.contiguous_format, torch.channels_last]:
                m = M(inplace)
                x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)

                _, graph = self.checkTrace(m, [x], dtype)
                self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 2)
                # oneDNN graph does not have silu OP. The bridge will convert silu to sigmoid - mul
                # Inplace op will become outplace op on the JIT graph
                patterns = [
                    ["aten::_convolution", 'aten::sigmoid', 'aten::mul'],
                    ["aten::_convolution"]
                ]
                silu_op = 'aten::silu_' if inplace else 'aten::silu'
                self.assertFused(graph, ['aten::_convolution', silu_op])
                self.checkPatterns(graph, patterns)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_ensure_tensor_is_rewrapped(self, dtype):
        # [FUNCTION DOC] Traces a two-branch model of convolutions and relus, adds the branches
        # and average-pools adaptively (an op the backend does not support), in channels-last
        # format, and asserts the result matches eager and that there are four fusion groups.
        # Guards against an accuracy bug when an output tensor is consumed by an unsupported op.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with two parallel conv-elementwise-conv-elementwise
            # branches summed and followed by adaptive average pooling.
            def __init__(self, eltwise_fn):
                # [FUNCTION DOC] HELPER: Creates four biased 32 channel convolutions, stores the
                # elementwise function and an adaptive average pooling to a 5 by 7 output.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = eltwise_fn
                self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Runs each of two inputs through its own
                # convolution pair with the elementwise function, adds the results
                # and applies the adaptive pooling.
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                y = self.conv3(y)
                y = self.eltwise(y)
                y = self.conv4(y)
                y = self.eltwise(y)

                x = torch.add(x, y)
                x = self.adaptive_avg_pool_2d(x)
                return x

        eltwise_fn_name = 'relu'
        eltwise_fn = get_eltwise_fn(eltwise_fn_name)
        m = M(eltwise_fn)
        m = m.to(memory_format=torch.channels_last)
        x = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        y = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        # Simply test if the output is accurate
        # The output of the second partition is input to adaptive_avg_pool2d, which is
        # unsupported by LLGA. In resnext101 32x16d, we encountered an accuracy issue.
        _, graph = self.checkTrace(m, [x, y], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 4)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_clamp(self, dtype):
        # [FUNCTION DOC] Traces five stacked convolutions each followed by a clamp with different
        # bound combinations (infinite, one-sided, two-sided), in contiguous and channels-last
        # formats, asserting five fusion groups with convolution and clamp both fused.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module of five convolutions each followed by a clamp with
            # different bounds.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates five biased 32 channel convolutions.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv4 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv5 = nn.Conv2d(32, 32, 3, padding=1, bias=True)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies each convolution then a clamp with min only
                # (negative infinity), min only (minus five), min zero and infinite max, min one
                # and max five, and max two only.
                x = self.conv1(x)
                x = torch.clamp(x, min=float('-inf'))
                x = self.conv2(x)
                x = torch.clamp(x, min=-5)
                x = self.conv3(x)
                x = torch.clamp(x, min=0, max=float('inf'))
                x = self.conv4(x)
                x = torch.clamp(x, min=1, max=5)
                x = self.conv5(x)
                x = torch.clamp(x, max=2)
                return x

        for memory_format in [torch.contiguous_format, torch.channels_last]:
            x = torch.rand(1, 32, 28, 28).to(memory_format=memory_format)
            m = M()
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 5)
            self.assertFused(graph, ['aten::_convolution', "aten::clamp"])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_bn(self, dtype):
        # [FUNCTION DOC] Traces a convolution followed by batch norm in eval mode (pre-fused with
        # the FX optimization for bfloat16) and asserts one fusion group containing convolution
        # and batch norm.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution followed by batch norm.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one biased convolution and a batch norm layer of
                # 32 channels.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies convolution then batch norm.
                x = self.conv1(x)
                x = self.bn1(x)
                return x

        m = M().eval()
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm'])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_bn_relu(self, dtype):
        # [FUNCTION DOC] Same as the convolution batch norm test with a relu after, asserting one
        # fusion group with convolution, batch norm and relu fused.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with a convolution, batch norm and relu.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates one biased convolution and a batch norm layer of
                # 32 channels.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.bn1 = nn.BatchNorm2d(32)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies convolution, batch norm and relu.
                x = self.conv1(x)
                x = self.bn1(x)
                x = F.relu(x)
                return x

        m = M().eval()
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu'])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_bn2d_eltwise(self, dtype):
        # [FUNCTION DOC] Traces batch norm followed by relu in eval mode and asserts one fusion
        # group with relu fused.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module applying batch norm then a given elementwise function.
            def __init__(self, eltwise_fn):
                # [FUNCTION DOC] HELPER: Stores the elementwise function and creates a 32 channel
                # batch norm.
                super().__init__()
                self.eltwise = eltwise_fn
                self.bn = nn.BatchNorm2d(32)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies batch norm and then the elementwise function.
                x = self.bn(x)
                x = self.eltwise(x)
                return x

        for eltwise in ['relu']:
            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn).eval()
            x = torch.rand(1, 32, 28, 28)
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::' + eltwise])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_linear_eltwise(self, dtype):
        # [FUNCTION DOC] Traces a linear layer with and without bias followed by each of several
        # elementwise functions, asserting one fusion group with the function fused.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with a linear layer from 28 to 64 features followed by an
            # elementwise function.
            def __init__(self, eltwise_fn, bias):
                # [FUNCTION DOC] HELPER: Creates the linear layer with the given bias flag and
                # stores the elementwise function.
                super().__init__()
                self.linear = nn.Linear(28, 64, bias)
                self.eltwise = eltwise_fn

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear layer and the elementwise function.
                x = self.linear(x)
                x = self.eltwise(x)
                return x

        for [has_bias, eltwise] in itertools.product(
                [True, False],
                ['relu', 'gelu', 'sigmoid', 'hardtanh', 'relu6', 'elu']):

            eltwise_fn = get_eltwise_fn(eltwise)
            m = M(eltwise_fn, has_bias)
            x = torch.rand(32, 28, requires_grad=False)
            _, graph = self.checkTrace(m, [x], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
            self.assertFused(graph, ['aten::' + eltwise])

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_conv2d_sum(self, dtype):
        # [FUNCTION DOC] Traces a residual-like model where two convolution plus batch norm
        # branches are added, passed through relu and then another convolution plus batch norm,
        # with and without convolution bias, asserting three fusion groups.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module with two convolution plus batch norm branches whose sum
            # goes through relu and a third convolution plus batch norm.
            def __init__(self, bias=False):
                # [FUNCTION DOC] HELPER: Creates three 32 channel convolutions with the given bias
                # flag, each paired with a batch norm, and a relu.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn1 = nn.BatchNorm2d(32)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn2 = nn.BatchNorm2d(32)
                self.relu = nn.ReLU()
                self.conv3 = nn.Conv2d(32, 32, 3, padding=1, bias=bias)
                self.bn3 = nn.BatchNorm2d(32)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the first two branches, adds and relus them, then
                # applies the third convolution and batch norm.
                x = self.conv1(x)
                x = self.bn1(x)
                y = self.conv2(y)
                y = self.bn2(y)
                z = self.relu(x + y)
                z = self.conv3(z)
                z = self.bn3(z)
                return z

        for bias in [True, False]:
            m = M(bias).eval()
            if dtype == torch.bfloat16:
                m = optimization.fuse(m)
            x = torch.rand(1, 32, 16, 16, requires_grad=False)
            y = torch.rand(1, 32, 16, 16, requires_grad=False)
            _, graph = self.checkTrace(m, [x, y], dtype)
            self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 3)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_wildcard(self, dtype):
        # [FUNCTION DOC] Traces a convolution whose output is used both by relu and by a list
        # construction (a wildcard op) and asserts one fusion group containing only the
        # convolution, since relu cannot be fused with it.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module returning both a convolution output and
            # its relu in a list.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a biased convolution and a relu.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = nn.ReLU()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns a list with the convolution output and the relu
                # of it.
                x = self.conv1(x)
                y = self.eltwise(x)
                return [x, y]

        # The pattern is as the following:
        #      conv
        #     |    \
        # eltwise   \
        #    |       \
        #  ListConstruct
        #
        # The output of conv is used by a wildcard op: ListConstruct.
        # Thus conv-eltwise cannot be selected into the same Partition.
        m = M()
        x = torch.rand(1, 32, 28, 28)
        _, graph = self.checkTrace(m, [x], dtype)
        # conv can exist in a single-op oneDNN Graph partition but not relu
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 1)
        self.assertFused(graph, ['aten::_convolution'])

    @onlyCPU
    @dtypes(torch.int32)
    def test_wildcard_unsupported_dtype(self, dtype):
        # [FUNCTION DOC] Traces floor division of an integer tensor by two, which involves an
        # integer type the backend does not support, to check the bridge handles unsupported input
        # dtypes, and asserts no fusion group.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module returning the floor division of its input by two.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: Returns the input floor divided by two.
                y = x // 2
                return y

        # In shufflenet_v2_x1_0, channels_per_groups is computed as:
        # channels_per_group = num_channels // groups
        # JIT IR converts groups to Long dtype, which is unsupported
        # by oneDNN Graph, viz. Long(requires_grad=0, device=cpu) = prim::Constant[value={2}]()
        # This test just ensures that the bridge code can handle
        # unsupported dtypes for inputs to ops unsupported
        # by oneDNN Graph. In this particular UT, aten::floor_divide
        # would be added as a wildcard in graph-construction stage.
        m = M()
        x = torch.tensor([32], dtype=dtype)
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertGraphContainsExactly(graph, LLGA_FUSION_GROUP, 0)

    @onlyCPU
    @dtypes(torch.float32, torch.bfloat16)
    def test_rewrap_tensor_input_to_pytorch(self, dtype):
        # [FUNCTION DOC] Traces a model of two convolution and relu stages, adds a second input
        # and applies adaptive average pooling (unsupported by the backend), in channels-last
        # format, only checking that the result equals eager, so strides are passed correctly back
        # to PyTorch.
        class M(nn.Module):
            # [CLASS DOC] HELPER: Module of two convolution plus elementwise stages, an addition
            # with a second input and adaptive average pooling.
            def __init__(self, eltwise_fn):
                # [FUNCTION DOC] HELPER: Creates two biased 32 channel convolutions, stores the
                # elementwise function and an adaptive pooling to a 5 by 7 output.
                super().__init__()
                self.conv1 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.conv2 = nn.Conv2d(32, 32, 3, padding=1, bias=True)
                self.eltwise = eltwise_fn
                self.adaptive_avg_pool_2d = nn.AdaptiveAvgPool2d((5, 7))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Applies the two stages, adds the second input and applies
                # the adaptive pooling.
                x = self.conv1(x)
                x = self.eltwise(x)
                x = self.conv2(x)
                x = self.eltwise(x)
                x = torch.add(x, y)
                x = self.adaptive_avg_pool_2d(x)
                return x

        eltwise_fn_name = 'relu'
        eltwise_fn = get_eltwise_fn(eltwise_fn_name)
        m = M(eltwise_fn)
        m = m.to(memory_format=torch.channels_last)
        x = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        y = torch.rand(1, 32, 28, 28).to(memory_format=torch.channels_last)
        # Simply test if the output is accurate
        # The output of the second partition is input to adaptive_avg_pool2d, which is
        # unsupported by LLGA, so it must be handled by PyTorch, which should receive
        # correct strides info of the channels-last tensor.
        self.checkTrace(m, [x, y], dtype)

@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestEnableDisableLlgaFuser(JitTestCase):
    # [CLASS DOC] Tests turning the oneDNN Graph fuser on and off through the fuser context
    # manager. Setup disables the fuser and remembers the prior state; teardown restores it.
    def setUp(self):
        # [FUNCTION DOC] HELPER: Calls the parent setup and then disables the LLGA fuser, saving
        # its previous enabled flag.
        super().setUp()
        self.is_enabled = torch._C._jit_set_llga_enabled(False)

    def tearDown(self):
        # [FUNCTION DOC] HELPER: Restores the saved enabled flag of the LLGA fuser and then calls
        # the parent teardown.
        torch._C._jit_set_llga_enabled(self.is_enabled)
        super().tearDown()

    def test_context_manager(self):
        # [FUNCTION DOC] Scripts three small add functions: two under nested uses of the third
        # fuser option, which must yield fusion group nodes after two calls, and one after leaving
        # the context, which must have no fusion group.
        x = torch.randn(4, 8)
        y = torch.randn(4, 8)
        with torch.jit.fuser('fuser3'):
            with torch.jit.fuser('fuser3'):

                def t1(x, y):
                    # [FUNCTION DOC] HELPER: Returns the sum of the inputs plus two.
                    o = x + y
                    o = o + 2.0
                    return o
                t_jit = torch.jit.script(t1)
                t_jit(x, y)
                t_jit(x, y)
                self.assertGraphContains(t_jit.graph_for(x, y), LLGA_FUSION_GROUP)

            def t2(x, y):
                # [FUNCTION DOC] HELPER: Returns the sum of the inputs plus three.
                o = x + y
                o = o + 3.0
                return o
            t_jit_2 = torch.jit.script(t2)
            t_jit_2(x, y)
            t_jit_2(x, y)
            self.assertGraphContains(t_jit_2.graph_for(x, y), LLGA_FUSION_GROUP)

        def t3(x, y):
            # [FUNCTION DOC] HELPER: Returns the sum of the inputs plus four.
            o = x + y
            o = o + 4.0
            return o
        t_jit_3 = torch.jit.script(t3)
        t_jit_3(x, y)
        t_jit_3(x, y)
        self.assertGraphContainsExactly(t_jit_3.graph_for(x, y), LLGA_FUSION_GROUP, 0)


@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
@unittest.skip("Enable when integration with dynamo aot_autograd is more stable")
class TestDynamoAOT(JitTestCase):
    # [CLASS DOC] Test case for ahead-of-time autograd compilation with a TorchScript backend and
    # oneDNN fusion; the class is unconditionally skipped.
    def test_dynamo_aot_ts_onednn(self):
        # [FUNCTION DOC] Compiles a small two layer network with the AOT TorchScript backend
        # requiring a full graph, runs forward and backward ten times under the third fuser
        # option, then resets dynamo state. Currently skipped.
        class Seq(nn.Module):
            # [CLASS DOC] HELPER: Module wrapping a sequence of two linear layers with relus.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Builds a sequential container of linear, relu, linear and
                # relu layers of width ten.
                super().__init__()
                self.layers = nn.Sequential(
                    nn.Linear(10, 10),
                    nn.ReLU(),
                    nn.Linear(10, 10),
                    nn.ReLU(),
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the sequential layers on the input.
                return self.layers(x)

        mod = Seq()

        import torch._dynamo
        aot_mod = torch.compile(mod, backend="aot_ts", fullgraph=True)

        for _ in range(10):
            with torch.jit.fuser("fuser3"):
                loss = aot_mod(torch.rand([10, 10])).sum()
                loss.backward()

        torch._dynamo.reset()


@unittest.skipIf(IS_AVX512_UNSUPPORTED, "This test fails for BF16 on machines without AVX512.")
@unittest.skipIf(LLGA_NOT_ENABLED, "MKL-DNN build is disabled")
class TestModel(JitLlgaTestCase):
    # [CLASS DOC] Whole-model tests on TorchVision networks. Test methods are
    # attached dynamically after the class definition, one per model name and
    # dtype, each running in a separate process.
    @skipIfNoTorchVision
    def _test_vision(self, model_name, dtype):
        # [FUNCTION DOC] HELPER: Builds the named TorchVision model in eval mode (batch-norm-fused
        # for bfloat16), traces it with a small random image batch, and asserts the listed
        # operator kinds (convolution, batch norm, relu, linear, pooling) are no longer present
        # outside fusion groups. Skipped without TorchVision; it is a helper, not a test itself,
        # since its name has no test prefix.
        m = getattr(torchvision.models, model_name)().eval()
        if dtype == torch.bfloat16:
            m = optimization.fuse(m)
        x = torch.rand(1, 3, 224, 224) / 10
        _, graph = self.checkTrace(m, [x], dtype)
        self.assertFused(graph, ['aten::_convolution', 'aten::batch_norm',
                                 'aten::relu', 'aten::linear',
                                 'aten::avg_pool2d', 'aten::max_pool2d'])

for model_name, enabled in [
    ['resnet50', True],
    ['resnext50_32x4d', True],
    ['resnext101_32x8d', True],
    ['densenet121', True],
    ['densenet161', True],
    ['densenet169', True],
    ['densenet201', True],
    ['efficientnet_b0', True],
    ['efficientnet_b1', True],
    ['efficientnet_b2', True],
    ['efficientnet_b3', True],
    ['efficientnet_b4', True],
    ['efficientnet_b5', True],
    ['efficientnet_b6', True],
    ['efficientnet_b7', True],
    ['regnet_y_400mf', True],
    ['googlenet', TEST_SCIPY],
    ['mobilenet_v2', True],
    ['mobilenet_v3_large', True],
    ['mnasnet1_0', True],
    ['squeezenet1_0', True],
    ['vgg16', True],
    ['alexnet', True],
    ['shufflenet_v2_x1_0', True],
    ['wide_resnet50_2', True],
]:
    def _wrapper(mname, dtype):
        # [FUNCTION DOC] HELPER: Factory building a test method for a model name and dtype,
        # closing over the enabled flag of the loop variable.
        @unittest.skipIf(not enabled, 'Disabled')
        @separate_process
        def test(self, dtype=dtype):
            # [FUNCTION DOC] HELPER: Test method that runs the vision check for the captured model
            # in a separate process; skipped when the model is disabled. Accepts the dtype as a
            # default argument.
            return self._test_vision(mname, dtype)
        return test

    for dtype in [torch.bfloat16, torch.float32]:
        setattr(TestModel, 'test_vision_{}_{}'.format(model_name, str(dtype).split("torch.")[1]), _wrapper(model_name, dtype))


instantiate_device_type_tests(TestFusionPattern, globals())
instantiate_device_type_tests(TestOp, globals())

if __name__ == '__main__':
    if sys.version_info < (3, 14):
        run_tests()
