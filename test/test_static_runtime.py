# Owner(s): ["module: unknown"]
# ruff: noqa: F841
# [MODULE DOC] Tests for the Static Runtime (StaticModule) executor on TorchScript modules and
# functions. Covers fork/wait (sync and runAsync), exception propagation from forked subgraphs,
# attention and DLRM-style MLP modules, SetAttr/GetAttr handling, benchmark APIs, and (skipped)
# subgraph fusion. Import-time side effect: replaces the global linear function in
# torch.nn.functional with a matmul-based shim so scripted modules lower to plain matmul.

import unittest

import numpy as np
import torch
from torch import nn
from torch.testing._internal.common_utils import TestCase, run_tests
from torch.testing._internal.static_module import StaticModule


def linear_shim(
    input: torch.Tensor, weight: torch.Tensor, bias: torch.Tensor | None = None
) -> torch.Tensor:
    # [FUNCTION DOC] HELPER: Replacement for the functional linear op. Computes input times the
    # transposed weight, adds the bias in place when one is given, and returns the result.
    output = input.matmul(weight.t())
    if bias is not None:
        output += bias
    ret = output
    return ret


torch.nn.functional.linear = linear_shim


class MultiHeadAttentionLayer(nn.Module):
    # [CLASS DOC] HELPER: Multi-head scaled dot-product attention module used as a scripting and
    # Static Runtime test subject. Holds four linear projections (q, k, v, output) and a fixed
    # scale tensor equal to the square root of the per-head dimension. Dropout and masking are
    # disabled (commented out), so the mask argument is accepted but unused.
    def __init__(self, hid_dim, n_heads, dropout, device):
        # [FUNCTION DOC] HELPER: Stores sizes, builds the four projection layers and the scale
        # tensor on the given device. Raises AssertionError if hid_dim is not divisible by
        # n_heads. The dropout argument is accepted but ignored.
        super().__init__()
        if hid_dim % n_heads != 0:
            raise AssertionError(f"hid_dim ({hid_dim}) must be divisible by n_heads ({n_heads})")
        self.hid_dim = hid_dim
        self.n_heads = n_heads
        self.head_dim = hid_dim // n_heads
        self.fc_q = nn.Linear(hid_dim, hid_dim)
        self.fc_k = nn.Linear(hid_dim, hid_dim)
        self.fc_v = nn.Linear(hid_dim, hid_dim)
        self.fc_o = nn.Linear(hid_dim, hid_dim)
        # self.dropout = nn.Dropout(dropout)
        self.scale = torch.sqrt(torch.FloatTensor([self.head_dim])).to(device)

    def forward(self, query, key, value, mask):
        # [FUNCTION DOC] HELPER: Projects query, key and value, splits them into heads, computes
        # softmax-normalized scaled dot-product attention, merges heads and applies the output
        # projection. Returns a tuple of the output tensor and the attention weights. The mask
        # is unused.
        batch_size = query.shape[0]
        Q = self.fc_q(query)
        K = self.fc_k(key)
        V = self.fc_v(value)
        Q = Q.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        K = K.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        V = V.view(batch_size, -1, self.n_heads, self.head_dim).permute(0, 2, 1, 3)
        energy = torch.matmul(Q, K.permute(0, 1, 3, 2)) / self.scale
        # energy = energy.masked_fill(mask == 0, -1e10)
        attention = torch.softmax(energy, dim=-1)
        # x = torch.matmul(self.dropout(attention), V)
        x = torch.matmul(attention, V)
        x = x.permute(0, 2, 1, 3).contiguous()
        x = x.view(batch_size, -1, self.hid_dim)
        x = self.fc_o(x)
        return x, attention


# Taken from https://github.com/facebookresearch/dlrm/blob/master/dlrm_s_pytorch.py
def create_mlp(ln, sigmoid_layer):
    # [FUNCTION DOC] HELPER: Builds a scripted, eval-mode MLP following the DLRM reference code.
    # For each consecutive pair of sizes in the list ln it creates a linear layer with weights
    # drawn from a normal distribution (unseeded numpy RNG) and then appends a sigmoid if the
    # layer index equals sigmoid_layer, otherwise a relu. Scripts the sequential container under
    # no-grad and returns it.
    layers = nn.ModuleList()
    for i in range(len(ln) - 1):
        n = ln[i]
        m = ln[i + 1]

        LL = nn.Linear(int(n), int(m), bias=True)

        mean = 0.0  # std_dev = np.sqrt(variance)
        std_dev = np.sqrt(2 / (m + n))  # np.sqrt(1 / m) # np.sqrt(1 / n)
        W = np.random.normal(mean, std_dev, size=(m, n)).astype(np.float32)
        std_dev = np.sqrt(1 / m)  # np.sqrt(2 / (m + 1))
        bt = np.random.normal(mean, std_dev, size=m).astype(np.float32)
        LL.weight.data = torch.tensor(W, requires_grad=True)
        LL.bias.data = torch.tensor(bt, requires_grad=True)
        layers.append(LL)

        if i == sigmoid_layer:
            layers.append(nn.Sigmoid())
        else:
            layers.append(nn.ReLU())

    with torch.no_grad():
        s = torch.jit.script(torch.nn.Sequential(*layers))
    s.eval()
    return s


def trivial_graph(a, b, c):
    # [FUNCTION DOC] HELPER: Scripted test graph that returns a plus b times c plus a constant
    # 2x2 tensor of threes.
    s = torch.tensor([[3, 3], [3, 3]])
    return a + b * c + s

def elementwise_square_addition(input1, input2):
    # [FUNCTION DOC] HELPER: Returns the sum of the elementwise squares of the two inputs. Used
    # as the forked callee in the first fork/wait graph.
    return input1 * input1 + input2 * input2

def fork_wait_graph1(input1, input2):
    # [FUNCTION DOC] HELPER: Forks the squared-sum function on both inputs and waits on the
    # future, returning its result.
    fut = torch.jit.fork(elementwise_square_addition, input1, input2)
    return torch.jit.wait(fut)

def fork_wait_graph2(input1, input2):
    # [FUNCTION DOC] HELPER: Forks the loop graph with a fixed iteration count of 5 and returns
    # the waited result.
    fut = torch.jit.fork(loop_graph, input1, input2, 5)
    return torch.jit.wait(fut)

"""
   graph with multiple fork/wait operations
   :param input: torch.tensor input to forked subgraph
   :param iters: number of future/wait pairs to be created
"""
def fork_wait_graph3(input, iters: int):
    # [FUNCTION DOC] HELPER: Creates iters forks that each negate the input, waits on all
    # futures in order, stacks the results and returns their total sum as a tensor.
    futures : list[torch.jit.Future[torch.Tensor]] = []
    for _ in range(iters):
        futures.append(torch.jit.fork(torch.neg, input))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.sum(torch.stack(results))

"""
   graph with multi-level fork/wait operations
   :param input: torch.tensor input to forked subgraph
   :param num_forks: number of top level forks
   :param num_child_forks: number of child forks per parent fork
"""
def fork_wait_graph4(input, num_forks: int, num_child_forks: int):
    # [FUNCTION DOC] HELPER: Multi-level fork graph. Forks the multi-fork graph num_forks times,
    # each with num_child_forks child forks, waits on all of them and returns the sum of the
    # stacked results.
    futures : list[torch.jit.Future[torch.Tensor]] = []
    for _ in range(num_forks):
        futures.append(torch.jit.fork(fork_wait_graph3, input, num_child_forks))
    results = []
    for future in futures:
        results.append(torch.jit.wait(future))
    return torch.sum(torch.stack(results))

def add_tensor(input1, input2):
    # [FUNCTION DOC] HELPER: Returns the elementwise sum of the two inputs. Raises a shape error
    # when the sizes are not broadcastable.
    return input1 + input2

def fork_wait_graph_exception(input1, input2):
    # [FUNCTION DOC] HELPER: Forks the add function on the inputs and waits on the future. With
    # incompatible shapes the error from the forked call surfaces at the wait.
    fut = torch.jit.fork(add_tensor, input1, input2)
    return torch.jit.wait(fut)

def loop_graph(a, b, iters: int):
    # [FUNCTION DOC] HELPER: Loop test graph. Starts with a plus twice b, then iters times adds
    # b, doubles in place, and subtracts a in place. Returns the final tensor.
    c = a + b * 2
    for _ in range(iters):
        c = c + b
        c *= 2
        c -= a
    return c


def output_graph(a, b, c, iters: int):
    # [FUNCTION DOC] HELPER: Computes the sum of a plus b times c plus a constant 2x2
    # tensor, then returns a mapping from each integer in the range of iters to that sum
    # plus the integer.
    s = torch.tensor([[3, 3], [3, 3]])
    k = a + b * c + s
    d: dict[int, torch.Tensor] = {}
    for i in range(iters):
        d[i] = k + i
    return d


class SubModule(nn.Module):
    # [CLASS DOC] HELPER: Small module with constant integer attributes a=11 and b=2 whose
    # forward adds them to the input; used as a submodule in the attribute test.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the integer attributes a to 11 and b to 2.
        super().__init__()
        self.a = 11
        self.b = 2

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the input tensor plus the sum of the two attributes.
        return self.a + self.b + x


class SubModule2(nn.Module):
    # [CLASS DOC] HELPER: Like the first submodule (a is 12) but forward first overwrites
    # attribute b with 30, exercising attribute mutation in scripted code.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Sets the integer attributes a to 12 and b to 2.
        super().__init__()
        self.a = 12
        self.b = 2

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Sets attribute b to 30 (mutating module state) and returns the
        # input plus a plus b.
        self.b = 30
        return self.a + self.b + x


class TestModule(nn.Module):
    # [CLASS DOC] HELPER: Top-level module for the attribute test. Owns the two submodules and
    # integer attributes a=3 and b=4; forward mutates b and combines both submodule outputs.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Creates the two submodules and sets attributes a to 3 and
        # b to 4.
        super().__init__()
        self.sub1 = SubModule()
        self.sub2 = SubModule2()
        self.a = 3
        self.b = 4

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Sets b to 20, then returns the first submodule output plus a
        # plus b plus the second submodule output for the input.
        self.b = 20
        return self.sub1(x) + self.a + self.b + self.sub2(x)


class TestStaticModule(TestCase):
    # [CLASS DOC] Test case checking that StaticModule results match the eager TorchScript
    # results for fork/wait graphs, attention and MLP modules, attribute access, and
    # object creation.

    """
    Test Case: To test simple fork/wait operation in a graph
    fork is called on simple addition operation on input tensors
    """
    def test_fork_wait_1(self):
        # [FUNCTION DOC] Scripts the first fork/wait graph, runs it with ones and random
        # tensors, and checks the StaticModule output is close to the scripted reference.
        inp1 = torch.ones(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph1)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module(inp1, inp2)
        torch.testing.assert_close(output_test, output_ref)

    """
    Test Case: To test simple fork/wait operation with
    StaticRuntime runAsync API returning future
    """
    def test_fork_wait_1_async(self):
        # [FUNCTION DOC] Same as the synchronous fork/wait test but calls runAsync on the
        # StaticModule with a tuple of inputs and an empty kwargs mapping, waits on the returned
        # future and compares its value with the reference.
        inp1 = torch.ones(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph1)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module.runAsync((inp1, inp2), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    Test Case: To test fork/wait operation in a graph on
    a loop subgraph performing mix of operations
    """
    def test_fork_wait_2(self):
        # [FUNCTION DOC] Compares StaticModule and scripted outputs for a graph that forks a
        # loop subgraph, using two random 5x5 inputs.
        inp1 = torch.randn(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph2)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module(inp1, inp2)
        torch.testing.assert_close(output_test, output_ref)

    """
    Test Case: To test fork/wait operation on a loop
    subgraph with StaticRuntime runAsync API returning future
    """
    def test_fork_wait_2_async(self):
        # [FUNCTION DOC] Async variant of the loop-subgraph fork test: runs runAsync, waits on
        # the future and compares its value with the scripted reference.
        inp1 = torch.randn(5, 5)
        inp2 = torch.randn(5, 5)
        torch_graph = torch.jit.script(fork_wait_graph2)
        output_ref = torch_graph(inp1, inp2)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module.runAsync((inp1, inp2), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    Test Case: To test fork/wait operation in a graph on
    having multiple fork/wait operations
    """
    def test_fork_wait_3(self):
        # [FUNCTION DOC] Compares StaticModule and scripted outputs for the graph with ten
        # fork/wait pairs on a 3x3 ones tensor.
        input = torch.ones(3, 3)
        num_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph3)
        output_ref = torch_graph(input, num_forks)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module(input, num_forks)
        torch.testing.assert_close(output_test, output_ref)

    """
    Test Case: To test fork/wait operation in a graph with
    multiple fork/wait operations on runAsync API returning future
    """
    def test_fork_wait_3_async(self):
        # [FUNCTION DOC] Async variant of the multiple-fork test: runs runAsync with ten forks,
        # waits and compares the future value with the scripted reference.
        input = torch.ones(3, 3)
        num_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph3)
        output_ref = torch_graph(input, num_forks)
        static_runtime_module = StaticModule(torch_graph)
        output_test = static_runtime_module.runAsync((input, num_forks), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    Test Case: To test fork/wait operation in a graph on
    multiple nested fork/wait operations
    """
    @unittest.skip("Broken test: https://github.com/pytorch/pytorch/issues/109782")
    def test_fork_wait_4(self):
        # [FUNCTION DOC] Checks nested fork/wait (10 forks, 10 child forks each) matches the
        # scripted result. Skipped via a decorator because of a known broken issue.
        input = torch.ones(3, 3)
        num_forks = 10
        num_child_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph4)
        static_runtime_module = StaticModule(torch_graph)
        output_ref = torch_graph(input, num_forks, num_child_forks)
        output_test = static_runtime_module(input, num_forks, num_child_forks)
        torch.testing.assert_close(output_test, output_ref)

    """
    Test Case: To test fork/wait operation in a graph with multiple
    nested fork/wait operations on runAsync API returning future
    """
    @unittest.skip("Broken test: https://github.com/pytorch/pytorch/issues/109782")
    def test_fork_wait_4_async(self):
        # [FUNCTION DOC] Async variant of the nested fork/wait test using runAsync. Skipped
        # because of the same known broken issue.
        input = torch.ones(3, 3)
        num_forks = 10
        num_child_forks = 10
        torch_graph = torch.jit.script(fork_wait_graph4)
        static_runtime_module = StaticModule(torch_graph)
        output_ref = torch_graph(input, num_forks, num_child_forks)
        output_test = static_runtime_module.runAsync(
            (input, num_forks, num_child_forks), {})
        output_test.wait()
        torch.testing.assert_close(output_test.value(), output_ref)

    """
    Test Case: To test exception handling in fork/wait
    operation. Add.Tensor op is called for tensors with
    non-matching dims on the forked subgraph and the
    exception raised by subgraph is set on future returned
    by prim::fork to parent graph. Returned exception is
    checked for substring expected_error_msg as declared below
    """
    def test_fork_wait_exception(self):
        # [FUNCTION DOC] Runs the fork graph on tensors of shapes 4x7 and 4x5 so the forked add
        # fails. If an exception is raised, it must contain the expected size-mismatch message,
        # otherwise a RuntimeError is raised chained to the original error.
        # KNOWN ISSUE: if no exception is raised at all, the test passes silently.
        # incompatible tensors for add due to shape mismatch
        input1 = torch.randn(4, 7)
        input2 = torch.randn(4, 5)
        torch_graph = torch.jit.script(fork_wait_graph_exception)
        try:
            static_runtime_module = StaticModule(torch_graph)
            output_test = static_runtime_module(input1, input2)
        except Exception as error:
            expected_error_msg = (
                "The size of tensor a (7) must match the size "
                "of tensor b (5) at non-singleton dimension 1"
            )
            # test fails if error does not contain expected substr
            if str(error).find(expected_error_msg) == -1:
                raise RuntimeError(
                    "Tried execution of add.Tensors with incompatible shape. "
                    "Exception raised by forked runtime execution does "
                    f'not contain expected substring: "{expected_error_msg}"'
                ) from error

    """
    Test Case: To test exception handling in fork/wait
    operation with runAsync API. Add.Tensor op is called for
    tensors with non-matching dims on the forked subgraph
    and the exception raised by subgraph is set on future returned
    by prim::fork to parent graph. Returned exception is
    checked for substring expected_error_msg as declared below
    """
    def test_fork_wait_exception_async(self):
        # [FUNCTION DOC] Async variant of the exception test: calls runAsync with mismatched
        # shapes and validates the message of any exception raised by the call itself.
        # KNOWN ISSUE: if no exception is raised the test passes; an error stored on the
        #   returned future is never inspected since the future is not waited on.
        # incompatible tensors for add due to shape mismatch
        input1 = torch.randn(4, 7)
        input2 = torch.randn(4, 5)
        torch_graph = torch.jit.script(fork_wait_graph_exception)
        try:
            static_runtime_module = StaticModule(torch_graph)
            output_test = static_runtime_module.runAsync(
                (input1, input2), {})
        except Exception as error:
            expected_error_msg = (
                "The size of tensor a (7) must match the size "
                "of tensor b (5) at non-singleton dimension 1"
            )
            # test fails if error does not contain expected substr
            if str(error).find(expected_error_msg) == -1:
                raise RuntimeError(
                    "Tried execution of add.Tensors with incompatible shape. "
                    "Exception raised by forked runtime execution does "
                    f'not contain expected substring: "{expected_error_msg}"'
                ) from error

    def test_multihead_attention_layer(self):
        # [FUNCTION DOC] Scripts the attention layer in eval mode, runs it and a StaticModule on
        # random input of shape 128x8x256 (positional and with keyword arguments for value and
        # mask) and checks each returned tensor is close to the reference.
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device("cpu")
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        attention.eval()
        attention = torch.jit.script(attention)
        attention.eval()
        o_ref = attention(src, src, src, src_mask)

        attention_a = StaticModule(attention)
        o_test = attention_a(src, src, src, src_mask)
        o_test_kw = attention_a(src, src, value=src, mask=src_mask)

        for a, b in zip(o_ref, o_test):
            torch.testing.assert_close(a, b)

        for a, b in zip(o_ref, o_test_kw):
            torch.testing.assert_close(a, b)

    def test_multihead_attention_layer_benchmark(self):
        # [FUNCTION DOC] Smoke test of the benchmark and per-op benchmark APIs of StaticModule
        # on the scripted attention layer, with 2 warmup and 2 measured iterations. Asserts
        # nothing about the results.
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device("cpu")
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        attention.eval()
        attention = torch.jit.script(attention)
        attention_a = StaticModule(attention)

        attention_a.benchmark([src, src, src, src_mask], {}, 2, 2)
        metrics = attention_a.benchmark_individual_ops(
            [src, src, src, src_mask], {}, 2, 2
        )

    def test_mlp(self):
        # [FUNCTION DOC] Builds scripted bottom and top MLPs with the DLRM benchmark sizes,
        # wraps them in StaticModule, and checks outputs match the scripted modules for an
        # initial batch and for five further random batches of 2048 rows.
        # Arguments taken from benchmark script, ./bench/dlrm_s_benchmark.sh
        ln_bot = [512, 512, 64]
        sigmoid_bot = -1
        ln_top = [100, 1024, 1024, 1024, 1]
        sigmoid_top = 3
        bot_l = create_mlp(ln_bot, sigmoid_bot)
        bot_l_acc = StaticModule(bot_l)
        top_l = create_mlp(ln_top, sigmoid_top)
        top_l_acc = StaticModule(top_l)
        with torch.no_grad():
            bot_inp = torch.randn(2048, 512)  # torch.Size([2048, 512])
            top_inp = torch.randn(2048, 100)  # torch.Size([2048, 100])
        ref_bot = bot_l(bot_inp)
        acc_bot = bot_l_acc(bot_inp)
        torch.testing.assert_close(acc_bot, ref_bot)
        ref_top = top_l(top_inp)
        acc_top = top_l_acc(top_inp)
        torch.testing.assert_close(acc_top, ref_top)
        for _ in range(5):
            with torch.no_grad():
                bot_inp = torch.randn(2048, 512)  # torch.Size([2048, 512])
                top_inp = torch.randn(2048, 100)  # torch.Size([2048, 100])
            ref_bot = bot_l(bot_inp)
            acc_bot = bot_l_acc(bot_inp)
            torch.testing.assert_close(acc_bot, ref_bot)
            ref_top = top_l(top_inp)
            acc_top = top_l_acc(top_inp)
            torch.testing.assert_close(acc_top, ref_top)

    def test_trivial_graph(self):
        # [FUNCTION DOC] Checks the StaticModule output for the simple add/multiply graph with
        # 2x2 integer tensors equals the scripted result.
        s = torch.full((2, 2), 2)
        tg = torch.jit.script(trivial_graph)
        o_ref = tg(s, s, s)
        tg_a = StaticModule(tg)
        o_test = tg_a(s, s, s)
        torch.testing.assert_close(o_ref, o_test)

    def test_leaky_relu(self):
        # [FUNCTION DOC] Scripts a leaky relu module with slope 0.1 and checks the StaticModule
        # output on a random 5x5 tensor matches the scripted output.
        s = torch.randn(5, 5)
        tg = torch.jit.script(nn.LeakyReLU(0.1))
        o_ref = tg(s)
        tg_a = StaticModule(tg)
        o_test = tg_a(s)
        torch.testing.assert_close(o_ref, o_test)

    def test_attr(self):
        # [FUNCTION DOC] Checks SetAttr and GetAttr support: the scripted module is
        # wrapped in StaticModule and its output compared with the eager module on random
        # 2x2 input; then the benchmark and per-op benchmark APIs are exercised with
        # positional and keyword inputs.
        """
        TorchScript IR of TestModule() after freezing:
        graph(%self : __torch__.test_static_runtime.___torch_mangle_0.TestModule,
              %x.1 : Tensor):
            %18 : int = prim::Constant[value=30]()
            %30 : int = prim::Constant[value=13]()
            %3 : int = prim::Constant[value=20]()
            %2 : int = prim::Constant[value=1]()
            %self.sub2.a : int = prim::Constant[value=12]()
            %self.a : int = prim::Constant[value=3]()
            = prim::SetAttr[name="b"](%self, %3)
            %17 : Tensor = aten::add(%x.1, %30, %2)
            %7 : Tensor = aten::add(%17, %self.a, %2)
            %b.1 : int = prim::GetAttr[name="b"](%self)
            %9 : Tensor = aten::add(%7, %b.1, %2)
            %sub2 : __torch__.test_static_runtime.___torch_mangle_2.SubModule2 = prim::GetAttr[name="sub2"](%self)
            = prim::SetAttr[name="b"](%sub2, %18)
            %b : int = prim::GetAttr[name="b"](%sub2)
            %22 : int = aten::add(%self.sub2.a, %b)
            %23 : Tensor = aten::add(%x.1, %22, %2)
            %12 : Tensor = aten::add(%9, %23, %2)
            return (%12)
        """
        # test prim::SetAttr and prim::GetAttr impl in Static Runtime
        m = TestModule()

        m.eval()
        input = torch.randn(2, 2)
        output_s = m.forward(input)

        ms = torch.jit.script(m)
        sm = StaticModule(ms)
        output_sm = sm(input)
        torch.testing.assert_close(output_s, output_sm)
        sm.benchmark([input], {}, 2, 2)
        sm.benchmark_individual_ops([input], {}, 2, 2)
        sm.benchmark([], {"x": input}, 2, 2)
        sm.benchmark_individual_ops([], {"x": input}, 2, 2)

    @unittest.skip("Temporarily disabled")
    def test_fusion_trivial_graph(self):
        # [FUNCTION DOC] Fuses the trivial graph into a static subgraph, checks the graph
        # text contains the StaticSubgraph marker, and compares results. Skipped
        # (temporarily disabled).
        s = torch.full((2, 2), 2)
        tg = torch.jit.script(trivial_graph)
        o_ref = tg(s, s, s)
        torch._C._fuse_to_static_module(tg.graph)
        if "StaticSubgraph" not in str(tg.graph):
            raise AssertionError("StaticSubgraph not found in graph")
        o_test = tg(s, s, s)
        torch.testing.assert_close(o_ref, o_test)

    @unittest.skip("Temporarily disabled")
    def test_fusion_multihead_attention_layer(self):
        # [FUNCTION DOC] Fuses the scripted attention module into a static module and compares
        # its outputs with the pre-fusion reference. Skipped (temporarily disabled).
        HID_DIM = 256
        QUERY_LEN = 8
        BATCH_SIZE = 128
        LAYERS = 3
        HEADS = 8
        DROPOUT = 0.1
        device = torch.device("cpu")
        attention = MultiHeadAttentionLayer(HID_DIM, HEADS, DROPOUT, device).to(device)
        with torch.no_grad():
            src = torch.randn(BATCH_SIZE, QUERY_LEN, HID_DIM).to(device)
        src_mask = (src > 0)[:, :, 0].unsqueeze(1).unsqueeze(2).to(device)

        attention.eval()
        attention = torch.jit.script(attention)
        attention.eval()
        o_ref = attention(src, src, src, src_mask)

        torch._C._fuse_to_static_module(attention._c)
        o_test = attention(src, src, src, src_mask)

        for a, b in zip(o_ref, o_test):
            torch.testing.assert_close(a, b)

    @unittest.skip("Temporarily disabled")
    def test_fusion_loop(self):
        # [FUNCTION DOC] Fuses the scripted loop graph, checks for the StaticSubgraph marker and
        # compares outputs. Skipped (temporarily disabled).
        a = torch.randn(5, 5)
        b = torch.randn(5, 5)
        c = 4
        lg = torch.jit.script(loop_graph)
        o_ref = lg(a, b, c)
        torch._C._fuse_to_static_module(lg.graph)
        if "StaticSubgraph" not in str(lg.graph):
            raise AssertionError("StaticSubgraph not found in graph")
        o_test = lg(a, b, c)
        torch.testing.assert_close(o_ref, o_test)

    @unittest.skip("Temporarily disabled")
    def test_fusion_outputs(self):
        # [FUNCTION DOC] Fuses the scripted dictionary-returning graph, checks for the marker
        # and compares each dictionary value with the reference. Skipped (temporarily disabled).
        a = torch.randn(2, 2)
        b = torch.randn(2, 2)
        c = 4
        og = torch.jit.script(output_graph)
        o_ref = og(a, b, b, c)
        torch._C._fuse_to_static_module(og.graph)
        if "StaticSubgraph" not in str(og.graph):
            raise AssertionError("StaticSubgraph not found in graph")
        o_test = og(a, b, b, c)
        for i in o_ref:
            torch.testing.assert_close(o_ref[i], o_test[i])

    def test_create_object(self):
        # [FUNCTION DOC] Scripts and freezes a module that constructs a user-defined class
        # instance inside forward, then checks StaticModule gives the same output as the
        # scripted module.
        class Foo:
            # [CLASS DOC] HELPER: Plain class holding a tensor attribute x, created inside
            # scripted code.
            def __init__(self, x: torch.Tensor) -> None:
                # [FUNCTION DOC] HELPER: Stores the tensor in attribute x.
                self.x = x

        class Mod(torch.nn.Module):
            # [CLASS DOC] HELPER: Module whose forward builds an instance of the plain class
            # from the input and returns the input multiplied by the stored tensor.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Calls the base constructor only.
                super().__init__()

            def forward(self, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Wraps the input in the plain class and returns the
                # input times the wrapped tensor.
                foo = Foo(y)
                return y * foo.x

        mod = torch.jit.script(Mod()).eval()
        y = torch.randn((1, ))
        expected = mod(y)

        static_mod = StaticModule(torch.jit.freeze(mod))
        actual = static_mod(y)

        self.assertEqual(expected, actual)

if __name__ == "__main__":
    run_tests()
