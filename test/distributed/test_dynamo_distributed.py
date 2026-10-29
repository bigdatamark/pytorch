# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for TorchDynamo/torch.compile integration with torch.distributed: DDP graph
# splitting by DDPOptimizer, FSDP compilation and guard handling, compiler collectives
# (cross-rank compilation decisions), guard collectives, distributed autotuning, collective ops
# returning None under compile, and unsupported torch.distributed.nn.functional ops. Contains
# toy-model builders, a FakeDDP stand-in, and test classes: TestFakeDistributedSingleProc (no
# real process group), TestMultiGPU / TestMultiGPUDevice (multi-device, not truly distributed),
# TestMultiProc (spawned ranks, nccl/xccl), TestSingleProc (world size 1),
# TestNNFunctionalCompile. Import-time side effects: detects the installed transformers version
# to build expected-failure/skip decorators, creates a module logger, and instantiates
# device-type tests for TestMultiGPUDevice.
import contextlib
import copy
import functools
import logging
import random
import re
import unittest
from contextlib import contextmanager
from datetime import timedelta
from io import StringIO
from unittest.mock import patch

import numpy as np

import torch
import torch._dynamo
import torch._dynamo.logging
import torch._dynamo.test_case
import torch._inductor.test_case
import torch.distributed as dist
import torch.optim as optim
from torch import nn
from torch._C import FileCheck
from torch._dynamo import config
from torch._dynamo.backends.distributed import DDPOptimizer
from torch._dynamo.comptime import comptime
from torch._dynamo.device_interface import CudaInterface, DeviceGuard
from torch._dynamo.testing import collect_results, CompileCounter
from torch._dynamo.utils import same
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.compiler import set_enable_guard_collectives
from torch.distributed._functional_collectives import _maybe_wrap_tensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.wrap import (
    lambda_auto_wrap_policy,
    transformer_auto_wrap_policy,
)
from torch.nn.attention.flex_attention import flex_attention
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
    TEST_MULTIGPU,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import (
    _dynamo_dist_per_rank_init,
    DynamoDistributedMultiProcTestCase,
    DynamoDistributedSingleProcTestCase,
    import_transformers_or_skip,
    requires_accelerator_dist_backend,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    MI350_ARCH,
    skipIfRocmArch,
    skipIfTorchInductor,
    skipIfXpu,
)
from torch.testing._internal.inductor_utils import HAS_GPU
from torch.testing._internal.triton_utils import requires_cuda_and_triton


try:
    from importlib.metadata import version as pkg_version

    _transformers_version = tuple(
        int(x) for x in pkg_version("transformers").split(".")[:2]
    )
except Exception:
    _transformers_version = (0, 0)

# https://github.com/huggingface/transformers/issues/44188
# expectedFailure only applies to transformers >= 5.2 which introduced the bug
_expectedFailureIf_transformers_ge_5_2 = (
    unittest.expectedFailure if _transformers_version >= (5, 2) else lambda fn: fn
)
_skipIf_transformers_ge_5_2 = unittest.skipIf(
    _transformers_version >= (5, 2),
    "transformers >= 5.2 is affected by huggingface/transformers#44188",
)


log = logging.getLogger(__name__)


def reset_rng_state():
    # [FUNCTION DOC] HELPER: Reseeds torch, python random and numpy with the fixed seed 1337 so
    # eager and compiled runs see identical randomness.
    torch.manual_seed(1337)
    random.seed(1337)
    np.random.seed(1337)


def init_weights(m):
    # [FUNCTION DOC] HELPER: Module-apply initializer. For nn.Linear modules, applies Xavier
    # uniform init to the weight and fills the bias with 0.01 (in place); other modules are
    # left untouched.
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        m.bias.data.fill_(0.01)


@contextmanager
def enable_guard_collectives():
    # [FUNCTION DOC] HELPER: Context manager that turns on the torch.compiler guard-collectives
    # flag via set_enable_guard_collectives, yields, and restores the previous value in a
    # finally block. Usable as a decorator on tests.
    old = set_enable_guard_collectives(True)
    try:
        yield
    finally:
        set_enable_guard_collectives(old)


class ToyModel(nn.Module):
    # [CLASS DOC] HELPER: Four-block MLP (Linear + ReLU) with in_feat -> hidden_feat ->
    # hidden_feat -> hidden_feat -> out_feat sizes. Optionally runs the forward inside a user
    # supplied context manager factory (ctx_manager, called with no arguments).
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        # [FUNCTION DOC] HELPER: Stores ctx_manager and builds the sequential net of 4
        # Linear+ReLU layers.
        super().__init__()
        self.ctx_manager = ctx_manager
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Runs the net on inputs, inside a fresh ctx_manager() context if
        # one was given; returns the output tensor.
        if self.ctx_manager is not None:
            with self.ctx_manager():
                return self.net(inputs)
        else:
            return self.net(inputs)


def get_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    # [FUNCTION DOC] HELPER: Builds a ToyModel with the given sizes on device, applies
    # init_weights, creates random inputs of shape (bsz, in_feat) and runs the eager forward.
    # Returns (model, inputs, outputs).
    m = ToyModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs


class MutatingModel(nn.Module):
    # [CLASS DOC] HELPER: Same MLP as ToyModel but forward assigns the attribute state to 2 and
    # multiplies the output by it, so the module is mutated during forward (used to trigger
    # setattr graph breaks). Initial state is 1.
    def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None):
        # [FUNCTION DOC] HELPER: Builds the 4-block MLP and sets state to 1. ctx_manager is
        # stored but unused by forward.
        super().__init__()
        self.ctx_manager = ctx_manager
        self.net = nn.Sequential(
            *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
            + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
        )
        self.state = 1

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Mutates self.state to 2 and returns the net output multiplied
        # by state.
        self.state = 2
        return self.net(inputs) * self.state


def get_mutating_model(
    device, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
):
    # [FUNCTION DOC] HELPER: Like get_model but builds a MutatingModel. Returns (model, inputs,
    # outputs) with outputs from an eager forward (which also mutates the model state).
    m = MutatingModel(
        in_feat=in_feat,
        hidden_feat=hidden_feat,
        out_feat=out_feat,
        ctx_manager=ctx_manager,
    ).to(device)
    m.apply(init_weights)
    inputs = torch.rand(bsz, in_feat).to(device)
    outputs = m(inputs)
    return m, inputs, outputs


class ForcedGetAttrMod(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a registered Linear plus a second Linear hidden in the
    # instance __dict__ (not a registered submodule), and a call counter incremented in forward.
    # Used to test getattr on attributes that bypass nn.Module registration under FSDP.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Creates the registered linear, stores a second Linear on device
        # directly in __dict__ under forced_linear, and zeroes the counter.
        super().__init__()
        self.linear = torch.nn.Linear(1, 1)
        self.__dict__["forced_linear"] = torch.nn.Linear(1, 1).to(device=device)
        self.counter = 0

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Increments counter and returns x times linear(x) times the
        # forced linear weight.
        self.counter += 1
        return x * self.linear(x) * self.forced_linear.weight


def get_forced_getattr_module(device):
    # [FUNCTION DOC] HELPER: Builds a ForcedGetAttrMod on device with a (1, 1) random input.
    # Returns (module, input, eager output).
    mod = ForcedGetAttrMod(device).to(device=device)
    x = torch.randn(1, 1, device=device)
    return mod, x, mod(x)


class ToyInnerModel(nn.Module):
    # [CLASS DOC] HELPER: Two 100x100 Linear layers in a Sequential; building block for the
    # outer model.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Builds the Sequential of two Linear(100, 100) layers.
        super().__init__()
        self.layers = [nn.Linear(100, 100), nn.Linear(100, 100)]
        self.layers = nn.Sequential(*self.layers)

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Applies the layers to inputs.
        return self.layers(inputs)


class ToyOuterModel(nn.Module):
    # [CLASS DOC] HELPER: Sequential of two ToyInnerModel blocks separated by ReLUs, with each
    # inner block placed on the given device.
    def __init__(self, device):
        # [FUNCTION DOC] HELPER: Builds two ToyInnerModel instances on device and chains them
        # with ReLUs.
        super().__init__()
        self.layers = [ToyInnerModel().to(device) for _ in range(2)]
        self.layers = nn.Sequential(
            self.layers[0], nn.ReLU(), self.layers[1], nn.ReLU()
        )

    def forward(self, inputs):
        # [FUNCTION DOC] HELPER: Applies the layers to inputs.
        return self.layers(inputs)


def get_toy_model_for_activation_checkpointing(device):
    # [FUNCTION DOC] HELPER: Builds a ToyOuterModel on device, applies init_weights and creates
    # a random (100, 100) input. Returns (model, inputs).
    m = ToyOuterModel(device).to(device)
    m.apply(init_weights)
    inputs = torch.rand(100, 100).to(device)
    return m, inputs


def find_first_node(gm, func):
    # [FUNCTION DOC] HELPER: Returns the first node in gm.graph whose target is the given
    # callable, or None if there is none.
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def apply_fsdp_with_checkpointing(
    model, wrap_policy, checkpoint_policy, use_activation_checkpointing=True
):
    # [FUNCTION DOC] HELPER: Deep copies model and wraps it in FSDP using wrap_policy and
    # original parameters. If use_activation_checkpointing is true, applies non-reentrant
    # activation checkpointing to the submodules selected by checkpoint_policy (used as
    # check_fn). Returns the model.
    from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
        apply_activation_checkpointing,
        checkpoint_wrapper,
        CheckpointImpl,
    )

    model = FSDP(
        copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
    )
    if use_activation_checkpointing:
        checkpoint_wrapper_fn = functools.partial(
            checkpoint_wrapper,
            checkpoint_impl=CheckpointImpl.NO_REENTRANT,
        )
        apply_activation_checkpointing(
            model,
            checkpoint_wrapper_fn=checkpoint_wrapper_fn,
            check_fn=checkpoint_policy,
        )
    return model


def get_custom_model(device):
    # [FUNCTION DOC] HELPER: Builds a model of three Linear(512)+ReLU blocks where the middle
    # one is a custom module using torch.where with scalar branches (edge case for DDPOptimizer
    # fake tensor wrapping). Applies init_weights; the inputs are a tuple of the same (512, 512)
    # tensor twice (duplicate inputs). Returns (model, inputs tuple, eager outputs).
    class MyCustomLinear(torch.nn.Module):
        # [CLASS DOC] HELPER: Module with a 512x512 weight whose forward adds torch.where(tmp <
        # 0.5, 0.3, 0.6) to a matmul result, exercising a scalar where decomposition.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the 512x512 random weight parameter.
            super().__init__()
            self.weight = nn.Parameter(torch.randn(512, 512))

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Matmul of x with the transposed weight, plus the
            # where-based scalar term.
            tmp = torch.mm(x, self.weight.t())
            # test an edge case where torch.where.scalar was decomposed to aten.where.self(tensor, tensor, tensor)
            # and the tensors T(0.4) and T(0.5) were not wrapped in FakeTensors during DDPOptimizer compilation
            return tmp + torch.where(tmp < 0.5, 0.3, 0.6)

    class MyLinear(torch.nn.Module):
        # [CLASS DOC] HELPER: Thin wrapper around Linear(512, 512).
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Creates the inner Linear(512, 512).
            super().__init__()
            self.linear = torch.nn.Linear(512, 512)

        def forward(self, x):
            # [FUNCTION DOC] HELPER: Applies the inner linear to x.
            return self.linear(x)

    class MyModule(torch.nn.Module):
        # [CLASS DOC] HELPER: Sequential of three (layer, ReLU) pairs with the custom layer in
        # the middle so parameterless ops occur before and after it.
        def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the flattened Sequential from the three
            # layer/ReLU pairs.
            super().__init__()
            mods = [
                (MyLinear(), torch.nn.ReLU()),
                # sandwich the custom in the middle so it comes before and after
                (MyCustomLinear(), torch.nn.ReLU()),
                (MyLinear(), torch.nn.ReLU()),
            ]
            self.seq = torch.nn.Sequential(*[x for items in mods for x in items])

        def forward(self, x, y):
            # [FUNCTION DOC] HELPER: Runs the sequence on x + y; takes two inputs to model
            # duplicated graph inputs.
            # test special case where the 0th bucket (layers close to graph input) is at capacity, which would
            # trigger a new bucket, but there are only trivial ops without parameters to put into the new bucket.
            # optimize this case by fusing that 'empty bucket' back together with the previous full one
            return self.seq(x + y)

    m = MyModule().to(device)
    m.apply(init_weights)
    inputs = torch.rand((512, 512)).to(device)
    # test duplicated inputs
    inputs = (inputs, inputs)
    correct_outputs = m(*inputs)
    return m, inputs, correct_outputs


def get_hf_bert(rank):
    # [FUNCTION DOC] HELPER: Builds a randomly initialized BERT masked-LM from the default
    # BertConfig on the current accelerator (or cpu) device indexed by rank, in train mode, plus
    # a dict of random input_ids and labels of shape (4, 512). Returns (model, inputs). Raises
    # unittest.SkipTest if transformers cannot be imported.
    # Note: use @import_transformers_or_skip on your test case if you use this
    # in a multiprocessing test
    try:
        from transformers import AutoModelForMaskedLM, BertConfig
    except ImportError as e:
        raise unittest.SkipTest("Unable to import transformers") from e

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )
    batch_size, max_length, config, device = (
        4,
        512,
        BertConfig(),
        f"{device_type}:{rank}",
    )
    model = AutoModelForMaskedLM.from_config(config).to(device)
    input_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(device)
    decoder_ids = torch.randint(0, config.vocab_size, (batch_size, max_length)).to(
        device
    )
    inputs = {"input_ids": input_ids, "labels": decoder_ids}
    model.train()
    return model, inputs


class CheckSplitsCompiler:
    # [CLASS DOC] HELPER: Counting backend: compile_fn records how many times the compiler
    # was invoked (one call per graph split produced by DDPOptimizer) and returns the graph
    # module unchanged.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: Initializes the compiler_called counter to 0.
        self.compiler_called = 0

    def compile_fn(self, gm, example_inputs):
        # [FUNCTION DOC] HELPER: Backend function: increments compiler_called and returns gm
        # without compiling.
        self.compiler_called += 1
        return gm


# This simulates DDP, but it doesn't actually do any process communication;
# it just has enough properties so that the dynamo distributed optimization is
# able to optimize.  Feel free to simulate more properties as necessary.  The
# other important thing is patching _active_ddp_module, which is what actually
# triggers DDP optimization
class FakeDDP(nn.Module):
    # [CLASS DOC] HELPER: Lightweight DDP stand-in without communication. Wraps a module,
    # exposes bucket_bytes_cap (megabytes converted to bytes) and sets the DDP class attribute
    # _active_ddp_module during forward, which is what activates the dynamo DDP optimization.
    def __init__(self, module, bucket_cap_mb=25):
        # [FUNCTION DOC] HELPER: Stores the wrapped module and the bucket cap converted from MB
        # to bytes.
        super().__init__()
        self.module = module
        self.bucket_bytes_cap = int(bucket_cap_mb * 1024 * 1024)

    @contextmanager
    def _inside_ddp_forward(self):
        # [FUNCTION DOC] HELPER: Context manager that sets DDP._active_ddp_module to this
        # wrapper and restores the previous value (not None) on exit, so nested instances
        # restore the outer one.
        old = DDP._active_ddp_module
        DDP._active_ddp_module = self
        try:
            yield
        finally:
            DDP._active_ddp_module = old

    def forward(self, *inputs, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the wrapped module's forward. If no DDP module is active
        # yet, runs it inside the _inside_ddp_forward context; otherwise calls it directly.
        if not DDP._active_ddp_module:
            with self._inside_ddp_forward():
                return self.module.forward(*inputs, **kwargs)
        else:
            return self.module.forward(*inputs, **kwargs)


def run_hf_bert_ddp(self, model, inputs, backend):
    # [FUNCTION DOC] HELPER: Shared HF BERT check. Runs eager forward and backward with seed
    # reset, then compiles model with the given backend and repeats with the same seed, and
    # asserts (on the test case self) that results collected by collect_results for both runs
    # are equal under same().
    reset_rng_state()
    correct_outputs = model(**inputs)
    correct_loss = correct_outputs.loss
    correct_loss.backward()

    reset_rng_state()
    opt_model = torch.compile(model, backend=backend)
    opt_outputs = opt_model(**inputs)
    opt_loss = opt_outputs.loss
    opt_loss.backward()

    inputs_flat = [inputs[k] for k in inputs]
    correct_results = collect_results(
        model, correct_outputs.logits, correct_loss, inputs_flat
    )
    opt_results = collect_results(opt_model, opt_outputs.logits, opt_loss, inputs_flat)
    self.assertTrue(same(correct_results, opt_results))


class TestFakeDistributedSingleProc(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Dynamo tests of the DDP optimizer using FakeDDP, so no process group or
    # multiple processes are needed. Covers HF BERT, unbacked symbol splitting across graph
    # partitions, method-call forward, and nested DDP active-module restoration.
    @_expectedFailureIf_transformers_ge_5_2
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @unittest.skipIf(
        torch._inductor.config.triton.native_matmul,
        "FIXME : native matmul fails. RuntimeError: Cannot access data pointer of Tensor",
    )
    def test_hf_bert_ddp_inductor(self):
        # [FUNCTION DOC] Compiles FakeDDP-wrapped HF BERT with inductor under optimize_ddp and
        # compares with eager via run_hf_bert_ddp. Expected failure on transformers >= 5.2;
        # skipped without a GPU or when native_matmul is enabled.
        model, inputs = get_hf_bert(0)
        model = FakeDDP(model)
        run_hf_bert_ddp(self, model, inputs, "inductor")

    @_expectedFailureIf_transformers_ge_5_2
    @patch.object(config, "optimize_ddp", True)
    def test_hf_bert_ddp_aot_eager(self):
        # [FUNCTION DOC] Same as the inductor variant but with the aot_eager backend on
        # FakeDDP-wrapped BERT; expected failure on transformers >= 5.2.
        model, inputs = get_hf_bert(0)
        model = FakeDDP(model)
        run_hf_bert_ddp(self, model, inputs, "aot_eager")

    @patch.object(config, "optimize_ddp", True)
    def test_issue90375(self):
        # [FUNCTION DOC] Regression test (issue 90375): compiles a FakeDDP-wrapped module with
        # no inputs whose forward returns a product of random tensors, using aot_eager, and just
        # checks that it runs.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Input-less module returning randn(3) * randn(3).
            def forward(self):
                # [FUNCTION DOC] HELPER: Returns the product of two random vectors of size 3.
                return torch.randn(3) * torch.randn(3)

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(model, backend="aot_eager")
        opt_model()

    @patch.object(config, "optimize_ddp", True)
    def test_symbol_splitting(self):
        # [FUNCTION DOC] Compiles a FakeDDP model with dynamic shapes whose forward concatenates
        # the input with itself and applies two matmuls; checks DDPOptimizer can split the graph
        # where symbolic sizes appear in several partitions without raising.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two 512x512 parameters used in a cat plus two matmuls.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates the two weight parameters.
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Concatenates x with itself, then returns x + (x @ w1)
                # @ w2.
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = x + y @ self.weight2
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512))

    @patch.object(config, "optimize_ddp", True)
    def test_ddp_optimizer_inductor_strides_dont_specialize(self):
        # [FUNCTION DOC] Compiles a FakeDDP two-Linear model after marking dim 1 dynamic on two
        # inputs of different length, with DDP._active_ddp_module set manually. Asserts that the
        # second call causes no additional compilation (no stride specialization); resets the
        # active module in a finally.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two Linear(768, 768) layers applied in sequence.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates fc_0 and fc_1.
                super().__init__()
                self.fc_0 = nn.Linear(768, 768)
                self.fc_1 = nn.Linear(768, 768)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies fc_0 then fc_1.
                x = self.fc_0(x)
                x = self.fc_1(x)
                return x

        model = Model()
        model = FakeDDP(model)

        inp = torch.randn((16, 18, 768))
        inp2 = torch.randn((16, 20, 768))

        torch._dynamo.mark_dynamic(inp, 1)
        torch._dynamo.mark_dynamic(inp2, 1)

        torch._dynamo.utils.clear_compilation_metrics()
        torch._dynamo.reset()
        try:
            DDP._active_ddp_module = model
            opt_model = torch.compile(model)
            self.assertEqual(0, len(torch._dynamo.utils.get_compilation_metrics()))
            opt_model(inp)
            compile_count_before = len(torch._dynamo.utils.get_compilation_metrics())
            opt_model(inp2)
            compile_count_after = len(torch._dynamo.utils.get_compilation_metrics())
            # no recompiles
            self.assertEqual(compile_count_before, compile_count_after)
        finally:
            DDP._active_ddp_module = None

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_direct(self):
        # [FUNCTION DOC] Runs a dynamic-shape compile of a FakeDDP model where a scalar from
        # y.tolist() (an unbacked symbol) is used directly in the last partition; checks
        # that DDPOptimizer splitting copes with the symbol defined earlier than used.
        # Requires capture_scalar_outputs.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two-weight model using an unbacked scalar multiplier.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two 512x512 parameters.
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Reads the first scalar of y via tolist, builds the cat
                # plus two-matmul value and scales it by that scalar.
                u0, _ = y.tolist()
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * u0
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_indirect(self):
        # [FUNCTION DOC] Like the direct variant, but the unbacked scalar is used indirectly: it
        # sizes a tensor of ones whose sum scales the result. Verifies splitting with the symbol
        # flowing through a derived size. Requires capture_scalar_outputs.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two-weight model using an unbacked-size tensor sum
            # as multiplier.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two 512x512 parameters.
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Builds ones sized by the first scalar of y and scales
                # the matmul chain by its sum.
                u0, _ = y.tolist()
                a = torch.ones(u0)
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * a.sum()
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_scalar_outputs=True)
    def test_unbacked_symbol_splitting_torture_multi(self):
        # [FUNCTION DOC] Stress test with three weights and bucket_cap_mb=1 so the unbacked
        # symbol definition, its derived tensor and its use fall in three different partitions;
        # just checks compile and run.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Three-weight model whose forward is annotated by partition.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates three 512x512 parameters.
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))
                self.weight3 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Defines the scalar, runs two matmuls, builds the ones
                # tensor, then applies the third matmul and scales by the ones sum.
                # partition one (contains the u0 def)
                u0, _ = y.tolist()
                x = torch.cat([x, x])
                y1 = x @ self.weight1
                # partition two (contains the variable)
                y2 = y1 @ self.weight2
                a = torch.ones(u0)
                # partition three
                z = (x + y2 @ self.weight3) * a.sum()
                return z

        model = Model()
        model = FakeDDP(model, bucket_cap_mb=1)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([12, 13]))

    @config.patch(optimize_ddp=True, capture_dynamic_output_shape_ops=True)
    def test_unbacked_symbol_splitting_no_binding(self):
        # [FUNCTION DOC] Uses nonzero (data-dependent output shape) whose unbacked size
        # is not bound to a graph input; checks DDPOptimizer splitting still works.
        # Requires capture_dynamic_output_shape_ops.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two-weight model scaled by a sum over nonzero indices.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two 512x512 parameters.
                super().__init__()
                self.weight1 = nn.Parameter(torch.randn(512, 512))
                self.weight2 = nn.Parameter(torch.randn(512, 512))

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: Computes y.nonzero(), the cat plus matmul chain, and
                # multiplies by the sum of nonzero indices plus one.
                nz = y.nonzero()
                x = torch.cat([x, x])
                y = x @ self.weight1
                z = (x + y @ self.weight2) * (nz + 1).sum()
                return z

        model = Model()
        model = FakeDDP(model)

        opt_model = torch.compile(dynamic=True)(model)
        opt_model(torch.randn(20, 512), torch.tensor([0.0, 12.0, 0.0, 11.0]))

    @patch.object(config, "optimize_ddp", True)
    def test_call_method_forward(self):
        # [FUNCTION DOC] Compiles a FakeDDP model whose forward calls submodule .forward()
        # explicitly (LayerNorm called directly, MultiheadAttention via the forward
        # method), checking DDPOptimizer handles call_method nodes. Runs once on a (2, 129,
        # 100, 96) input.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: Two ModuleList layers each holding a LayerNorm(96) and a
            # 4-head attention.
            def __init__(
                self,
            ):
                # [FUNCTION DOC] HELPER: Builds the list of two layers and stores them as
                # a ModuleList.
                super().__init__()
                layers = []
                for _ in range(2):
                    layer = nn.ModuleList(
                        [
                            nn.LayerNorm(96),
                            nn.MultiheadAttention(
                                embed_dim=96, num_heads=4, batch_first=True
                            ),
                        ]
                    )
                    layers.append(layer)
                self.layers = nn.ModuleList(layers)

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: For each layer, reshapes to (B*F, T, H), applies
                # layernorm, calls the attention module's forward explicitly with x as q, k, v,
                # and reshapes back.
                # x: [Batch, Freq, Time, Feature]
                B, F, T, H = x.shape
                for m in self.layers:
                    x = x.reshape(B * F, T, H)
                    x = m[0](x)
                    x, _ = m[1].forward(x, x, x)
                    x = x.reshape(B, F, T, H)
                return x

        model = Model()
        model = FakeDDP(model)
        opt_model = torch.compile(model)
        opt_model(torch.randn(2, 129, 100, 96))

    @patch.object(config, "optimize_ddp", True)
    def test_nested_ddp_preserves_active_ddp_module(self):
        # [FUNCTION DOC] Checks that nested FakeDDP instances restore DDP._active_ddp_module to
        # the outer one on exit (None only after the outer exits), then wraps
        # DDPOptimizer.__init__ with a tracker and asserts DDPOptimizer is actually created when
        # compiling the nested model with aot_eager.
        """
        When an inner DDP (e.g., TorchRec's data-parallel embedding lookup) is
        nested inside an outer DDP, the inner DDP's _inside_ddp_forward exit
        must restore _active_ddp_module to the outer DDP, not clear it to None.
        Otherwise DDPOptimizer won't activate for compiled regions that run
        after the inner DDP forward.
        """

        class InnerModule(nn.Module):
            # [CLASS DOC] HELPER: Single Linear(10, 10) module.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the linear.
                super().__init__()
                self.linear = nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the linear.
                return self.linear(x)

        class OuterModel(nn.Module):
            # [CLASS DOC] HELPER: Contains a FakeDDP wrapped inner module and a 10x10 weight;
            # forward applies the inner DDP then a matmul.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates inner_ddp and the weight.
                super().__init__()
                self.inner_ddp = FakeDDP(InnerModule())
                self.weight = nn.Parameter(torch.randn(10, 10))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Runs the inner DDP, then multiplies by the weight.
                # Inner DDP forward runs first (like DP embedding lookup)
                x = self.inner_ddp(x)
                # After inner DDP exits, _active_ddp_module must still be set
                # for DDPOptimizer to split this region
                x = x @ self.weight
                return x

        outer_model = OuterModel()
        outer_ddp = FakeDDP(outer_model)

        # Verify the context manager nesting directly
        self.assertIsNone(DDP._active_ddp_module)
        with outer_ddp._inside_ddp_forward():
            self.assertIs(DDP._active_ddp_module, outer_ddp)
            with outer_ddp.module.inner_ddp._inside_ddp_forward():
                self.assertIs(DDP._active_ddp_module, outer_ddp.module.inner_ddp)
            # After inner exits, outer must be restored
            self.assertIs(DDP._active_ddp_module, outer_ddp)
        # After outer exits, must be None again
        self.assertIsNone(DDP._active_ddp_module)

        # Verify DDPOptimizer activates with nested DDP during compilation
        ddp_optimizer_activated = False
        orig_init = DDPOptimizer.__init__

        def tracking_init(self_opt, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Replacement for DDPOptimizer.__init__ that sets the
            # enclosing flag ddp_optimizer_activated to true and delegates to the
            # original constructor.
            nonlocal ddp_optimizer_activated
            ddp_optimizer_activated = True
            return orig_init(self_opt, *args, **kwargs)

        with patch.object(DDPOptimizer, "__init__", tracking_init):
            opt_model = torch.compile(outer_ddp, backend="aot_eager")
            opt_model(torch.randn(4, 10))

        self.assertTrue(
            ddp_optimizer_activated,
            "DDPOptimizer should activate for compiled regions inside nested DDP",
        )


# These tests aren't really distributed, but need multiple GPUs to run
class TestMultiGPU(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Compile tests that need more than one GPU but do not use process groups:
    # device exchange, DeviceGuard, events, streams, current-device handling and device-keyed
    # cache behavior. Tests are skipped when TEST_MULTIGPU is false.
    @unittest.skipIf(not TEST_MULTIGPU, "Requires multiple gpus")
    def test_cuda__exchange_device(self):
        # [FUNCTION DOC] Compares eager and fullgraph eager-compiled results of a function using
        # torch.cuda._exchange_device(0); recompiles under device 1 and checks the current
        # device is restored to its initial value.
        def fn(x):
            # [FUNCTION DOC] HELPER: Switches to device 0 via _exchange_device, computes sin(x +
            # previous device index), restores the device, and returns the result.
            dev = torch.cuda._exchange_device(0)
            x = torch.sin(x + dev)
            torch.cuda._maybe_exchange_device(dev)
            return x

        initial_dev = torch.cuda.current_device()
        x = torch.randn((2, 2), device="cuda")
        ref = fn(x)
        opt_fn = torch.compile(backend="eager", fullgraph=True)(fn)
        res = opt_fn(x)
        self.assertEqual(ref, res)

        # make sure we recompile if device changes
        with torch.cuda.device(1):
            ref = fn(x)
            res = opt_fn(x)
        self.assertEqual(ref, res)
        self.assertEqual(torch.cuda.current_device(), initial_dev)

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_device_guard(self):
        # [FUNCTION DOC] Checks DeviceGuard(CudaInterface, 1): inside the guard the current
        # device is 1 and prev_idx/idx are 0 and 1; after exit the original device is restored
        # (attributes remain).
        current_device = torch.cuda.current_device()

        device_guard = DeviceGuard(CudaInterface, 1)

        with device_guard as _:
            self.assertEqual(torch.cuda.current_device(), 1)
            self.assertEqual(device_guard.prev_idx, 0)
            self.assertEqual(device_guard.idx, 1)

        self.assertEqual(torch.cuda.current_device(), current_device)
        self.assertEqual(device_guard.prev_idx, 0)
        self.assertEqual(device_guard.idx, 1)

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_new_event_api(self) -> None:
        # [FUNCTION DOC] Uses a custom backend that creates events through new_event,
        # records one on a stream of cuda:1, asserts two events are distinct objects, and
        # inserts a get_external_object_by_index call node into the graph. Then runs a
        # compiled add on cuda:0.
        from torch._dynamo.graph_bytecode_inputs import get_external_object_by_index
        from torch._dynamo.variables.streams import new_event

        def event_generation_backend(gm, *args, **kwargs):  # type: ignore[no-untyped-def]
            # [FUNCTION DOC] HELPER: Backend that does the event creation checks and graph edit
            # described by the enclosing test; returns the modified graph module.
            e0_ind = new_event()
            with torch.Stream(device="cuda:1"):
                get_external_object_by_index(e0_ind).record()
            e1_ind = new_event()
            self.assertNotEqual(e0_ind, e1_ind)
            self.assertNotEqual(
                get_external_object_by_index(e0_ind),
                get_external_object_by_index(e1_ind),
            )
            with gm.graph.inserting_after(next(iter(gm.graph.nodes))):
                gm.graph.call_function(
                    get_external_object_by_index, args=(1,), kwargs={}
                )
            return gm

        @torch.compile(backend=event_generation_backend)
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x + 1.
            return x + 1

        fn(torch.ones(2, 2, device="cuda:0"))

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_get_current_stream_return_no_index(self):
        # [FUNCTION DOC] Fullgraph compiles a function that nests stream contexts for cuda:1 and
        # cuda:0 and returns the current stream of the index-less cuda device; asserts it equals
        # the eager result.
        def fn(x, s0, s1):
            # [FUNCTION DOC] HELPER: Enters s1 then s0 and returns the accelerator current
            # stream for cuda.
            with s1:
                with s0:
                    s = torch.accelerator.current_stream(torch.device("cuda"))
            return s

        s0 = torch.Stream(device="cuda:0")
        s1 = torch.Stream(device="cuda:1")
        inp = (torch.ones(2, 2) + 1, s0, s1)
        fn_opt = torch.compile(fn, fullgraph=True)
        s_act = fn_opt(*inp)
        s_exp = fn(*inp)
        self.assertEqual(s_act, s_exp)

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_get_current_stream_return_different_device(self):
        # [FUNCTION DOC] Same as the previous test but queries the current stream of
        # device cuda:1 while the innermost context is cuda:0; asserts compiled and eager
        # results match.
        def fn(x, s0, s1):
            # [FUNCTION DOC] HELPER: Enters s1 then s0 and returns the current stream of cuda:1.
            with s1:
                with s0:
                    s = torch.accelerator.current_stream(torch.device("cuda:1"))
            return s

        s0 = torch.Stream(device="cuda:0")
        s1 = torch.Stream(device="cuda:1")
        inp = (torch.ones(2, 2) + 1, s0, s1)
        fn_opt = torch.compile(fn, fullgraph=True)
        s_act = fn_opt(*inp)
        s_exp = fn(*inp)
        self.assertEqual(s_act, s_exp)

    @unittest.skipIf(not TEST_MULTIGPU, "detected only one GPU")
    def test_gpu_current_device(self):
        # [FUNCTION DOC] Checks that a function allocating on the current accelerator device
        # index is recompiled when the current device changes: one frame under device 0 and two
        # after entering device 1, with eager-equal outputs.
        def fn(x):
            # [FUNCTION DOC] HELPER: Allocates an empty (2, 3) tensor on the current device
            # index, copies x in, and returns sin(y + device index).
            y = torch.empty(
                (2, 3),
                dtype=torch.float32,
                device=torch.accelerator.current_device_index(),
            )
            y.copy_(x)
            return torch.sin(y + y.device.index)

        counter = torch._dynamo.testing.CompileCounter()
        opt_fn = torch.compile(backend=counter, fullgraph=True)(fn)

        with torch.accelerator.device_index(0):
            x = torch.randn(2, 3)
            self.assertEqual(opt_fn(x), fn(x))
            self.assertEqual(counter.frame_count, 1)
            with torch.accelerator.device_index(1):
                self.assertEqual(opt_fn(x), fn(x))
                self.assertEqual(counter.frame_count, 2)

    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_symint_as_device_kwarg_multi_gpu(self):
        # [FUNCTION DOC] Compiles dynamically a function that uses a tensor size minus 2 as the
        # device index. After calling with sizes 2 then 3, asserts that a guard failure was
        # reported mentioning a size mismatch of tensor rank at index 0.
        def fn(rank):
            # [FUNCTION DOC] HELPER: Returns ones of size 10 on the device whose index is
            # rank.size(0) - 2.
            # -2 to make device id smaller for easier testing on CI
            return torch.ones(10, device=rank.size(0) - 2)

        x = torch.randn(2)
        out = fn(torch.randn(2))

        guard_failure = None

        def guard_failures(failure):
            # [FUNCTION DOC] HELPER: Guard failure callback that stores the failure in the
            # enclosing variable.
            nonlocal guard_failure
            guard_failure = failure

        opt_fn = torch._dynamo.optimize(
            "eager", guard_fail_fn=guard_failures, dynamic=True
        )(fn)
        self.assertEqual(out, opt_fn(x))

        x = torch.randn(3)
        self.assertEqual(fn(x), opt_fn(x))
        self.assertTrue(guard_failure is not None)
        self.assertIn("""tensor 'rank' size mismatch at index 0""", guard_failure[0])

    def _clear_dynamo_and_codecache(self):
        # [FUNCTION DOC] HELPER: Resets dynamo state and purges the inductor PyCodeCache.
        """
        Clear unrelated caches, like dynamo and PyCodeCache
        """
        torch._dynamo.reset()
        torch._inductor.codecache.PyCodeCache.cache_clear(purge=True)

    @torch._inductor.config.patch("fx_graph_cache", True)
    @torch._inductor.config.patch("fx_graph_remote_cache", False)
    @torch._functorch.config.patch({"enable_autograd_cache": True})
    @unittest.skipIf(not TEST_MULTIGPU, "only one GPU detected")
    def test_constant_tensor_device_guards(self):
        # [FUNCTION DOC] With fx graph and autograd caches enabled, compiles a no-argument
        # function creating a CUDA tensor under device 0, clears caches, and recompiles under
        # device 1; asserts each result is on the matching device, i.e. the device index is part
        # of the cache key.
        """
        Usually, when there are example inputs, the device index of the inputs
        is sufficient to make sure we don't cache hit with the results from different
        cuda devices.
        When the input has no arguments, we still need to have the cuda
        device index in the cache key.
        """

        @torch.compile(backend="inductor")
        def f():
            # [FUNCTION DOC] HELPER: Creates a tensor [5] on the current CUDA device and returns
            # it in a tuple.
            y = torch.tensor([5], device="cuda")
            return (y,)

        with torch.cuda._DeviceGuard(0):
            torch.cuda.set_device(0)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:0"))

        self._clear_dynamo_and_codecache()

        with torch.cuda._DeviceGuard(1):
            torch.cuda.set_device(1)
            result = f()
            self.assertEqual(result[0].device, torch.device("cuda:1"))


class TestMultiGPUDevice(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Device-parametrized test holder (instantiated for cuda, hpu, xpu via
    # instantiate_device_type_tests); skipped unless multiple GPUs exist.
    @unittest.skipIf(not TEST_MULTIGPU, "need multiple GPU")
    def test_gpu_set_device(self, device):
        # [FUNCTION DOC] Compiles a function that allocates ones on the device and then calls
        # set_device(1) (causing a graph break) while running under device 0. Asserts the result
        # is on the device type with index 0 and that two frames were compiled.
        def fn():
            # [FUNCTION DOC] HELPER: Allocates ones, switches the device module to index 1,
            # returns the tensor plus one.
            a = torch.ones(2, device=device)
            torch.get_device_module(device).set_device(1)
            return a + 1

        with torch.get_device_module(device).device(0):
            counter = CompileCounter()
            opt_fn = torch.compile(fn, backend=counter)
            res = opt_fn()
            self.assertTrue(res.device.type in device)
            self.assertEqual(res.device.index, 0)
            self.assertEqual(counter.frame_count, 2)


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(
    TestMultiGPUDevice, globals(), only_for=devices, allow_xpu=True
)


# Are these tests failing?  Check and see if TestFakeDistributedSingleProc has a
# single process version; if it's just a problem in the Dynamo distributed
# # optimizer, you should be able to repro it single process!
@requires_accelerator_dist_backend(["nccl", "xccl"])
class TestMultiProc(DynamoDistributedMultiProcTestCase):
    # [CLASS DOC] Multi-process (spawned, one rank per GPU) integration tests for DDP and FSDP
    # with dynamo/inductor, compiler collectives, guard collectives, autotuning and asymmetric
    # compilation. Class attribute device_type is the current accelerator type or cpu. Slow;
    # use sparingly.
    """
    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )

    @skip_if_lt_x_gpu(2)
    @config.patch(optimize_ddp=False, enable_compiler_collectives=True)
    def test_ddp_baseline_aot_eager_multiprocess(self):
        # [FUNCTION DOC] Compiles a DDP-wrapped toy model with aot_eager and optimize_ddp off on
        # each rank and asserts the output equals the eager output.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            m = DDP(m, device_ids=[self.rank])
            m = torch.compile(m, backend="aot_eager")
            outputs = m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    def _test_hf_bert_ddp_inductor(self, static_graph):
        # [FUNCTION DOC] HELPER: Per rank, builds HF BERT wrapped in real DDP (optionally
        # static_graph) and checks inductor compile against eager via run_hf_bert_ddp.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model, static_graph=static_graph)
            run_hf_bert_ddp(self, model, inputs, "inductor")

    @_skipIf_transformers_ge_5_2
    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor(self):
        # [FUNCTION DOC] Runs the BERT DDP inductor comparison without static graph; skipped on
        # transformers >= 5.2 and without a GPU.
        self._test_hf_bert_ddp_inductor(static_graph=False)

    @_skipIf_transformers_ge_5_2
    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    @patch.object(torch._inductor.config, "fallback_random", True)
    def test_hf_bert_ddp_inductor_static_graph(self):
        # [FUNCTION DOC] Runs the BERT DDP inductor comparison with static_graph enabled.
        self._test_hf_bert_ddp_inductor(static_graph=True)

    def _test_hf_bert_aot_eager(self, static_graph):
        # [FUNCTION DOC] HELPER: Same as the inductor helper but with the aot_eager backend.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_hf_bert(self.rank)
            model = DDP(model, static_graph=static_graph)
            run_hf_bert_ddp(self, model, inputs, "aot_eager")

    @_skipIf_transformers_ge_5_2
    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    def test_hf_bert_ddp_aot_eager(self):
        # [FUNCTION DOC] Runs the BERT DDP aot_eager comparison without static graph.
        self._test_hf_bert_aot_eager(static_graph=False)

    @_skipIf_transformers_ge_5_2
    @skip_if_lt_x_gpu(2)
    @import_transformers_or_skip()
    @config.patch(optimize_ddp=True, enable_compiler_collectives=True)
    def test_hf_bert_ddp_aot_eager_static_graph(self):
        # [FUNCTION DOC] Runs the BERT DDP aot_eager comparison with static_graph enabled.
        self._test_hf_bert_aot_eager(static_graph=True)

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(optimize_ddp=False, enable_compiler_collectives=True)
    def test_ddp_activation_checkpointing(self):
        # [FUNCTION DOC] With optimize_ddp off, applies non-reentrant activation checkpointing
        # to every Linear of a three-layer model, wraps it in DDP, and asserts the compiled
        # output equals the eager output.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            apply_activation_checkpointing,
            checkpoint_wrapper,
            CheckpointImpl,
        )

        class MyModel(torch.nn.Module):
            # [CLASS DOC] HELPER: Three Linear layers (64->32->16->8) applied in sequence.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates fc1, fc2, fc3.
                super().__init__()
                self.fc1 = torch.nn.Linear(64, 32)
                self.fc2 = torch.nn.Linear(32, 16)
                self.fc3 = torch.nn.Linear(16, 8)

            def forward(self, inp):
                # [FUNCTION DOC] HELPER: Applies fc1, fc2, fc3 in order.
                return self.fc3(self.fc2(self.fc1(inp)))

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            self.assertFalse(config.optimize_ddp)
            model = MyModel().to(device=self.device_type)

            # Activation checkpointing for Linear layers.
            non_reentrant_wrapper = functools.partial(
                checkpoint_wrapper,
                checkpoint_impl=CheckpointImpl.NO_REENTRANT,
            )
            check_fn = lambda submodule: isinstance(  # noqa: E731
                submodule, torch.nn.Linear
            )
            apply_activation_checkpointing(
                model, checkpoint_wrapper_fn=non_reentrant_wrapper, check_fn=check_fn
            )

            model = DDP(model)
            x = torch.randn(10, 64).to(self.device_type)
            correct_outputs = model(x)

            opt_model = torch.compile(model)
            outputs = opt_model(x)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_aot_eager(self):
        # [FUNCTION DOC] Compiles an FSDP model (original params) with aot_eager, first wrapped
        # at the root and then with nested FSDP around each Linear, asserting outputs match
        # eager in both cases.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="aot_eager")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

            # Test with recursive wrapping, nested FSDP around each Linear
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            fsdp_m = torch.compile(fsdp_m, backend="aot_eager")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @skip_if_lt_x_gpu(2)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @requires_cuda_and_triton
    def test_ddp_optimizer_cudagraph(self):
        # [FUNCTION DOC] Trains a conv/batchnorm network under DDP compiled in reduce-overhead
        # mode for 10 iterations and asserts the cudagraph tree manager's next graph id is 4,
        # i.e. two forward and two backward graphs were recorded (the large channel count
        # triggers a DDPOptimizer split).
        class Net(nn.Module):
            # [CLASS DOC] HELPER: Conv(46->640, 3x3), batch norm, Conv(640->640, 1x1) network.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates the two convolutions and the batch norm layer.
                super().__init__()
                # need a large channel to trigger ddp optimizer split module
                self.CHANNELS = 640
                self.convi = nn.Conv2d(46, self.CHANNELS, 3, padding=1, bias=False)
                self.convp = nn.Conv2d(
                    self.CHANNELS, self.CHANNELS, 1, padding=0, bias=False
                )
                self.bni = nn.BatchNorm2d(self.CHANNELS)

            def forward(self, bitmap_channels):
                # [FUNCTION DOC] HELPER: Applies convi, bni, convp in order.
                x = self.convi(bitmap_channels)
                x = self.bni(x)
                x = self.convp(x)
                return x

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            net = Net().to(self.rank)
            optimizer = torch.optim.SGD(
                net.parameters(),
                lr=5e-2,
            )

            net = DDP(net, device_ids=[self.rank])
            opt_net = torch.compile(net, mode="reduce-overhead")
            opt_net.train()

            for _ in range(10):
                optimizer.zero_grad()
                data = torch.randn((16, 46, 8, 8), dtype=torch.float32, device="cuda")
                opt_net(data).sum().backward()

            # 2 fwd and 2 bwd graph such that 4 graphs in total
            graph_id = (
                torch._inductor.cudagraph_trees.get_container(self.rank)
                .tree_manager.new_graph_id()
                .id
            )
            self.assertTrue(graph_id == 4)

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_setattr(self):
        # [FUNCTION DOC] Compiles an FSDP-wrapped module that sets an attribute in forward using
        # the eager backend, and asserts there is exactly one graph break whose reason mentions
        # setattr on Tensor.requires_grad.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            from torch._dynamo.utils import counters

            counters.clear()
            m, inputs, correct_outputs = get_mutating_model(
                f"{self.device_type}:{self.rank}"
            )
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="eager", fullgraph=False)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))
            self.assertEqual(len(counters["graph_break"]), 1)
            first_graph_break = list(counters["graph_break"].keys())[0]  # noqa: RUF015
            self.assertIn("setattr() on Tensor.requires_grad", first_graph_break)

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    def test_fsdp_unspecialized_forced_getattr_inline(self):
        # [FUNCTION DOC] Compiles an FSDP-wrapped module that has a Linear stored only in its
        # __dict__ and asserts the output matches eager (counters are cleared but not checked).
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            from torch._dynamo.utils import counters

            counters.clear()
            m, inputs, correct_outputs = get_forced_getattr_module(
                f"{self.device_type}:{self.rank}"
            )
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="eager", fullgraph=False)
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @skipIfRocmArch(MI350_ARCH)  # regression in ROCm 7.2
    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_inductor(self):
        # [FUNCTION DOC] Same as the aot_eager FSDP test but with the inductor backend, for
        # root-level and per-Linear nested wrapping.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            # Test with basic FSDP wrapping (outer wrap around whole model)
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(m, use_orig_params=True)
            fsdp_m = torch.compile(fsdp_m, backend="inductor")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

            # Test with recursive wrapping, nested FSDP around each Linear
            m, inputs, correct_outputs = get_model(f"{self.device_type}:{self.rank}")
            fsdp_m = FSDP(
                m,
                auto_wrap_policy=functools.partial(
                    transformer_auto_wrap_policy, transformer_layer_cls=(nn.Linear,)
                ),
                use_orig_params=True,
            )
            fsdp_m = torch.compile(fsdp_m, backend="inductor")
            outputs = fsdp_m(inputs)
            self.assertTrue(same(correct_outputs, outputs))

    @config.patch(enable_compiler_collectives=True)
    @skip_if_lt_x_gpu(1)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_fsdp_activation_checkpointing(self):
        # [FUNCTION DOC] Wraps the nested toy model with FSDP and activation checkpointing,
        # compiles with a counting inductor backend, and asserts equal outputs, two frames (one
        # per FSDP module) and a tag_activation_checkpoint node in the first graph.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            model, inputs = get_toy_model_for_activation_checkpointing(
                f"{self.device_type}:{self.rank}"
            )
            is_inner = lambda module: isinstance(module, ToyInnerModel)  # noqa: E731
            wrap_policy = functools.partial(lambda_auto_wrap_policy, lambda_fn=is_inner)
            model = apply_fsdp_with_checkpointing(model, wrap_policy, is_inner)
            correct_outputs = model(inputs)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("inductor")
            opt_model = torch.compile(model, backend=cnt)
            outputs = opt_model(inputs)
            self.assertTrue(same(correct_outputs, outputs))
            # Each FSDP module is a separate graph
            self.assertEqual(cnt.frame_count, 2)
            self.assertTrue(
                find_first_node(cnt.graphs[0], tag_activation_checkpoint) is not None
            )

    @_expectedFailureIf_transformers_ge_5_2
    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @config.patch(enable_compiler_collectives=True)
    @unittest.skipIf(
        PLATFORM_SUPPORTS_FLASH_ATTENTION or PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "Inaccurate results with fused SDPA kernels",
    )
    def test_hf_bert_fsdp(self):
        # [FUNCTION DOC] Compares eager and inductor-compiled FSDP-wrapped BERT (no recursive
        # wrapping) for outputs, loss and collected gradients with the same seed. Expected
        # failure on transformers >= 5.2; skipped when fused SDPA kernels are available.
        def apply_fsdp(model, wrap_policy):
            # [FUNCTION DOC] HELPER: Returns an FSDP wrapper (original params) around a deep
            # copy of model.
            model = FSDP(
                copy.deepcopy(model), auto_wrap_policy=wrap_policy, use_orig_params=True
            )
            return model

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for wrap_policy, test_instance in (
                (None, "FSDP without recursive wrapping"),
            ):
                print(f"Running hf_bert test for {test_instance}")
                model, inputs = get_hf_bert(self.rank)
                reset_rng_state()
                eager_model = apply_fsdp(model, wrap_policy)
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                reset_rng_state()
                opt_model = apply_fsdp(model, wrap_policy)
                opt_model = torch.compile(opt_model, backend="inductor")
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                self.assertTrue(same(correct_results, opt_results))

    @_expectedFailureIf_transformers_ge_5_2
    @import_transformers_or_skip()
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    # TODO(whc) Investigate why cudagraphs breaks inductor+fsdp for hf_bert
    @patch.object(torch._inductor.config.triton, "cudagraphs", False)
    @patch.object(torch._inductor.config, "fallback_random", True)
    @config.patch(guard_nn_modules=True, enable_compiler_collectives=True)
    def test_hf_bert_fsdp_activation_checkpointing(self):
        # [FUNCTION DOC] Same comparison with FSDP wrapping each BertLayer plus non-reentrant
        # activation checkpointing on BertLayer, with guard_nn_modules enabled.
        from transformers.models.bert.modeling_bert import BertLayer

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            for wrap_policy, test_instance in (
                (
                    functools.partial(
                        transformer_auto_wrap_policy, transformer_layer_cls=(BertLayer,)
                    ),
                    "FSDP with recursive wrapping BertLayer instances",
                ),
            ):
                print(
                    f"Running hf_bert_activation_checkpointing test for {test_instance}"
                )
                model, inputs = get_hf_bert(self.rank)
                check_fn = lambda submodule: isinstance(  # noqa: E731
                    submodule, BertLayer
                )
                reset_rng_state()
                eager_model = apply_fsdp_with_checkpointing(
                    model, wrap_policy, check_fn
                )
                correct_outputs = eager_model(**inputs)
                correct_loss = correct_outputs.loss
                correct_loss.backward()

                reset_rng_state()
                opt_model = apply_fsdp_with_checkpointing(model, wrap_policy, check_fn)
                opt_model = torch.compile(opt_model, backend="inductor")
                opt_outputs = opt_model(**inputs)
                opt_loss = opt_outputs.loss
                opt_loss.backward()

                inputs_flat = [inputs[k] for k in inputs]
                correct_results = collect_results(
                    eager_model, correct_outputs.logits, correct_loss, inputs_flat
                )
                opt_results = collect_results(
                    opt_model, opt_outputs.logits, opt_loss, inputs_flat
                )
                self.assertTrue(same(correct_results, opt_results))

    @skipIfTorchInductor(msg="https://github.com/pytorch/pytorch/issues/152944")
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_tensor(self):
        # [FUNCTION DOC] Trains a DDP model whose forward is compiled while ranks feed
        # batches of different sizes (rank 0: 5, 8, 6; others: 6, 6, 3) and asserts all ranks
        # report the same number of compilations (automatic dynamic decided collectively by
        # compiler collectives).
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):

            class SimpleModel(nn.Module):
                # [CLASS DOC] HELPER: Single Linear layer model.
                def __init__(self, input_size, output_size):
                    # [FUNCTION DOC] HELPER: Creates the linear layer.
                    super().__init__()
                    self.linear = nn.Linear(input_size, output_size)

                def forward(self, x):
                    # [FUNCTION DOC] HELPER: Applies the linear layer.
                    return self.linear(x)

            torch._dynamo.utils.clear_compilation_metrics()

            model = SimpleModel(10, 2).to(self.rank)
            model.forward = torch.compile(model.forward)
            ddp_model = DDP(model, device_ids=[self.rank])

            loss_fn = nn.CrossEntropyLoss()
            optimizer = optim.SGD(ddp_model.parameters(), lr=0.001)

            def B(s):
                # [FUNCTION DOC] HELPER: Builds a batch: random (s, 10) features and random
                # binary labels of size s.
                return [torch.randn(s, 10), torch.randint(0, 2, (s,))]

            if self.rank == 0:
                dataloader = [B(5), B(8), B(6)]
            else:
                dataloader = [B(6), B(6), B(3)]

            for data, labels in dataloader:
                data, labels = data.to(self.rank), labels.to(self.rank)
                optimizer.zero_grad()
                output = ddp_model(data)
                loss = loss_fn(output, labels)
                loss.backward()
                optimizer.step()

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_scalar(self):
        # [FUNCTION DOC] Compiles a function taking a Python int argument that differs across
        # ranks (3,3,7 vs 3,4,9) and asserts all ranks end with the same compilation count.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            # TODO: This should be possible to do inside the function, but
            device = f"{self.device_type}:{self.rank}"

            @torch.compile()
            def f(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus the sum of ones sized by y.
                return x + torch.ones(y, device=device).sum()

            if self.rank == 0:
                dataloader = [3, 3, 7]
            else:
                dataloader = [3, 4, 9]

            for data in dataloader:
                f(torch.randn(5, device=self.rank), data)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_automatic_dynamic_speculation_divergence(self):
        # [FUNCTION DOC] Compiles a function that reads shapes of two inputs where non-zero
        # ranks see size 3 on the first call and rank 0 sees 4; checks compile counts stay equal
        # across ranks despite divergent speculation.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x, y):
                # [FUNCTION DOC] HELPER: Reads both shapes and returns the sum of both
                # tensors' sums.
                zx = x.shape  # noqa: F841
                zy = y.shape  # noqa: F841
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [4, 4]
            else:
                dataloader = [3, 4]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_graph_break_empty_graph_still_collective(self):
        # [FUNCTION DOC] Same idea, but the function has a print graph break before reading
        # shapes, so the first graph is empty; asserts compile counts equal across ranks.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x, y):
                # [FUNCTION DOC] HELPER: Copies y, prints, reads shapes and returns the sum
                # of sums.
                z = y  # noqa: F841
                print("woof")
                zx = x.shape  # noqa: F841
                zy = y.shape  # noqa: F841
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [5, 5, 6]
            else:
                dataloader = [3, 4, 5]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_dim_mismatch(self):
        # [FUNCTION DOC] Ranks pass tensors of different dimensionality (rank 0: shape [4, 2],
        # others: [3]); asserts compile counts equal across ranks.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x, y):
                # [FUNCTION DOC] HELPER: Reads both shapes and returns the sum of sums.
                zx = x.shape  # noqa: F841
                zy = y.shape  # noqa: F841
                return x.sum() + y.sum()

            if self.rank == 0:
                dataloader = [[4, 2]]
            else:
                dataloader = [[3]]

            for data in dataloader:
                f(
                    torch.randn(data, device=self.rank),
                    torch.randn(data, device=self.rank),
                )

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_missing_source(self):
        # [FUNCTION DOC] Indexes a Python list of tensors by the rank argument (value with no
        # tensor source) and asserts equal compile counts across ranks.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(rank, xs):
                # [FUNCTION DOC] HELPER: Returns the sum of the element at index rank of xs.
                return xs[rank].sum()

            xs = []
            for _ in range(self.world_size):
                xs.append(torch.randn(10, device=self.rank))

            f(self.rank, xs)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_scalar_missing_source(self):
        # [FUNCTION DOC] Same as the previous test but the list holds Python ints, which are
        # turned into a tensor; asserts equal compile counts across ranks.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(rank, xs):
                # [FUNCTION DOC] HELPER: Returns a tensor built from xs at index rank on the
                # rank's device.
                return torch.tensor(xs[rank], device=self.rank)

            xs = []
            for i in range(self.world_size):
                xs.append(10 + i)

            f(self.rank, xs)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @config.patch(enable_compiler_collectives=True)
    def test_compiler_collectives_type_mismatch(self):
        # [FUNCTION DOC] Rank 0 passes a tensor and other ranks pass an int to a function
        # branching on type. Asserts each rank compiled at least once (counts may differ). The
        # reverse case is documented as deadlocking and is kept only inside an unused string.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x):
                # [FUNCTION DOC] HELPER: Returns a tensor from an int input, otherwise the sum
                # of the tensor.
                if isinstance(x, int):
                    return torch.tensor(x, device=self.rank)
                else:
                    return x.sum()

            if self.rank == 0:
                x = torch.randn(10, device=self.rank)
            else:
                x = 12
            f(x)

            # This deadlocks, I guess we don't support this
            """
            if self.rank == 0:
                x = torch.randn(12, device=self.rank)
            else:
                x = 10
            f(x)
            """

            # Ranks take different code paths (tensor vs int), so compilation counts
            # can differ. Check that every rank triggered compilation at least once.
            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res:
                self.assertGreaterEqual(r, 1, "each rank should compile at least once")

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @enable_guard_collectives()
    def test_guard_collective(self):
        # [FUNCTION DOC] With guard collectives enabled, rank 0 reuses a size while other ranks
        # use a new size (recompile on one rank); asserts equal compile counts across ranks.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x.sum().
                return x.sum()

            x = torch.randn(10, device=self.rank)
            f(x)

            if self.rank == 0:
                x = torch.randn(10, device=self.rank)
            else:
                x = torch.randn(12, device=self.rank)  # recompile on one rank
            f(x)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._dynamo.config, "enable_compiler_collectives", True)
    @patch.object(torch._inductor.config, "max_autotune_gemm", True)
    @patch.object(torch._inductor.config, "distributed_max_autotune_gemm", True)
    def test_multiproc_autotune(self):
        # [FUNCTION DOC] With distributed max-autotune GEMM enabled, compiles a function with
        # three bfloat16 matmuls of fixed shapes; logs and re-raises any exception, asserts
        # equal compile counts across ranks and prints the result.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(a, b, c):
                # [FUNCTION DOC] HELPER: Sum of (a @ b) + 1, sum of relu(b @ c), and sum of c
                # @ a.
                res = (
                    torch.sum((a @ b) + 1.0)
                    + torch.sum(torch.relu(b @ c))
                    + torch.sum(c @ a)
                )

                return res

            a = torch.randn(1024, 1024, device=self.rank, dtype=torch.bfloat16)
            b = torch.randn(1024, 2048, device=self.rank, dtype=torch.bfloat16)
            c = torch.randn(2048, 1024, device=self.rank, dtype=torch.bfloat16)

            try:
                f(a, b, c)
            except Exception:
                log.exception("Caught exception running f")
                raise

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

            print(f"Result from {self.rank} is {f(a, b, c)}")

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._dynamo.config, "enable_compiler_collectives", True)
    @patch.object(torch._inductor.config, "max_autotune_gemm", True)
    @patch.object(torch._inductor.config, "distributed_max_autotune_gemm", True)
    def test_multiproc_autotune_dynamic_shapes(self):
        # [FUNCTION DOC] Like the previous test but marks both dims of a, b, c dynamic; after
        # the first run, calls with smaller shapes and asserts no recompilation and equal counts
        # across ranks.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            @torch.compile()
            def f(a, b, c):
                # [FUNCTION DOC] HELPER: Sum of (a @ b) + 1, sum of relu(b @ c), and sum of c
                # @ a.
                res = (
                    torch.sum((a @ b) + 1.0)
                    + torch.sum(torch.relu(b @ c))
                    + torch.sum(c @ a)
                )

                return res

            a = torch.randn(1024, 1024, device=self.rank, dtype=torch.bfloat16)
            b = torch.randn(1024, 2048, device=self.rank, dtype=torch.bfloat16)
            c = torch.randn(2048, 1024, device=self.rank, dtype=torch.bfloat16)

            # Mark tensors as dynamic on dimension 0
            torch._dynamo.mark_dynamic(a, 0)
            torch._dynamo.mark_dynamic(a, 1)
            torch._dynamo.mark_dynamic(b, 0)
            torch._dynamo.mark_dynamic(b, 1)
            torch._dynamo.mark_dynamic(c, 0)
            torch._dynamo.mark_dynamic(c, 1)

            try:
                f(a, b, c)
            except Exception:
                log.exception("Caught exception running f")
                raise

            metrics = torch._dynamo.utils.get_compilation_metrics()
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

            print(f"Result from {self.rank} is {f(a, b, c)}")

            # Store the initial compilation count
            initial_compile_count = len(metrics)

            # # Test with different sizes to ensure dynamic shapes work without recompilation
            a2 = torch.randn(512, 512, device=self.rank, dtype=torch.bfloat16)
            b2 = torch.randn(512, 2048, device=self.rank, dtype=torch.bfloat16)
            c2 = torch.randn(2048, 512, device=self.rank, dtype=torch.bfloat16)

            try:
                result2 = f(a2, b2, c2)
                print(f"Result2 from {self.rank} is {result2}")
            except Exception:
                log.exception("Caught exception running f with different sizes")
                raise

            # Verify no recompilation occurred
            metrics_after = torch._dynamo.utils.get_compilation_metrics()
            final_compile_count = len(metrics_after)
            self.assertEqual(
                initial_compile_count,
                final_compile_count,
                "Expected no recompilation with dynamic shapes",
            )

            # Verify all ranks have the same compilation count
            res_after = [None] * self.world_size
            torch.distributed.all_gather_object(res_after, final_compile_count)
            for r in res_after[1:]:
                self.assertEqual(res_after[0], r)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_get_pg_attr(self):
        # [FUNCTION DOC] Fullgraph compiles a function branching on _rank_not_in_group(pg) with
        # pg a closure variable: with the default group the result is x - 1; after rebinding pg
        # to NON_GROUP_MEMBER the result is x + 1.
        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            pg = dist.distributed_c10d._get_default_group()

            device = f"{self.device_type}:{self.rank}"

            @torch.compile(fullgraph=True)
            def f(x):
                # [FUNCTION DOC] HELPER: Returns x + 1 if the rank is not in group pg, else x
                # - 1.
                if dist.distributed_c10d._rank_not_in_group(pg):
                    return x + 1
                else:
                    return x - 1

            x = torch.ones(4, device=device)
            self.assertEqual(f(x), x - 1)

            pg = dist.distributed_c10d.GroupMember.NON_GROUP_MEMBER
            self.assertEqual(f(x), x + 1)

    @skipIfXpu  # ProcessGroupXCCL doesn't support _set_default_timeout yet.
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "fx_graph_cache", False)
    @patch.object(torch._inductor.config, "fx_graph_remote_cache", False)
    def test_asymmetric_compilation(self):
        # [FUNCTION DOC] Rank 0 sleeps at compile time through comptime, so ranks compile at
        # different times. Sets a short backend timeout, adds an ephemeral timeout on non-zero
        # ranks, and asserts equal compile counts across ranks. Skipped on XPU.
        from torch._dynamo.comptime import comptime

        with _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            device = f"{self.device_type}:{self.rank}"

            pg = dist.distributed_c10d._get_default_group()

            cnt = torch._dynamo.testing.CompileCounter()
            sleep_time = 5

            @torch.compile(backend=cnt)
            def f(x):
                # [FUNCTION DOC] HELPER: Rank 0 sleeps at compile time; returns the sum of 2
                # * x.
                if self.rank == 0:
                    comptime.sleep(sleep_time)

                y = 2 * x
                return y.sum()

            backend = pg._get_backend(torch.device(device))
            backend._set_default_timeout(timedelta(seconds=sleep_time - 2))

            x = torch.ones(4, device=device)

            # NCCL startup is lazy
            w = pg.allreduce(x)
            w.wait()

            f(x)
            if self.rank != 0:
                # test fails with NCCL timeout without this line
                dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                    timedelta(seconds=sleep_time)
                )

            w = pg.allreduce(x)
            w.wait()
            torch.accelerator.synchronize(device)

            metrics = torch._dynamo.utils.get_compilation_metrics()
            # Number of compiles same on all nodes
            res = [None] * self.world_size
            torch.distributed.all_gather_object(res, len(metrics))
            for r in res[1:]:
                self.assertEqual(res[0], r)

    @skipIfXpu  # ProcessGroupXCCL doesn't support _set_default_timeout yet.
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(torch._inductor.config, "fx_graph_cache", True)
    @patch.object(torch._inductor.config, "fx_graph_remote_cache", False)
    @patch.object(torch._inductor.config, "sleep_sec_TESTING_ONLY", 10)
    def test_asymmetric_compilation_with_fx_cache(self):
        # [FUNCTION DOC] With fx graph cache enabled and an artificial inductor sleep, rank 0
        # recompiles in a fresh cache (expects 2 misses, 0 hits) while the others hit the cache
        # (1 miss, 1 hit); counters checked, with collectives afterward. Skipped on XPU.
        from torch._dynamo.utils import counters
        from torch._inductor.utils import fresh_cache

        with fresh_cache(), _dynamo_dist_per_rank_init(self.rank, self.world_size):
            torch._dynamo.utils.clear_compilation_metrics()

            device = f"{self.device_type}:{self.rank}"

            pg = dist.distributed_c10d._get_default_group()

            @torch.compile
            def f(x):
                # [FUNCTION DOC] HELPER: Returns the sum of 2 * x.
                y = 2 * x
                return y.sum()

            backend = pg._get_backend(torch.device(device))
            backend._set_default_timeout(timedelta(seconds=5))
            counters.clear()

            x = torch.ones(4, device=device)

            f(x)

            self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
            self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
            self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            w = pg.allreduce(x)
            w.wait()
            torch.accelerator.synchronize(device)
            torch._dynamo.reset()

            if self.rank == 0:
                with fresh_cache():
                    f(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 2)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 0)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)
            else:
                f(x)
                self.assertEqual(counters["inductor"]["fxgraph_cache_miss"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_hit"], 1)
                self.assertEqual(counters["inductor"]["fxgraph_cache_bypass"], 0)

            w = pg.allreduce(x)
            w.wait()
            torch.accelerator.synchronize(device)


@requires_accelerator_dist_backend(["nccl", "xccl"])
@unittest.skipUnless(torch.accelerator.is_available(), "Requires accelerator")
class TestSingleProc(DynamoDistributedSingleProcTestCase):
    # [CLASS DOC] Single-process (world size 1) dynamo tests on a real process group: DDP
    # baseline compilation, DDPOptimizer graph splitting and inductor config forwarding,
    # FSDP guard and dedup behavior, and collective ops returning None under compile. Needs
    # an accelerator.
    """
    Test harness initializes dist process group.

    Test simple things here since they are simpler to debug.
    Use TestMultiProc for things that really need to run on multiple nodes
    """

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )

    def get_model(
        self, bsz=20, in_feat=10, hidden_feat=5000, out_feat=5, ctx_manager=None
    ):
        # [FUNCTION DOC] HELPER: Builds a ToyModel on self.device, applies init_weights, and
        # runs the eager forward. Returns (model, random inputs of shape (bsz, in_feat),
        # eager outputs).
        m = ToyModel(
            in_feat=in_feat,
            hidden_feat=hidden_feat,
            out_feat=out_feat,
            ctx_manager=ctx_manager,
        ).to(self.device)
        m.apply(init_weights)
        inputs = torch.rand(bsz, in_feat).to(self.device)
        outputs = m(inputs)
        return m, inputs, outputs

    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_aot_eager(self):
        # [FUNCTION DOC] Wraps the toy model in DDP, compiles with aot_eager (DDP optimization
        # off), asserts the output matches eager.
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch.compile(ddp_m, backend="aot_eager")
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @skipIfRocmArch(MI350_ARCH)  # regression in ROCm 7.2
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    @patch.object(config, "optimize_ddp", False)
    def test_ddp_baseline_inductor(self):
        # [FUNCTION DOC] Same as the aot_eager baseline but with the inductor backend.
        from torch.nn.parallel import DistributedDataParallel as DDP

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids)
        ddp_m = torch.compile(ddp_m, backend="inductor")
        outputs = ddp_m(inputs)
        self.assertTrue(same(correct_outputs, outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split(self):
        # [FUNCTION DOC] With DDP optimization on and a 25MB bucket, counts compiler calls (3
        # expected) and checks dynamo explain yields 3 break reasons that mention DDPOptimizer.
        # KNOWN ISSUE: the triple-quoted text after the config check is not a docstring (it
        # follows a statement), so it is just a discarded string expression.
        if not config.optimize_ddp:
            raise AssertionError("Expected config.optimize_ddp to be True")
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """

        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        check_splits_compiler = CheckSplitsCompiler()

        @torch.compile(backend=check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            # [FUNCTION DOC] HELPER: Runs the DDP model.
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

        # ensure compatibility with dynamo explain

        explain_out = torch._dynamo.explain(ddp_m)(inputs)
        break_reasons = explain_out.break_reasons
        self.assertEqual(len(break_reasons), 3)
        self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))

    @patch.object(config, "optimize_ddp", True)
    def test_inductor_config_override_with_ddp(self):
        # [FUNCTION DOC] With reduce-overhead mode and a graph break giving two frames, checks
        # that a config override applies only to frames after the first: first compile records
        # cudagraphs per frame (True for both), then with a debug_inductor_config_override the
        # second frame sees False and the first True, by tracking config_patches passed to
        # compile_fx. Outputs must match eager.
        """
        Verifies that TORCH_COMPILE_OVERRIDE_INDUCTOR_CONFIGS works end-to-end
        when DDPOptimizer is active: config_patches should be forwarded through
        DDPOptimizer to compile_fx for each subgraph.

        Uses reduce-overhead mode (which enables cudagraphs=True). The model
        has a graph break producing 2 Dynamo frames. The override targets only
        the second frame, so frame 0 keeps cudagraphs=True while frame 1 gets
        cudagraphs=False.
        """
        from torch._dynamo.graph_id_filter import _create_inductor_config_router
        from torch._inductor import compile_fx as compile_fx_mod

        N = 5000

        class GraphBreakModel(nn.Module):
            # [CLASS DOC] HELPER: Two large linear layers separated by an explicit dynamo
            # graph break.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates two Linear(5000, 5000) layers.
                super().__init__()
                self.linear0 = nn.Linear(N, N)
                self.linear1 = nn.Linear(N, N)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear0, forces a graph break, then
                # applies linear1.
                x = self.linear0(x)
                torch._dynamo.graph_break()
                x = self.linear1(x)
                return x

        m = GraphBreakModel().to(self.device)
        m.apply(init_weights)
        inputs = torch.rand(20, N).to(self.device)
        correct_outputs = m(inputs)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        cudagraphs_by_frame: dict[int, bool] = {}
        original_compile_fx = compile_fx_mod.compile_fx

        def tracking_compile_fx(model_, inputs_, **kwargs):
            # [FUNCTION DOC] HELPER: Wrapper for compile_fx that records, once per frame
            # id, the triton.cudagraphs value from config_patches, then delegates to the
            # real compile_fx.
            compile_id = torch._guards.CompileContext.current_compile_id()
            fid = compile_id.frame_id
            if fid not in cudagraphs_by_frame:
                patches = kwargs.get("config_patches", {})
                cudagraphs_by_frame[fid] = patches.get("triton.cudagraphs")
            return original_compile_fx(model_, inputs_, **kwargs)

        # First compile: discover frame_ids without any override
        with patch.object(compile_fx_mod, "compile_fx", tracking_compile_fx):
            opt_fn = torch.compile(ddp_m, mode="reduce-overhead")
            opt_outputs = opt_fn(inputs)

        self.assertTrue(same(correct_outputs, opt_outputs))
        frame_ids = sorted(cudagraphs_by_frame.keys())
        self.assertEqual(len(frame_ids), 2)
        # Both frames should have cudagraphs=True from reduce-overhead
        for fid in frame_ids:
            self.assertTrue(cudagraphs_by_frame[fid])

        # Second compile: override only the second frame
        torch._dynamo.reset()
        _create_inductor_config_router.cache_clear()
        cudagraphs_by_frame.clear()

        override = f">{frame_ids[0]}:triton.cudagraphs=False"
        with (
            torch._dynamo.config.patch(debug_inductor_config_override=override),
            patch.object(compile_fx_mod, "compile_fx", tracking_compile_fx),
        ):
            opt_fn = torch.compile(ddp_m, mode="reduce-overhead")
            opt_outputs = opt_fn(inputs)

        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(len(cudagraphs_by_frame), 2)
        # First frame: no override, reduce-overhead keeps cudagraphs=True
        self.assertTrue(cudagraphs_by_frame[frame_ids[0]])
        # Second frame: override sets cudagraphs=False
        self.assertFalse(cudagraphs_by_frame[frame_ids[1]])

    @patch.object(config, "optimize_ddp", True)
    def test_graph_split_ctx_manager(self):
        # [FUNCTION DOC] For both a counting and a plain aot_eager backend, and for autocast
        # fp16, enable_grad and no_grad contexts inside the model, compiles a DDP model with a
        # 3.5MB bucket. Checks outputs, the 4 compiler calls and 4 explain break reasons, and
        # the effect of each context on the output.
        """
        Ensures that we get the right number of splits and that the respective
        context managers' effects are applied to the computation.
        """

        for get_compiler in [
            lambda: CheckSplitsCompiler(),
            lambda: None,
        ]:
            for ctx_manager, output_test in [
                (
                    lambda: torch.autocast(
                        torch.device(self.device).type, torch.float16
                    ),
                    lambda out: self.assertEqual(out.dtype, torch.float16),
                ),
                (torch.enable_grad, lambda out: self.assertTrue(out.requires_grad)),
                (torch.no_grad, lambda out: self.assertTrue(not out.requires_grad)),
            ]:
                m, inputs, correct_outputs = self.get_model(
                    out_feat=1000,
                    hidden_feat=1000,
                    in_feat=1000,
                    ctx_manager=ctx_manager,
                )
                # inp - 1000 * 1000 matrix of float32 (4 bytes) = 4MB
                # hidden - 1000 * 1000 matrix of float32 (4 bytes) = 4MB
                bucket_cap_mb = 3.5  # 4MB
                ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=bucket_cap_mb)

                compiler = get_compiler()

                @torch.compile(backend=compiler.compile_fn if compiler else "aot_eager")
                def opt_fn(inputs):
                    # [FUNCTION DOC] HELPER: Runs the DDP model.
                    return ddp_m(inputs)

                opt_outputs = opt_fn(inputs)
                self.assertTrue(same(correct_outputs, opt_outputs))
                if compiler:
                    self.assertEqual(compiler.compiler_called, 4)

                output_test(opt_outputs)

                # ensure compatibility with dynamo explain

                explain_out = torch._dynamo.explain(ddp_m)(inputs)
                break_reasons = explain_out.break_reasons
                self.assertEqual(len(break_reasons), 4)
                self.assertTrue(all("DDPOptimizer" in r.reason for r in break_reasons))

    @skipIfXpu  # XPU device doesn't support flex_attention yet.
    @patch.object(config, "optimize_ddp", True)
    def test_compiled_flex_attention_full_model_ddp(self):
        # [FUNCTION DOC] Compiles a model that calls flex_attention with an alibi score_mod,
        # then wraps the compiled model in DDP; runs once on a (16, 512, 768) input and
        # synchronizes. Skipped on XPU.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Projection of query/key/value followed by flex_attention with
            # an alibi bias.
            def __init__(self, S, H, D):
                # [FUNCTION DOC] HELPER: Stores sizes, registers the alibi bias buffer, uses
                # flex_attention, and creates the projections.
                super().__init__()

                self.S = S
                self.H = H
                self.D = D

                alibi_bias = self.generate_alibi_bias(H)
                self.register_buffer("alibi_bias", alibi_bias, persistent=True)
                self.attention = flex_attention

                self.project_qk = torch.nn.Linear(H * D, H * D * 2)
                self.project_v = torch.nn.Linear(H * D, H * D)

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: Projects to q, k, v, reshapes to (batch, heads, seq,
                # dim) and calls attention with score_mod.
                batch_size, _, _ = hidden_states.size()

                query, key = self.project_qk(hidden_states).chunk(2, dim=2)
                query = query.view(self.S, batch_size, self.H, self.D)
                query = query.permute(1, 2, 0, 3)

                key = key.view(self.S, batch_size, self.H, self.D)
                key = key.permute(1, 2, 0, 3)

                value = self.project_v(hidden_states)
                value = value.view(self.S, batch_size, self.H, self.D)
                value = value.permute(1, 2, 0, 3)

                return self.attention(query, key, value, score_mod=self.alibi_score_mod)

            def generate_alibi_bias(self, num_heads):
                # [FUNCTION DOC] HELPER: Returns a tensor of negative per-head
                # slopes -(i+1)*8/num_heads.
                alibi_bias = [-((i + 1) * 8.0) / num_heads for i in range(num_heads)]
                return torch.tensor(alibi_bias)

            def alibi_score_mod(self, score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: Score modifier adding (q_idx - kv_idx) times the head's
                # alibi bias.
                bias = (q_idx - kv_idx) * self.alibi_bias[h]
                return score + bias

        B = 16
        H = 12
        S = 512
        D = 64

        model = Model(S, H, D)
        model.to(self.device_type)
        model = torch.compile(model)
        model = DDP(model, device_ids=self.device_ids)

        hidden_states = torch.randn(B, S, H * D).to(self.device_type)
        model(hidden_states)
        torch.accelerator.synchronize()

    @skipIfXpu  # XPU device doesn't support flex_attention yet.
    @patch.object(config, "optimize_ddp", True)
    def test_compiled_flex_attention_local_ddp(self):
        # [FUNCTION DOC] Same as the full-model test except only the attention call is wrapped
        # with torch.compile inside the module. Skipped on XPU.
        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: Same model as the full-model test with a locally
            # compiled flex_attention.
            def __init__(self, S, H, D):
                # [FUNCTION DOC] HELPER: Stores sizes, registers the alibi bias buffer, compiles
                # flex_attention and creates the projections.
                super().__init__()

                self.S = S
                self.H = H
                self.D = D

                alibi_bias = self.generate_alibi_bias(H)
                self.register_buffer("alibi_bias", alibi_bias, persistent=True)
                self.attention = torch.compile(flex_attention)

                self.project_qk = torch.nn.Linear(H * D, H * D * 2)
                self.project_v = torch.nn.Linear(H * D, H * D)

            def forward(self, hidden_states):
                # [FUNCTION DOC] HELPER: Projects to q, k, v, reshapes and calls attention
                # with score_mod.
                batch_size, _, _ = hidden_states.size()

                query, key = self.project_qk(hidden_states).chunk(2, dim=2)
                query = query.view(self.S, batch_size, self.H, self.D)
                query = query.permute(1, 2, 0, 3)

                key = key.view(self.S, batch_size, self.H, self.D)
                key = key.permute(1, 2, 0, 3)

                value = self.project_v(hidden_states)
                value = value.view(self.S, batch_size, self.H, self.D)
                value = value.permute(1, 2, 0, 3)

                return self.attention(query, key, value, score_mod=self.alibi_score_mod)

            def generate_alibi_bias(self, num_heads):
                # [FUNCTION DOC] HELPER: Returns per-head negative slopes -(i+1)*8/num_heads.
                alibi_bias = [-((i + 1) * 8.0) / num_heads for i in range(num_heads)]
                return torch.tensor(alibi_bias)

            def alibi_score_mod(self, score, b, h, q_idx, kv_idx):
                # [FUNCTION DOC] HELPER: Score modifier adding (q_idx - kv_idx) times the head's
                # alibi bias.
                bias = (q_idx - kv_idx) * self.alibi_bias[h]
                return score + bias

        B = 16
        H = 12
        S = 512
        D = 64

        model = Model(S, H, D)
        model.to(self.device_type)
        model = torch.compile(model)
        model = DDP(model, device_ids=self.device_ids)

        hidden_states = torch.randn(B, S, H * D).to(self.device_type)
        model(hidden_states)
        torch.accelerator.synchronize()

    @skipIfRocmArch(MI350_ARCH)  # regression in ROCm 7.2
    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor(self):
        # [FUNCTION DOC] Same as the graph split test but compiled with inductor; asserts the
        # output matches eager.
        # KNOWN ISSUE: the triple-quoted text after the config check is not a docstring.
        if not config.optimize_ddp:
            raise AssertionError("Expected config.optimize_ddp to be True")
        """
        Same as above, but using inductor backend.
        We observed issues with inductor/fx interface in the past.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch.compile(backend="inductor")
        def opt_fn(inputs):
            # [FUNCTION DOC] HELPER: Runs the DDP model.
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))

    @torch._inductor.config.patch(
        {"layout_optimization": True, "keep_output_stride": False}
    )
    @patch.object(config, "optimize_ddp", True)
    def _test_graph_split_inductor_layout_optimizations_impl(self, context):
        # [FUNCTION DOC] HELPER: Under layout_optimization and optimize_ddp, builds a model
        # of four 1x1 convs with 512 channels, wraps in DDP (25MB bucket), compiles with
        # inductor inside the context produced by the context argument and asserts the output
        # matches eager.
        if not config.optimize_ddp:
            raise AssertionError("Expected config.optimize_ddp to be True")
        channel_dim = 512
        # channel dim must be > 64 for inductor to do layout optimization and use NHWC

        class ToyModelConv(nn.Module):
        # [CLASS DOC] HELPER: Sequential of four 1x1 Conv2d+ReLU blocks with
        # channel_dim channels.
            def __init__(self) -> None:
            # [FUNCTION DOC] HELPER: Builds the sequential of convolutions and ReLUs.
                super().__init__()
                self.net = nn.Sequential(
                    *[
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                    + [
                        nn.Conv2d(channel_dim, channel_dim, 1, stride=1, bias=False),
                        nn.ReLU(),
                    ]
                )

            def forward(self, inputs):
            # [FUNCTION DOC] HELPER: Applies the sequential net.
                return self.net(inputs)

        def get_model():
        # [FUNCTION DOC] HELPER: Builds the conv model and a (2, 512, 512, 128) input; returns
        # model, inputs, eager outputs.
            m = ToyModelConv().to(self.device)
            m.apply(init_weights)
            inputs = torch.rand(2, channel_dim, channel_dim, 128).to(self.device)
            outputs = m(inputs)
            return m, inputs, outputs

        with context():
            m, inputs, correct_outputs = get_model()
            ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

            @torch.compile(backend="inductor")
            def opt_fn(inputs):
                # [FUNCTION DOC] HELPER: Runs the DDP model.
                return ddp_m(inputs)

            opt_outputs = opt_fn(inputs)
            self.assertTrue(same(correct_outputs, opt_outputs))

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_layout_optimizations_training(self):
        # [FUNCTION DOC] Runs the layout optimization check with a null context (training,
        # grad enabled).
        self._test_graph_split_inductor_layout_optimizations_impl(
            contextlib.nullcontext
        )

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_layout_optimizations_inference(self):
        # [FUNCTION DOC] Runs the layout optimization check inside no_grad (inference).
        self._test_graph_split_inductor_layout_optimizations_impl(torch.no_grad)

    @patch.object(config, "optimize_ddp", True)
    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_graph_split_inductor_transpose(self):
        # [FUNCTION DOC] Wraps an inductor-compiled model with transpose and flatten in DDP and
        # checks outputs for three batch sizes (100, 200, 300) against eager, covering dynamic
        # batch recompiles.
        if not config.optimize_ddp:
            raise AssertionError("Expected config.optimize_ddp to be True")

        B = 100
        N = 30
        D = 50
        K = 70

        class Foo(nn.Module):
            # [CLASS DOC] HELPER: Linear on the transposed input, flatten, then Linear to 2048.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates linear0 (N->K) and linear1 (D*K->2048).
                super().__init__()
                self.linear0 = nn.Linear(N, K)
                self.linear1 = torch.nn.Linear(D * K, 2048)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Transposes dims 1 and 2, applies linear0, flattens and
                # applies linear1.
                xt = x.transpose(2, 1)
                xt = self.linear0(xt).flatten(1)
                return self.linear1(xt)

        mod = Foo().to(self.device)

        compiled_mod = torch.compile(mod, backend="inductor")
        ddp_compiled_mod = DDP(compiled_mod, device_ids=self.device_ids)

        x = torch.randn((B, N, D), dtype=torch.float32, device=self.device)
        self.assertTrue(same(mod(x), ddp_compiled_mod(x)))

        x_1 = torch.randn((B * 2, N, D), dtype=torch.float32, device=self.device)
        self.assertTrue(same(mod(x_1), ddp_compiled_mod(x_1)))

        x_2 = torch.randn((B * 3, N, D), dtype=torch.float32, device=self.device)
        self.assertTrue(same(mod(x_2), ddp_compiled_mod(x_2)))

    @patch.object(config, "optimize_ddp", True)
    def test_no_split(self):
        # [FUNCTION DOC] With a tiny model and a 250MB bucket, asserts the DDPOptimizer produces
        # no split: the backend compiler is called once and outputs match eager.
        """
        Ensures the DDPOptimizer returns a correct, compiled module without
        introducing graph splits. (Based on model parameters fitting in the bucket)
        """
        # DDP will always do a 'first bucket' with a really small size;  so only a tiny model will escape this
        m, inputs, correct_outputs = self.get_model(hidden_feat=5)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=250)
        check_splits_compiler = CheckSplitsCompiler()

        @torch.compile(backend=check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            # [FUNCTION DOC] HELPER: Runs the DDP model.
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 1)

    @patch.object(config, "optimize_ddp", True)
    def test_aot_autograd(self):
        # [FUNCTION DOC] Compiles with aot_eager and runs forward and backward, checking that
        # AOT autograd backends work with graph splits (example inputs propagated).
        """
        Explicitly check AotAutograd family of compilers work,
        since they require example inputs propagated between graph splits.
        """
        m, inputs, correct_outputs = self.get_model()
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)

        @torch.compile(backend="aot_eager")
        def opt_fn(inputs):
            # [FUNCTION DOC] HELPER: Runs the DDP model.
            return ddp_m(inputs)

        opt_outputs = opt_fn(inputs)
        opt_outputs.sum().backward()
        self.assertTrue(same(correct_outputs, opt_outputs))

    @patch.object(config, "optimize_ddp", True)
    def test_custom_layer(self):
        # [FUNCTION DOC] Uses the custom model with a 1MB bucket and asserts 3 compiler calls
        # and matching outputs.
        """
        Just ensures that the appropriate number of splits happen (based on
        bucket size and model parameters) - verifies the number of times
        the user-provided compiler is called by the DDPOptimizer which is
        doing the graph splitting
        """
        m, inputs, correct_outputs = get_custom_model(self.device)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=1)

        check_splits_compiler = CheckSplitsCompiler()

        @torch.compile(backend=check_splits_compiler.compile_fn)
        def opt_fn(inputs):
            # [FUNCTION DOC] HELPER: Runs the DDP model on the unpacked inputs tuple.
            return ddp_m(*inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 3)

    @unittest.skipIf(not HAS_GPU, "Inductor+gpu needs triton and recent GPU arch")
    def test_empty_graph_inductor(self):
        # [FUNCTION DOC] Compiles a function with no tensor ops that returns the world size with
        # inductor. Swallows exceptions, then asserts the result is 1. Skipped without a GPU.
        def fn():
            # [FUNCTION DOC] HELPER: Returns a tuple with the distributed world size.
            get_world_size = torch.distributed.distributed_c10d.get_world_size()
            return (get_world_size,)

        opt_fn = torch.compile(fn, backend="inductor")
        res = None
        try:
            res = opt_fn()[0]
        except Exception:
            pass
        self.assertEqual(res, 1)

    @patch.object(config, "optimize_ddp", False)
    def test_ignored_parameters(self):
        # [FUNCTION DOC] Marks two parameters as ignored in DDP, builds a DDPOptimizer manually,
        # compiles with it, and asserts 2 compiler calls, matching outputs and that no bucket
        # contains an ignored parameter id.
        """
        Verifies ddp graph-split logic ignores parameters marked to ignore on DDP module.
        Hooks up graph-split optimizer manually so it can peek at internal state.
        """
        m, inputs, correct_outputs = get_custom_model(self.device)
        parameters_to_ignore = ["seq.2.weight", "seq.4.linear.bias"]
        DDP._set_params_and_buffers_to_ignore_for_model(m, parameters_to_ignore)
        ddp_m = DDP(m, device_ids=self.device_ids, bucket_cap_mb=25)
        parameter_ids_to_ignore = [
            id(ddp_m.module.get_parameter(p)) for p in ddp_m.parameters_to_ignore
        ]

        check_splits_compiler = CheckSplitsCompiler()
        ddp_optimizer = DDPOptimizer(
            bucket_bytes_cap=ddp_m.bucket_bytes_cap,
            backend_compile_fn=check_splits_compiler.compile_fn,
        )

        @torch.compile(backend=ddp_optimizer.compile_fn)
        def opt_fn(inputs):
            # [FUNCTION DOC] HELPER: Runs the DDP model on the unpacked inputs tuple.
            return ddp_m(*inputs)

        opt_outputs = opt_fn(inputs)
        self.assertTrue(same(correct_outputs, opt_outputs))
        self.assertEqual(check_splits_compiler.compiler_called, 2)
        for b in ddp_optimizer.buckets:
            for p_id in b.param_ids:
                self.assertFalse(p_id in parameter_ids_to_ignore)

    @patch.object(config, "optimize_ddp", True)
    def test_higher_order_op(self):
        # [FUNCTION DOC] Compiles a DDP model whose forward uses non-reentrant checkpoint on two
        # inner modules with aot_eager under a compile counter; checks only that tracing does
        # not fail.
        from torch.utils.checkpoint import checkpoint

        N = 1000

        class InnerModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Two chained linear layers.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates two Linear(1000, 1000) layers.
                super().__init__()
                self.linear1 = torch.nn.Linear(N, N)
                self.linear2 = torch.nn.Linear(N, N)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies linear1 then linear2.
                a = self.linear1(x)
                a = self.linear2(a)
                return a

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Two InnerModule instances, each checkpointed and followed
            # by cos.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates inner_mod1 and inner_mod2.
                super().__init__()
                self.inner_mod1 = InnerModule()
                self.inner_mod2 = InnerModule()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Checkpoints inner_mod1, applies cos, checkpoints
                # inner_mod2, applies cos.
                a = checkpoint(self.inner_mod1, x, use_reentrant=False)
                a = torch.cos(a)
                a = checkpoint(self.inner_mod2, a, use_reentrant=False)
                a = torch.cos(a)
                return a

        mod = MockModule().to(self.device_type)
        mod = DDP(mod, bucket_cap_mb=1)
        x = torch.randn(N, N, device=self.device_type, requires_grad=True)
        args = (x,)

        backend = "aot_eager"
        cnt = torch._dynamo.testing.CompileCounterWithBackend(backend)

        torch.compile(mod, backend=cnt)(*args)

    @patch.object(config, "optimize_ddp", True)
    def test_selective_activation_checkpoint(self):
        # [FUNCTION DOC] Compiles a DDP model using selective activation checkpoint contexts
        # with a policy that always prefers recompute, runs forward and backward, and
        # asserts the policy function was called (DDPOptimizer must keep the checkpoint
        # context metadata).
        """DDPOptimizer must not clobber SAC's _checkpoint_context_fn metadata."""
        from torch.utils.checkpoint import (
            checkpoint,
            CheckpointPolicy,
            create_selective_checkpoint_contexts,
        )

        N = 1000
        policy_call_count = 0

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Counts calls in the enclosing variable and
            # returns PREFER_RECOMPUTE.
            nonlocal policy_call_count
            policy_call_count += 1
            return CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = functools.partial(create_selective_checkpoint_contexts, policy_fn)

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: Two Linear layers each called through checkpoint with the
            # selective context_fn.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear1 and linear2.
                super().__init__()
                self.linear1 = torch.nn.Linear(N, N)
                self.linear2 = torch.nn.Linear(N, N)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the two checkpointed linear layers in sequence.
                x = checkpoint(
                    self.linear1, x, use_reentrant=False, context_fn=context_fn
                )
                return checkpoint(
                    self.linear2, x, use_reentrant=False, context_fn=context_fn
                )

        mod = MockModule().to(self.device_type)
        mod = DDP(mod, bucket_cap_mb=1)
        x = torch.randn(N, N, device=self.device_type, requires_grad=True)

        compiled = torch.compile(mod, backend="aot_eager")
        compiled(x).sum().backward()
        self.assertGreater(policy_call_count, 0)

    def test_fsdp_orig_params_assert(self):
        # [FUNCTION DOC] Wraps the model in FSDP without original params and calls torch.compile
        # on it; passes if no exception is raised. The compiled module is never called.
        # Test with basic FSDP wrapping (outer wrap around whole model)
        m, inputs, _ = get_model(f"{self.device_type}:{self.rank}")
        fsdp_m = FSDP(m, use_orig_params=False)
        # Test is that this function call does not throw an exception.
        fsdp_m = torch.compile(fsdp_m)

    def test_fsdp_skip_guards(self):
        # [FUNCTION DOC] For skip_fsdp_guards true/false, compiles an FSDP model whose forward
        # uses a comptime hook to print guards into the module-global GUARDS_FILE, then checks
        # printed guards carry the guard source local_fsdp_module (skip) or
        # local_unspecialized_nn_module (no skip). Sets the global GUARDS_FILE and resets dynamo
        # each iteration.
        """
        It's currently difficult to test dynamo guards.  Most guards tests are indirect- modify something and
        observe that the guard in question failed. In this case, since the FSDP guards were already deemed
        useless and skipping them is expected to have no practical effect, it's pretty contrived to even try to
        make those guards fail.  Instead, we observe the 'guard source' printed by dynamo's comptime print_guards
        function.

        Note: comptime prints the guards before the time they get installed or not installed, so in both cases
        (skip or no skip) the same guards get printed.  The difference is that in the skip case, they show up
        with a special 'guard source' which will cause them to not be installed.  So all we check for is the expected
        guard source 'local_fsdp_module'.
        """
        global GUARDS_FILE
        GUARDS_FILE = StringIO()

        for skip_guards, expected_guard_source in (
            (True, "local_fsdp_module"),
            (False, "local_unspecialized_nn_module"),
        ):
            torch._dynamo.reset()

            class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Local MLP (shadows the module level ToyModel) whose forward
            # prints guards.
                def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
                # [FUNCTION DOC] HELPER: Builds the four-block Linear+ReLU net.
                    super().__init__()
                    self.net = nn.Sequential(
                        *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                        + [nn.Linear(hidden_feat, out_feat), nn.ReLU()]
                    )

                def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Runs the net and a comptime hook printing guards,
                # returns the output.
                    out = self.net(inputs)

                    @comptime
                    def _(ctx):
                        # [FUNCTION DOC] HELPER: Comptime callback that prints the current
                        # guards to GUARDS_FILE.
                        ctx.print_guards(file=GUARDS_FILE)

                    return out

            device = f"{self.device_type}:{self.rank}"
            m = ToyModel(
                in_feat=10,
                hidden_feat=5000,
                out_feat=5,
            ).to(device)
            inputs = torch.rand(20, 10).to(device)
            m.apply(init_weights)
            correct_outputs = m(inputs)
            fsdp_m = FSDP(m, use_orig_params=True)

            with torch._dynamo.config.patch(skip_fsdp_guards=skip_guards):
                opt_m = torch.compile(fsdp_m, backend="aot_eager")
                outputs = opt_m(inputs)

            # far from an exhaustive check of all the expected guards, just check a couple of them.
            FileCheck().check("""local "L['self']" TYPE_MATCH""").check(
                f"""{expected_guard_source} "L['self']._modules['net']" TYPE_MATCH"""
            ).check(
                f"""{expected_guard_source} "L['self']._modules['net']._modules['0']" TYPE_MATCH"""
            ).run(GUARDS_FILE.getvalue())

            self.assertTrue(same(correct_outputs, outputs))

    def test_fsdp_skip_register_attr_or_module(self):
        # [FUNCTION DOC] Compiles an FSDP model with a debug backend that asserts no get_attr
        # node in the FX graph has a name containing the parameter names of the first two layers
        # (FSDP modules are not registered as attributes) and that output equals eager.
        """
        ensure FSDP module is not registered as attributes
        in the fx graph
        see `not source.guard_source().is_fsdp_module()`
        before calling `register_attr_or_module`
        in variables/builder.py
        """

        class ToyModel(nn.Module):
            # [CLASS DOC] HELPER: Two Linear+ReLU blocks.
            def __init__(self, in_feat=10, hidden_feat=5000, out_feat=5):
                # [FUNCTION DOC] HELPER: Builds the net.
                super().__init__()
                self.net = nn.Sequential(
                    *[nn.Linear(in_feat, hidden_feat), nn.ReLU()]
                    + [nn.Linear(hidden_feat, hidden_feat), nn.ReLU()]
                )

            def forward(self, inputs):
                # [FUNCTION DOC] HELPER: Applies the net.
                out = self.net(inputs)
                return out

        torch._dynamo.reset()

        device = f"{self.device_type}:{self.rank}"
        m = ToyModel(
            in_feat=10,
            hidden_feat=5000,
            out_feat=5,
        ).to(device)
        inputs = torch.rand(20, 10).to(device)
        m.apply(init_weights)
        correct_outputs = m(inputs)
        fsdp_m = FSDP(m, use_orig_params=True)

        def debug_compiler(gm, _):
            # [FUNCTION DOC] HELPER: Backend that checks get_attr node names against the
            # forbidden parameter names and returns the graph module unchanged.
            for node in gm.graph.nodes:
                if node.op == "get_attr":
                    for name in [
                        "l__self___net_0_weight",
                        "l__self___net_0_bias",
                        "l__self___net_2_weight",
                        "l__self___net_2_bias",
                    ]:
                        self.assertFalse(
                            name in node.name,
                            lambda msg: f"{msg}\nFSDP module {name} should not be registered as attributes",
                        )
            return gm

        opt_m = torch.compile(fsdp_m, backend=debug_compiler)
        outputs = opt_m(inputs)

        self.assertTrue(same(correct_outputs, outputs))

    def test_fsdp_dup_tensors_same_source(self):
        # [FUNCTION DOC] Compiles an FSDP-wrapped module that uses a plain tensor and a buffer
        # twice each, and asserts output equals the unwrapped module.
        """
        Tests that FSDP-managed modules' parameters and buffers with the same
        source are de-duplicated, meaning that they are each only passed once
        as a graph input.
        """

        class DuplicateModule(nn.Module):
            # [CLASS DOC] HELPER: Holds a plain tensor and a Buffer on the accelerator and uses
            # each twice in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates random _param tensor and _buf buffer on the
                # accelerator device.
                super().__init__()
                device_type = (
                    acc.type
                    if (acc := torch.accelerator.current_accelerator())
                    else "cpu"
                )
                self._param = torch.randn((3,), device=device_type)
                self._buf = torch.nn.Buffer(
                    torch.randn((3,), requires_grad=False, device=device_type)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Adds the buffer twice to x and then the tensor twice,
                # in place.
                # Use `_param` and `_buf` each twice in this compiled forward
                # to exercise if they are de-duplicated by TorchDynamo
                z = x + self._buf + self._buf
                z += self._param + self._param
                return z

        model = DuplicateModule()
        fsdp_model = FSDP(copy.deepcopy(model), use_orig_params=True)
        fsdp_model = torch.compile(fsdp_model, backend="aot_eager")
        inp = torch.randn((2, 3), device=self.device_type)
        local_out = model(inp)
        fsdp_out = fsdp_model(inp)
        self.assertEqual(local_out, fsdp_out)

    @patch.object(config, "guard_nn_modules", True)
    def test_fsdp_dup_tensors_diff_source(self):
        # [FUNCTION DOC] With guard_nn_modules on, compiles an FSDP model sharing one buffer
        # through two paths and mutating it, runs 15 times and asserts only one frame was
        # compiled (no incorrect dedup guards).
        """
        Tests that FSDP-managed modules' parameters and buffers with different
        source do not result in incorrect AOTAutograd de-dup guards like
        ``a is b``, where ``a`` and ``b`` are certainly not the same. We check
        this by checking for per-invocation recompiles.
        """

        class BufModule(nn.Module):
            # [CLASS DOC] HELPER: Module holding one buffer added to its input.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a random buffer on the accelerator device.
                super().__init__()
                device_type = (
                    acc.type
                    if (acc := torch.accelerator.current_accelerator())
                    else "cpu"
                )
                self._buf = nn.Buffer(
                    torch.randn((3,), requires_grad=False, device=device_type)
                )

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Returns x plus the buffer.
                return x + self._buf

        class Model(nn.Module):
            # [CLASS DOC] HELPER: Model sharing a buffer between itself and a child module and
            # mutating it in forward.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: Creates a parameter, a child BufModule and aliases
                # its buffer.
                super().__init__()
                device_type = (
                    acc.type
                    if (acc := torch.accelerator.current_accelerator())
                    else "cpu"
                )
                self._param = nn.Parameter(torch.randn((1,), device=device_type))
                self._buf_module = BufModule()
                # Share the buffer, meaning same tensor but different source
                self._buf = self._buf_module._buf

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Doubles the shared buffer in place, adds it, runs the
                # child, adds the parameter.
                # Use the same buffer tensor twice in the compiled forward,
                # including a data mutation to trigger de-dup logic
                self._buf.mul_(2)
                z = x + self._buf
                z = self._buf_module(z)
                z += self._param
                return z

        fsdp_model = FSDP(Model(), use_orig_params=True)
        cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
        fsdp_model = torch.compile(fsdp_model, backend=cnt)
        inp = torch.randn((2, 3), device=self.device_type)
        for _ in range(15):
            fsdp_model(inp)
        # Check for no recompiles (if there were incorrect de-dup guards, then
        # the frame count would be equal to the number of forward calls)
        self.assertEqual(cnt.frame_count, 1)

    def test_fsdp_staticmethod(self):
        # [FUNCTION DOC] Compiles FSDP-wrapped modules calling a static method via class and via
        # self, with a compile counter; asserts one frame each and that outputs match the
        # unwrapped reference.
        """
        Tests that Dynamo compiles staticmethods for FSDP-managed modules
        correctly both when the staticmethod is invoked from the class and from
        the object itself.
        """

        class ModuleWithStaticMethod(nn.Module):
            # [CLASS DOC] HELPER: Module with a parameter and a static add method called through
            # class or instance.
            def __init__(self, use_self: bool):
                # [FUNCTION DOC] HELPER: Stores use_self, seeds RNG with 42 and creates a
                # deterministic parameter.
                super().__init__()
                self._use_self = use_self
                torch.manual_seed(42)  # force `_param` to be deterministic
                device_type = (
                    acc.type
                    if (acc := torch.accelerator.current_accelerator())
                    else "cpu"
                )
                self._param = nn.Parameter(torch.randn((3,), device=device_type))

            def forward(self, x: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Calls _add either through self or the class, then
                # doubles the result in place.
                if self._use_self:
                    z = self._add(x, self._param)
                else:
                    z = ModuleWithStaticMethod._add(x, self._param)
                z *= 2
                return z

            @staticmethod
            def _add(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
                # [FUNCTION DOC] HELPER: Static method returning x + y.
                return x + y

        model = ModuleWithStaticMethod(False)
        x = torch.randn((2, 3), device=self.device_type)
        ref_out = model(x)
        test_outs: list[torch.Tensor] = []

        for use_self in (False, True):
            model = ModuleWithStaticMethod(use_self)
            fsdp_model = FSDP(model, use_orig_params=True)
            cnt = torch._dynamo.testing.CompileCounterWithBackend("aot_eager")
            fsdp_model = torch.compile(fsdp_model, backend=cnt)
            test_outs.append(fsdp_model(x))
            # Check for no recompiles, which could happen if incorrectly
            # passing args to the staticmethod (e.g. doubly passing `self`)
            self.assertEqual(cnt.frame_count, 1)
        for test_out in test_outs:
            self.assertEqual(test_out, ref_out)

    def test_async_subclass_no_specialize(self):
        # [FUNCTION DOC] Compiles fullgraph with dynamic shapes a function over async-wrapped
        # tensors of size 10 and 12 and asserts only one frame is compiled.
        cnt = torch._dynamo.testing.CompileCounterWithBackend("eager")

        @torch.compile(backend=cnt, fullgraph=True, dynamic=True)
        def f(x):
            # [FUNCTION DOC] HELPER: Returns x + 1.
            return x + 1

        f(_maybe_wrap_tensor(torch.randn(10)))
        f(_maybe_wrap_tensor(torch.randn(12)))

        self.assertEqual(cnt.frame_count, 1)

    def test_compiled_all_reduce_returns_none(self):
        # [FUNCTION DOC] Compiled fullgraph function checks dist.all_reduce with async_op False
        # returns None, then does a matmul; asserts result equals eager matmul.
        def fn(x, w):
            # [FUNCTION DOC] HELPER: All-reduces x, checks the result is None and returns x @ w.
            result = dist.all_reduce(x, async_op=False)
            assert result is None  # noqa: S101
            return x @ w

        x = torch.randn(4, 4, device=self.device)
        w = torch.randn(4, 4, device=self.device)
        expected = x @ w
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(x, w)
        self.assertEqual(actual, expected)

    def test_compiled_all_gather_into_tensor_returns_none(self):
        # [FUNCTION DOC] Same check for dist.all_gather_single with async_op False.
        def fn(output, input, w):
            # [FUNCTION DOC] HELPER: Calls all_gather_single into output, checks it returns
            # None, returns output @ w.
            result = dist.all_gather_single(output, input, async_op=False)
            assert result is None  # noqa: S101
            return output @ w

        input = torch.randn(4, 4, device=self.device)
        output = torch.empty(4, 4, device=self.device)
        w = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(output, input, w)
        self.assertEqual(actual, input @ w)

    def test_compiled_reduce_scatter_tensor_returns_none(self):
        # [FUNCTION DOC] Same check for dist.reduce_scatter_single with async_op False.
        def fn(output, input, w):
            # [FUNCTION DOC] HELPER: Calls reduce_scatter_single, checks it returns None,
            # returns output @ w.
            result = dist.reduce_scatter_single(output, input, async_op=False)
            assert result is None  # noqa: S101
            return output @ w

        input = torch.randn(4, 4, device=self.device)
        output = torch.empty(4, 4, device=self.device)
        w = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(output, input, w)
        self.assertEqual(actual, input @ w)

    def test_compiled_all_to_all_single_returns_none(self):
        # [FUNCTION DOC] Same check for dist.all_to_all_single with async_op False.
        def fn(output, input, w):
            # [FUNCTION DOC] HELPER: Calls all_to_all_single, checks it returns None, returns
            # output @ w.
            result = dist.all_to_all_single(output, input, async_op=False)
            assert result is None  # noqa: S101
            return output @ w

        input = torch.randn(4, 4, device=self.device)
        output = torch.empty(4, 4, device=self.device)
        w = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(output, input, w)
        self.assertEqual(actual, input @ w)

    def test_compiled_all_gather_returns_none(self):
        # [FUNCTION DOC] Same check for dist.all_gather into a list with async_op False.
        def fn(tensor_list, tensor, w):
            # [FUNCTION DOC] HELPER: Calls all_gather into tensor_list, checks it returns None,
            # returns tensor_list[0] @ w.
            result = dist.all_gather(tensor_list, tensor, async_op=False)
            assert result is None  # noqa: S101
            return tensor_list[0] @ w

        tensor = torch.randn(4, 4, device=self.device)
        tensor_list = [torch.empty(4, 4, device=self.device)]
        w = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(tensor_list, tensor, w)
        self.assertEqual(actual, tensor @ w)

    def test_compiled_reduce_scatter_base_returns_none(self):
        # [FUNCTION DOC] Same check for dist._reduce_scatter_base with async_op False.
        def fn(output, input, w):
            # [FUNCTION DOC] HELPER: Calls _reduce_scatter_base, checks it returns None, returns
            # output @ w.
            result = dist._reduce_scatter_base(output, input, async_op=False)
            assert result is None  # noqa: S101
            return output @ w

        input = torch.randn(4, 4, device=self.device)
        output = torch.empty(4, 4, device=self.device)
        w = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(output, input, w)
        self.assertEqual(actual, input @ w)

    def test_compiled_all_gather_base_returns_none(self):
        # [FUNCTION DOC] Same check for dist._all_gather_base, run first on (4, 4) tensors
        # and then, with a freshly compiled function, on 1-D tensors; each result is compared
        # with eager.
        def fn(output, input, w):
            # [FUNCTION DOC] HELPER: Calls _all_gather_base, checks it returns None, returns
            # output @ w.
            result = dist._all_gather_base(output, input, async_op=False)
            assert result is None  # noqa: S101
            return output @ w

        input = torch.randn(4, 4, device=self.device)
        output = torch.empty(4, 4, device=self.device)
        w = torch.randn(4, 4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(output, input, w)
        self.assertEqual(actual, input @ w)

        input = torch.ones(4, device=self.device)
        output = torch.empty(4, device=self.device)
        w = torch.ones(4, device=self.device)
        compiled_fn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        actual = compiled_fn(output, input, w)
        self.assertEqual(actual, input @ w)


@unittest.skipIf(not dist.is_available(), "requires distributed")
class TestNNFunctionalCompile(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that torch.distributed.nn.functional collectives raise a RuntimeError
    # under fullgraph torch.compile, naming the op and suggesting the functional collectives
    # alternative. Uses a fake backend process group of world size 2 created per test.
    """Tests for torch.distributed.nn.functional under torch.compile."""

    def setUp(self):
        # [FUNCTION DOC] Calls the base setUp and initializes a fake-backend process group with
        # rank 0, world 2.
        super().setUp()
        dist.init_process_group(backend="fake", rank=0, world_size=2)

    def tearDown(self):
        # [FUNCTION DOC] Destroys the process group, then calls the base tearDown.
        dist.destroy_process_group()
        super().tearDown()

    def _assert_not_supported(self, fn, x, name, *, suggestion=None):
        # [FUNCTION DOC] HELPER: Builds the expected message for the given op name (escaped as a
        # regex and including the suggested replacement if provided) and asserts that fn(x)
        # raises RuntimeError matching it.
        expected = f"torch.distributed.nn.functional.{name} is not supported under torch.compile."
        if suggestion:
            expected = re.escape(expected + f" Use {suggestion} instead.")
        with self.assertRaisesRegex(RuntimeError, expected):
            fn(x)

    def test_nn_functional_all_reduce_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.all_reduce under fullgraph compile is
        # unsupported, suggesting the functional collectives all_reduce.
        from torch.distributed.nn.functional import all_reduce

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the nn.functional all_reduce with SUM.
            return all_reduce(x, op=dist.ReduceOp.SUM)

        self._assert_not_supported(
            fn,
            torch.randn(4, 4),
            "all_reduce",
            suggestion="torch.distributed._functional_collectives.all_reduce",
        )

    def test_nn_functional_all_gather_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.all_gather is unsupported; suggestion
        # is all_gather_single.
        from torch.distributed.nn.functional import all_gather

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls the nn.functional all_gather.
            return all_gather(x)

        self._assert_not_supported(
            fn,
            torch.randn(4, 4),
            "all_gather",
            suggestion="torch.distributed._functional_collectives.all_gather_single",
        )

    def test_nn_functional_reduce_scatter_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.reduce_scatter is unsupported; suggestion
        # is reduce_scatter_single.
        from torch.distributed.nn.functional import reduce_scatter

        @torch.compile(fullgraph=True)
        def fn(input_list):
            # [FUNCTION DOC] HELPER: Calls reduce_scatter with an output tensor and an
            # input list.
            output = torch.empty(4, 4)
            return reduce_scatter(output, input_list)

        self._assert_not_supported(
            fn,
            [torch.randn(4, 4), torch.randn(4, 4)],
            "reduce_scatter",
            suggestion="torch.distributed._functional_collectives.reduce_scatter_single",
        )

    def test_nn_functional_all_to_all_single_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.all_to_all_single is unsupported; suggests
        # functional all_to_all_single.
        from torch.distributed.nn.functional import all_to_all_single

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls all_to_all_single with an empty-like output.
            output = torch.empty_like(x)
            return all_to_all_single(output, x)

        self._assert_not_supported(
            fn,
            torch.randn(4, 4),
            "all_to_all_single",
            suggestion="torch.distributed._functional_collectives.all_to_all_single",
        )

    def test_nn_functional_broadcast_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.broadcast is unsupported; suggestion is the
        # functional broadcast.
        from torch.distributed.nn.functional import broadcast

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls broadcast from source 0.
            return broadcast(x, src=0)

        self._assert_not_supported(
            fn,
            torch.randn(4),
            "broadcast",
            suggestion="torch.distributed._functional_collectives.broadcast",
        )

    def test_nn_functional_reduce_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.reduce is unsupported (no suggestion in
        # the message).
        from torch.distributed.nn.functional import reduce

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls reduce to destination 0.
            return reduce(x, dst=0)

        self._assert_not_supported(fn, torch.randn(4), "reduce")

    def test_nn_functional_gather_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.gather is unsupported (no suggestion in
        # the message).
        from torch.distributed.nn.functional import gather

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls gather to destination 0.
            return gather(x, dst=0)

        self._assert_not_supported(fn, torch.randn(4), "gather")

    def test_nn_functional_scatter_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.scatter is unsupported (no suggestion in
        # the message).
        from torch.distributed.nn.functional import scatter

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls scatter of two copies of x from source 0.
            return scatter([x, x], src=0)

        self._assert_not_supported(fn, torch.randn(4), "scatter")

    def test_nn_functional_all_to_all_unsupported(self):
        # [FUNCTION DOC] Asserts nn.functional.all_to_all is unsupported (no suggestion in
        # the message).
        from torch.distributed.nn.functional import all_to_all

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls all_to_all with an empty-like output list and an
            # input list.
            return all_to_all([torch.empty_like(x)], [x])

        self._assert_not_supported(fn, torch.randn(4), "all_to_all")

    def test_nn_functional_all_gather_base_unsupported(self):
        # [FUNCTION DOC] Asserts the private nn.functional._all_gather_base is unsupported
        # (no suggestion).
        from torch.distributed.nn.functional import _all_gather_base

        @torch.compile(fullgraph=True)
        def fn(x):
            # [FUNCTION DOC] HELPER: Calls _all_gather_base with an output of size 8.
            output = torch.empty(8)
            return _all_gather_base(output, x)

        self._assert_not_supported(fn, torch.randn(4), "_all_gather_base")


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
