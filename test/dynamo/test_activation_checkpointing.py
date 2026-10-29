# Owner(s): ["module: dynamo"]
# [MODULE DOC] Dynamo/AOTAutograd tests for activation checkpointing
# (torch.utils.checkpoint) under torch.compile.
# Contains: helper utilities (op counting on FX graphs, SAC policies, invalid-context stubs), the
# tag-based checkpoint test class (ActivationCheckpointingViaTagsTests), selective activation
# checkpointing (SAC) tests, region memory budget tests, the RematerializeACNodesPassTests class, a
# non-strict tracer test class and a nested-compile test class.
# Import-time side effects: computes device_type from the current accelerator (else cpu), defines a
# triton add_one_kernel only when a GPU with triton exists, builds requires_distributed, and
# instantiates ActivationCheckpointingViaTagsTests per non-CPU device type (so its test methods
# receive a device argument); run as a script it calls the dynamo run_tests.
import contextlib
import copy
import functools
import math
import re
import unittest
from importlib import import_module
from types import SimpleNamespace

import torch
import torch._dynamo.config
import torch._dynamo.test_case
import torch._functorch.config
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint
from functorch.compile import (
    default_partition,
    min_cut_rematerialization_partition,
    nop,
)
from torch._dynamo.backends.common import aot_autograd
from torch._dynamo.source import ConstantSource
from torch._dynamo.testing import (
    AotEagerAndRecordGraphs,
    CompileCounterWithBackend,
    normalize_gm,
)
from torch._functorch.partitioners import has_recomputable_rng_ops, is_rng_op
from torch._higher_order_ops.wrap import tag_activation_checkpoint
from torch.fx.experimental.sym_node import SymNode
from torch.fx.experimental.symbolic_shapes import DimDynamic, ShapeEnv
from torch.testing._internal.common_cuda import (
    PLATFORM_SUPPORTS_CUDNN_ATTENTION,
    PLATFORM_SUPPORTS_FLASH_ATTENTION,
    PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
)
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    IS_WINDOWS,
    parametrize,
    skipIfHpu,
    skipIfXpu,
    TEST_CUDA,
)
from torch.testing._internal.inductor_utils import HAS_GPU_AND_TRITON
from torch.testing._internal.triton_utils import requires_gpu_and_triton
from torch.testing._internal.two_tensor import TwoTensor
from torch.utils.checkpoint import (
    checkpoint,
    CheckpointPolicy,
    create_selective_checkpoint_contexts,
)


device_type = acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"


if HAS_GPU_AND_TRITON:
    import triton
    from triton import language as tl

    # [FUNCTION DOC] HELPER: triton kernel (defined only when a GPU and triton are available) that
    # reads n_elements values from in_ptr0, adds one to each and stores them to out_ptr. Each
    # program handles BLOCK_SIZE consecutive elements and masks out-of-range offsets.
    @triton.jit
    def add_one_kernel(
        in_ptr0,
        out_ptr,
        n_elements,
        BLOCK_SIZE: "tl.constexpr",
    ):
        pid = tl.program_id(axis=0)
        block_start = pid * BLOCK_SIZE
        offsets = block_start + tl.arange(0, BLOCK_SIZE)
        mask = offsets < n_elements
        x = tl.load(in_ptr0 + offsets, mask=mask)
        output = x + 1
        tl.store(out_ptr + offsets, output, mask=mask)


requires_distributed = functools.partial(
    unittest.skipIf, not dist.is_available(), "requires distributed"
)


def checkpoint_wrapper(fn):
    # [FUNCTION DOC] HELPER: wraps fn so the returned callable runs fn through the
    # reentrant torch.utils.checkpoint.checkpoint. Returns the wrapper; nothing is executed
    # until it is called.
    def inner(*args):
        # [FUNCTION DOC] HELPER: calls checkpoint with the closed-over fn and the positional args,
        # forcing the reentrant implementation. Returns fn's output.
        return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

    return inner


def _grad(*args, **kwargs):
    # [FUNCTION DOC] HELPER: thin forwarding wrapper around torch.autograd.grad; passes all
    # positional and keyword arguments through and returns the resulting gradient tuple.
    return torch.autograd.grad(*args, **kwargs)


def count_ops(
    gm, args, freq=None, freq_ge=None, op=None, freqs=None, freqs_ge=None, ops=None
):
    # [FUNCTION DOC] HELPER: graph-module "compiler" that checks how often given ops appear in gm
    # and returns gm unchanged, so it can be used as an fw_compiler or bw_compiler. A single op
    # (with freq or freq_ge) or parallel lists (ops with freqs or freqs_ge) can be given; a single
    # op must not be a list.
    # If freqs are provided each op must appear exactly that many times; otherwise freqs_ge must be
    # provided and each op must appear at least that many times. RNG higher-order-op wrappers
    # around an op also count. Raises AssertionError on a wrong count, on an op given as a list, or
    # when no frequency list is given.
    def match_rng_op(node, op):
        # [FUNCTION DOC] HELPER: returns True if node is an RNG higher-order op
        # (run_and_save_rng_state, run_with_rng_state or graphsafe_run_with_rng_state) that wraps
        # the given op, looking at args[0] or, for run_with_rng_state, at the second argument.
        # Returns False for every other node.
        if isinstance(node.target, torch._ops.HigherOrderOperator):
            if node.name == "run_and_save_rng_state":
                return node.args[0] == op
            elif node.name == "run_with_rng_state":
                return node.args[1] == op
            elif node.name == "graphsafe_run_with_rng_state":
                return node.args[0] == op
        return False

    # assert ((freq or freq_ge) and op) or ((freqs or freqs_ge) and ops)
    if op is not None:
        if isinstance(op, list):
            raise AssertionError("Expected op to not be a list")
        ops = [op]
    if freq is not None:
        freqs = [freq]
    if freq_ge is not None:
        freqs_ge = [freq_ge]
    if freqs:
        for op, freq in zip(ops, freqs):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            err_msg = f"In graph {gm}, expected {op} to have occurred {freq} times in the graph, but got {actual_count}."
            if actual_count != freq:
                raise AssertionError(err_msg)
    else:
        if freqs_ge is None:
            raise AssertionError("Expected freqs_ge to not be None")
        for op, freq_ge in zip(ops, freqs_ge):
            actual_count = 0
            for node in gm.graph.nodes:
                if match_rng_op(node, op) or node.target == op:
                    actual_count += 1
            if actual_count < freq_ge:
                raise AssertionError(
                    f"In graph {gm}, expected {op} to have occurred at least {freq_ge} times in the graph, but got {actual_count}."
                )
    return gm


def collect_fwd_graph_outputs(graph: torch.fx.Graph, *, fwd_outputs: set[str]):
    # [FUNCTION DOC] HELPER: post-grad custom pass used to record forward graph outputs. When not
    # inside a compiled autograd region (so the graph is a forward graph) it checks that the last
    # node is the output node and adds the string name of every returned value to the fwd_outputs
    # set (mutated in place). Does nothing for backward graphs. Raises AssertionError if the last
    # node is not an output node.
    if not torch._dynamo.compiled_autograd.in_compiled_autograd_region:  # fwd graph
        return_node = list(graph.nodes)[-1]
        if return_node.target != "output":
            raise AssertionError(
                f"Expected return_node.target to be 'output', got {return_node.target}"
            )
        for x in return_node.args[0]:
            fwd_outputs.add(str(x))


class _InvalidContext:
    # [CLASS DOC] HELPER: a context manager that does nothing; entering returns itself and exit
    # does not suppress exceptions. Used by _invalid_context_gen to build a context_fn result for
    # invalid-context checkpoint tests.
    def __init__(self) -> None:
        # [FUNCTION DOC] HELPER: no-op constructor with no state.
        pass

    def __enter__(self):
        # [FUNCTION DOC] HELPER: returns the context manager itself.
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # [FUNCTION DOC] HELPER: does nothing and returns None, so exceptions propagate.
        pass


def _invalid_context_gen():
    # [FUNCTION DOC] HELPER: context_fn stand-in that returns a pair of fresh _InvalidContext
    # objects (one for forward, one for recompute) instead of real selective-checkpoint contexts.
    return _InvalidContext(), _InvalidContext()


def find_first_node(gm, func):
    # [FUNCTION DOC] HELPER: returns the first node in gm.graph whose target is exactly func
    # (identity comparison), or None when there is no such node.
    for node in gm.graph.nodes:
        if node.target is func:
            return node
    return None


def op_count(gm):
    # [FUNCTION DOC] HELPER: counts the nodes of gm.graph whose op kind contains "call"
    # (call_function, call_method, call_module) and returns the count.
    result = 0
    for node in gm.graph.nodes:
        if "call" in node.op:
            result += 1
    return result


def _get_custom_policy(no_recompute_list=None, must_recompute_list=None):
    # [FUNCTION DOC] HELPER: builds a selective-checkpoint policy callable. The policy returns
    # MUST_SAVE when the op is in no_recompute_list, MUST_RECOMPUTE when it is in
    # must_recompute_list, and PREFER_RECOMPUTE otherwise; a list left as None is ignored.
    # no_recompute_list is checked first.
    def _custom_policy(ctx, func, *args, **kwargs):
        # [FUNCTION DOC] HELPER: the policy: receives ctx, the op func and its args/kwargs and maps
        # func to a CheckpointPolicy using the closed-over lists (no_recompute_list first, then
        # must_recompute_list, else PREFER_RECOMPUTE).
        if no_recompute_list is not None and func in no_recompute_list:
            return CheckpointPolicy.MUST_SAVE
        if must_recompute_list is not None and func in must_recompute_list:
            return CheckpointPolicy.MUST_RECOMPUTE
        else:
            return CheckpointPolicy.PREFER_RECOMPUTE

    return _custom_policy


class ActivationCheckpointingViaTagsTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Test case for activation checkpointing under torch.compile with the
    # AOTAutograd backend. Most tests wrap a small function with torch.utils.checkpoint
    # (reentrant or not, with or without selective-checkpoint context_fn), compile it with a
    # custom aot_autograd backend whose fw/bw compilers count ops, and compare results and
    # gradients with eager. Also covers tags in tlparse, graph breaks, kwargs, symints, RNG,
    # hooks and region activation memory budgets. Helpers _validate and
    # _compare_orig_and_checkpointed_fns are the shared comparison drivers.
    def _validate(
        self,
        fn,
        backend,
        *args,
        skip_check=False,
        fullgraph=True,
        compiled_autograd=False,
    ):
        # [FUNCTION DOC] HELPER: runs fn eagerly on args (seed 0) and backward on the summed
        # output, then compiles a deep copy of fn with torch.compile (given backend, fullgraph
        # flag) and runs it on detached clones of args that keep requires_grad (seed 0 again). With
        # compiled_autograd the backward runs inside a compiled-autograd context that compiles the
        # backward graph with the same backend.
        # Unless skip_check, asserts the outputs and every argument's .grad match eager;
        # raises through assertEqual on mismatch. Side effects: populates .grad on the
        # original args and clones.
        cloned_args = []
        for arg in args:
            cloned_args.append(arg.detach().clone().requires_grad_(arg.requires_grad))

        cloned_fn = copy.deepcopy(fn)

        torch.manual_seed(0)
        expected = fn(*args)
        expected.sum().backward()

        torch.manual_seed(0)
        compiled_fn = torch.compile(cloned_fn, fullgraph=fullgraph, backend=backend)
        ctx = contextlib.nullcontext()
        if compiled_autograd:
            ctx = torch._dynamo.compiled_autograd._enable(
                lambda gm: torch.compile(gm, fullgraph=fullgraph, backend=backend)
            )
        with ctx:
            result = compiled_fn(*cloned_args)
            result.sum().backward()

        if not skip_check:
            self.assertEqual(
                result,
                expected,
                msg="Output mismatch between torch.compile and eager versions",
            )
            for arg, cloned_arg in zip(args, cloned_args):
                self.assertEqual(
                    arg.grad,
                    cloned_arg.grad,
                    msg="Gradient mismatch between torch.compile and eager versions",
                )

    def _compare_orig_and_checkpointed_fns(
        self, orig_fn, checkpointed_fn, *args, fullgraph=True
    ):
        # [FUNCTION DOC] HELPER: compares orig_fn and checkpointed_fn: each is compiled and run on
        # cloned inputs with seed 0, then backward on the summed output, and outputs and argument
        # gradients are asserted equal. The comparison runs once with torch.compile (honoring
        # fullgraph) and, only when fullgraph is True, a second time after exporting the function
        # to torch IR and compiling with the aot_eager_decomp_partition backend.
        # The original version and the checkpointed version of the same function
        # should produce the same outputs and the same gradients under torch.compile.

        def clone_args(args):
            # [FUNCTION DOC] HELPER: returns a list of detached clones of args that keep each
            # original requires_grad flag.
            cloned_args = []
            for arg in args:
                cloned_args.append(
                    arg.detach().clone().requires_grad_(arg.requires_grad)
                )
            return cloned_args

        def run(compiler):
            # [FUNCTION DOC] HELPER: with the given compiler callable, compiles orig_fn and a deep
            # copy of checkpointed_fn, runs both on freshly cloned args under seed 0,
            # backpropagates the summed outputs and asserts the outputs and gradients are equal.
            # Run original version
            cloned_args_orig_fn = clone_args(args)
            torch.manual_seed(0)
            compiled_orig_fn = compiler(orig_fn)
            result_orig_fn = compiled_orig_fn(*cloned_args_orig_fn)
            result_orig_fn.sum().backward()

            # Run checkpointed version
            cloned_args_checkpointed_fn = clone_args(args)
            torch.manual_seed(0)
            compiled_checkpointed_fn = compiler(copy.deepcopy(checkpointed_fn))
            result_checkpointed_fn = compiled_checkpointed_fn(
                *cloned_args_checkpointed_fn
            )
            result_checkpointed_fn.sum().backward()

            # Check that outputs and gradients are equal
            self.assertEqual(
                result_orig_fn,
                result_checkpointed_fn,
                msg="Output mismatch between the original version and the checkpointed version of the same function",
            )
            for cloned_arg_orig_fn, cloned_arg_checkpointed_fn in zip(
                cloned_args_orig_fn, cloned_args_checkpointed_fn
            ):
                self.assertEqual(
                    cloned_arg_orig_fn.grad,
                    cloned_arg_checkpointed_fn.grad,
                    msg="Gradient mismatch between the original version and the checkpointed version of the same function",
                )

        run(functools.partial(torch.compile, fullgraph=fullgraph))
        if fullgraph:

            def export_compiler(fn):
                # [FUNCTION DOC] HELPER: "compiler" that wraps fn in a module and returns
                # runtime_wrapper, which exports that module to torch IR and compiles it each time
                # it is called.
                class WrapAsModule(nn.Module):
                    # [CLASS DOC] HELPER: nn.Module whose forward simply calls the enclosed fn with
                    # the given arguments; lets a plain function be exported.
                    def forward(self, *args, **kwargs):
                        # [FUNCTION DOC] HELPER: forwards all arguments to the enclosed fn and
                        # returns its result.
                        return fn(*args, **kwargs)

                mod = WrapAsModule()

                def runtime_wrapper(*runtime_args):
                    # [FUNCTION DOC] HELPER: exports the module to torch IR (no dynamic shapes,
                    # using cloned test args as example inputs, not the runtime args), then
                    # compiles the result with the aot_eager_decomp_partition backend (not
                    # fullgraph) and calls it with runtime_args; returns the output. The compile
                    # step is needed so RNG ops get added to the graph.
                    from torch.export import _trace

                    gm = _trace._export_to_torch_ir(
                        f=mod,
                        args=tuple(clone_args(args)),
                        kwargs={},
                        dynamic_shapes=None,
                        preserve_module_call_signature=(),
                        restore_fqn=False,
                        prefer_deferred_runtime_asserts_over_guards=False,
                        _log_export_usage=False,
                    )
                    # NOTE: this is necessary for rng to be added to the exported graph
                    return torch.compile(
                        gm, fullgraph=False, backend="aot_eager_decomp_partition"
                    )(*runtime_args)

                return runtime_wrapper

            run(export_compiler)

    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_function(self, device, partition_fn):
        # [FUNCTION DOC] Checkpointed function (reentrant) computing sigmoid of a matmul on sin(x)
        # and y. Compiles with aot_autograd using the parametrized partition function and expects 1
        # mm in the forward graph and 3 in the backward (one recomputed plus two for gradients);
        # compares with eager through _validate.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of the matmul of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: applies reentrant torch.utils.checkpoint.checkpoint to gn on
            # sin(x) and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_function_via_global_checkpoint(self, device, partition_fn):
        # [FUNCTION DOC] Same as test_tags_function but the checkpoint call goes through the
        # globally imported checkpoint name (resolved by VariableBuilder in dynamo). Expects 1 mm
        # forward and 3 mm backward with each partition function and eager parity.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of the matmul of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: calls the globally imported checkpoint (reentrant) on gn with
            # sin(x) and y.
            # This goes through VariableBuilder
            return checkpoint(gn, torch.sin(x), y, use_reentrant=True)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_function_with_kwargs(self, device, partition_fn):
        # [FUNCTION DOC] Same pattern as test_tags_function but using the non-reentrant
        # checkpoint (use_reentrant False). Expects 1 forward mm, 3 backward mm and eager parity
        # for each partition function.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of the matmul of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn on sin(x) and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=False
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=1, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=3, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    def test_checkpoint_shows_tags_in_tlparse(self, device):
        # [FUNCTION DOC] Checks that checkpointed forward nodes are labeled in the joint graph
        # readable output: the partition_fn asserts the printed joint graph matches an ac_graph_id
        # comment with PREFER_RECOMPUTE, then delegates to the min-cut partitioner. Compiles only;
        # no numeric comparison.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of the matmul of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of gn on sin(x) and y
            # with preserve_rng_state disabled.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=True, preserve_rng_state=False
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            # [FUNCTION DOC] HELPER: wraps the min-cut partitioner; first prints the joint graph
            # readably and asserts via assertTrue that it contains an ac_graph_id annotation with
            # PREFER_RECOMPUTE, then returns the min-cut partition of the joint graph.
            gm_str = joint_gm.print_readable(print_output=False)
            # Check for the pattern with any graph ID (the ID depends on test order)
            self.assertTrue(
                re.search(r"# ac_graph_id: \d+ - PREFER_RECOMPUTE", gm_str),
                lambda msg: f"{msg}\nExpected ac_graph_id pattern not found in:\n{gm_str}",
            )
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        _ = torch.compile(fn, backend=backend)(x, y)

    @requires_gpu_and_triton
    def test_ac_tags_through_custom_autograd_function(self, device):
        # [FUNCTION DOC] Verifies that checkpoint tags (recompute and ac_graph_id metadata) reach
        # forward mm nodes when the checkpointed function uses a custom autograd.Function. Uses a
        # partition_fn that asserts those metadata keys exist on every forward mm node; then runs
        # forward and backward with no-op compilers.
        class MyMM(torch.autograd.Function):
            # [CLASS DOC] HELPER: custom autograd.Function computing a matrix product of x and w,
            # saving both inputs, with backward returning grad times w transposed and x
            # transposed times grad.
            @staticmethod
            def forward(x, w):
                # [FUNCTION DOC] HELPER: returns the matrix product of x and w.
                return x @ w

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: saves x and w from inputs for backward; the
                # output is unused.
                x, w = inputs
                ctx.save_for_backward(x, w)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: returns gradients for x and w computed from
                # the saved tensors.
                x, w = ctx.saved_tensors
                return grad @ w.t(), x.t() @ grad

        def gn(x, w):
            # [FUNCTION DOC] HELPER: applies MyMM to x and w.
            return MyMM.apply(x, w)

        def fn(x, w):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn on x and w.
            return torch.utils.checkpoint.checkpoint(gn, x, w, use_reentrant=False)

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            # [FUNCTION DOC] HELPER: collects forward-tagged (partitioner_tag is_forward) aten.mm
            # nodes in the joint graph, asserts the list is non-empty and that each has recompute
            # and ac_graph_id metadata, then returns the min-cut partition of the joint graph.
            fwd_mm_nodes = [
                node
                for node in joint_gm.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.aten.mm.default
                and node.meta.get("partitioner_tag") == "is_forward"
            ]
            self.assertTrue(
                fwd_mm_nodes, "Expected forward mm nodes in the joint graph"
            )
            for node in fwd_mm_nodes:
                self.assertIn("recompute", node.meta)
                self.assertIn("ac_graph_id", node.meta)
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        out = torch.compile(fn, backend=backend)(x, w)
        out.sum().backward()

    @requires_gpu_and_triton
    def test_sac_tags_through_custom_autograd_function(self, device):
        # [FUNCTION DOC] Same as the previous test but with selective checkpointing: the context_fn
        # uses a policy that always returns PREFER_RECOMPUTE. Asserts forward mm nodes carry
        # recompute and ac_graph_id metadata when a custom autograd.Function is used, then runs
        # forward and backward with no-op compilers.
        class MyMM(torch.autograd.Function):
            # [CLASS DOC] HELPER: custom autograd.Function computing a matrix product of x and w,
            # saving both inputs, with backward returning grad times w transposed and x
            # transposed times grad.
            @staticmethod
            def forward(x, w):
                # [FUNCTION DOC] HELPER: returns the matrix product of x and w.
                return x @ w

            @staticmethod
            def setup_context(ctx, inputs, output):
                # [FUNCTION DOC] HELPER: saves x and w from inputs for backward; the
                # output is unused.
                x, w = inputs
                ctx.save_for_backward(x, w)

            @staticmethod
            def backward(ctx, grad):
                # [FUNCTION DOC] HELPER: returns gradients for x and w computed from
                # the saved tensors.
                x, w = ctx.saved_tensors
                return grad @ w.t(), x.t() @ grad

        def gn(x, w):
            # [FUNCTION DOC] HELPER: applies MyMM to x and w.
            return MyMM.apply(x, w)

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts,
            lambda ctx,
            op,
            *args,
            **kwargs: torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE,
        )

        def fn(x, w):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn on x and w using
            # the PREFER_RECOMPUTE context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn, x, w, use_reentrant=False, context_fn=context_fn
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        w = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            # [FUNCTION DOC] HELPER: asserts that the joint graph has forward-tagged aten.mm nodes
            # each with recompute and ac_graph_id metadata, then returns the min-cut partition.
            fwd_mm_nodes = [
                node
                for node in joint_gm.graph.nodes
                if node.op == "call_function"
                and node.target == torch.ops.aten.mm.default
                and node.meta.get("partitioner_tag") == "is_forward"
            ]
            self.assertTrue(
                fwd_mm_nodes, "Expected forward mm nodes in the joint graph"
            )
            for node in fwd_mm_nodes:
                self.assertIn("recompute", node.meta)
                self.assertIn("ac_graph_id", node.meta)
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        out = torch.compile(fn, backend=backend)(x, w)
        out.sum().backward()

    @requires_gpu_and_triton
    def test_tangent_placeholders_have_is_backward_tag(self, device):
        # [FUNCTION DOC] Checks that in the joint graph placeholders whose target mentions tangents
        # get a partitioner_tag containing is_backward, while all other placeholders get
        # is_forward. Verified inside a partition_fn that then delegates to the min-cut
        # partitioner; no numeric comparison.
        """Test that tangent placeholders in the joint graph are tagged with is_backward."""

        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of the matmul of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn on sin(x) and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=False
            )

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        def partition_fn(joint_gm, *args, **kwargs):
            # [FUNCTION DOC] HELPER: for every placeholder node in the joint graph asserts the
            # partitioner_tag is_backward for tangent inputs and is_forward otherwise, then returns
            # the min-cut partition.
            # Check partitioner_tag on placeholder nodes
            for node in joint_gm.graph.nodes:
                if node.op == "placeholder":
                    if "tangents" in str(node.target):
                        self.assertTrue(
                            "is_backward" in node.meta.get("partitioner_tag", "")
                        )
                    else:
                        self.assertTrue(
                            "is_forward" in node.meta.get("partitioner_tag", "")
                        )
            return min_cut_rematerialization_partition(joint_gm, *args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop, bw_compiler=nop, partition_fn=partition_fn
        )
        _ = torch.compile(fn, backend=backend)(x, y)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_sequential_layers(self, device, partition_fn):
        # [FUNCTION DOC] Two sequential checkpoints of a function with a cos, three matmuls and a
        # cos. Expects 6 mm in the forward graph and, in the backward, 2 cos and 18 mm
        # (recomputation plus gradient matmuls); checks eager parity for each partition function.
        def gn(x):
            # [FUNCTION DOC] HELPER: applies cos, three successive self matmuls and cos to x and
            # returns the result.
            x = x.cos()
            for _ in range(3):
                x = torch.mm(x, x)
            x = x.cos()
            return x

        def fn(x):
            # [FUNCTION DOC] HELPER: applies checkpoint (default reentrant setting) to gn
            # twice in sequence.
            x = torch.utils.checkpoint.checkpoint(gn, x)
            x = torch.utils.checkpoint.checkpoint(gn, x)
            return x

        x = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=6, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops,
            freqs=[2, 18],
            ops=[torch.ops.aten.cos.default, torch.ops.aten.mm.default],
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_multiple_checkpoints(self, device, partition_fn):
        # [FUNCTION DOC] Two reentrant checkpoints of the same function separated by a sin.
        # Expects 2 mm in the forward and 6 in the backward graph and eager parity, for
        # each partition function.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of the matmul of x and y.
            return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: sin of x, checkpoint gn, sin again, checkpoint gn again,
            # return the final result.
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(z)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return z

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        bw_compiler = functools.partial(
            count_ops, freq=6, op=torch.ops.aten.mm.default
        )  # mm recomputed in the bwd
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_module(self, device, partition_fn):
        # [FUNCTION DOC] Checkpoints an nn.Module (linear plus sigmoid) in reentrant mode. Expects
        # exactly one sigmoid in each of the forward and backward graphs (recomputed) and eager
        # parity, per partition function.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a Linear(10, 10) layer whose forward applies sigmoid
            # to the linear output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the Linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns sigmoid of the linear layer applied to x.
                return torch.sigmoid(self.linear(x))

        mod = MockModule().to(device)

        def fn(x):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of the module on sin(x).
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, device=device, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.sigmoid.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_tags_decomps(self, device, partition_fn):
        # [FUNCTION DOC] Ensures checkpoint tags survive decomposition: a checkpointed linear plus
        # gelu is compiled with the inductor decomposition table, and exactly one erf must be
        # present in each of the forward and backward graphs (gelu recomputed). Checks eager parity
        # for each partition function.
        # Ensures that tags are passed on through decompositions as well
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a Linear(10, 10) layer whose forward applies gelu to
            # the linear output.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the Linear layer.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns gelu of the linear layer applied to x.
                return torch.nn.functional.gelu(self.linear(x))

        mod = MockModule().to(device)

        def fn(x):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of the module on sin(x).
            return torch.utils.checkpoint.checkpoint(
                mod, torch.sin(x), use_reentrant=True
            )

        x = torch.randn(10, 10, device=device, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        bw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.aten.erf.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
            decompositions=lambda: import_module(
                "torch._inductor.compile_fx"
            ).select_decomp_table(),
        )
        self._validate(fn, backend, x)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_recomputed_rand(self, device):
        # [FUNCTION DOC] Compiles with the inductor backend (fallback_random enabled) two
        # checkpoints of a function that multiplies by random values, and checks outputs
        # and gradients against eager through _validate; verifies recomputed random ops
        # give consistent results.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: returns sigmoid of a random tensor times y, multiplied by x.
            return torch.sigmoid(torch.rand_like(x) * y) * x

        def fn(x, y):
            # [FUNCTION DOC] HELPER: sin, checkpoint gn, sin, checkpoint gn again, and returns
            # the final result.
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(x)
            z = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return z

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_rand(self, device):
        # [FUNCTION DOC] Compiles with inductor (fallback_random enabled) a checkpoint of two
        # successive matmuls between sin ops and compares result and gradients with eager using
        # _validate. Despite the name no random ops are called here.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: applies mm with y twice to x.
            x = torch.mm(x, y)
            x = torch.mm(x, y)
            return x

        def fn(x, y):
            # [FUNCTION DOC] HELPER: sin, a reentrant checkpoint of gn, and sin again.
            x = torch.sin(x)
            x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            x = torch.sin(x)
            # x = torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)
            return x

        x = torch.randn(4, 4, device=device, requires_grad=True)
        y = torch.randn(4, 4, device=device, requires_grad=True)

        # fw_compiler = functools.partial(count_ops, freq=2, op=torch.ops.aten.mm.default)
        # bw_compiler = functools.partial(
        #     count_ops, freq=6, op=torch.ops.aten.mm.default
        # )  # mm recomputed in the bwd
        # backend = aot_autograd(fw_compiler=fw_compiler, bw_compiler=bw_compiler)
        # backend = "aot_eager"
        backend = "inductor"
        self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    @torch._inductor.config.patch(fallback_random=True)
    def test_tags_dropout(self, device):
        # [FUNCTION DOC] Compiles a checkpointed linear plus dropout module with inductor
        # (fallback_random) and runs forward and backward through _validate with skip_check, since
        # the decomposed random ops do not match eager numerically.
        # Figure out a way to test the number of inductor_random calls
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with a Linear(10, 10) followed by Dropout(0.2).
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the linear and dropout layers.
                super().__init__()
                self.linear = torch.nn.Linear(10, 10)
                self.dropout = torch.nn.Dropout(0.2)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the linear layer then dropout.
                return self.dropout(self.linear(x))

        mod = MockModule().to(device)

        def fn(x):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of the module on x.
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        x = torch.randn(10, 10, device=device, requires_grad=True)
        backend = "inductor"
        # rand decomps do not have have numerical results as eager
        self._validate(fn, backend, x, skip_check=True)

    @skipIfHpu
    @torch._functorch.config.patch(recompute_views=True)
    @torch._inductor.config.patch(fx_graph_cache=False)
    def test_tags_must_save_tensor_that_has_backward_hook(self):
        # [FUNCTION DOC] Compiles (inductor, compiled autograd, fullgraph) a module with a non-
        # reentrant checkpointed submodule twice: once plain and once with a forward hook that
        # registers a backward hook on the submodule output. Collects the forward graph outputs of
        # each via a post_grad custom pre-pass and asserts the hooked version saves exactly one
        # extra non-primal tensor, named mul (the product y times y). Resets dynamo between runs.
        # Skipped on HPU; runs with recompute_views on and fx_graph_cache off.
        def my_post_forward_hook(submod, args, output):
            # [FUNCTION DOC] HELPER: forward hook that registers my_backward_hook on the submodule
            # output tensor and returns that output unchanged.
            output.register_hook(my_backward_hook)
            return output

        def my_backward_hook(grad):
            # [FUNCTION DOC] HELPER: backward hook that returns the incoming gradient unchanged.
            return grad

        class MySubmod(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward computes y as the matmul of x with itself
            # and returns y times y.
            def __init__(self):
                # [FUNCTION DOC] HELPER: calls the base constructor only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns the elementwise square of the matmul of
                # x with itself.
                y = torch.matmul(x, x)
                z = y * y
                return z

        class MyMod(torch.nn.Module):
            # [CLASS DOC] HELPER: holds a MySubmod and a LayerNorm(4); forward runs the submodule
            # under non-reentrant checkpoint and then the layer norm.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the submodule and the layer norm.
                super().__init__()
                self.submod = MySubmod()
                self.norm = torch.nn.LayerNorm(4)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of the submodule on x, followed
                # by the layer norm; returns the normalized output.
                out = torch.utils.checkpoint.checkpoint(
                    self.submod, x, use_reentrant=False
                )
                norm_out = self.norm(out)
                return norm_out

        def _factory_fn():
            # [FUNCTION DOC] HELPER: returns a fresh MyMod, a 4x4 float32 ones input that requires
            # grad, and the backend name inductor.
            mod = MyMod()
            x = torch.ones(4, 4, dtype=torch.float32, requires_grad=True)
            backend = "inductor"
            return mod, x, backend

        mod_no_hook, x, backend = _factory_fn()
        mod_no_hook_fwd_outputs = set()

        with torch._inductor.config.patch(
            post_grad_custom_pre_pass=functools.partial(
                collect_fwd_graph_outputs, fwd_outputs=mod_no_hook_fwd_outputs
            )
        ):
            self._validate(
                mod_no_hook, backend, x, fullgraph=True, compiled_autograd=True
            )

        torch._dynamo.reset()
        mod_with_hook, x, backend = _factory_fn()
        mod_with_hook.submod.register_forward_hook(my_post_forward_hook)
        mod_with_hook_fwd_outputs = set()

        with torch._inductor.config.patch(
            post_grad_custom_pre_pass=functools.partial(
                collect_fwd_graph_outputs, fwd_outputs=mod_with_hook_fwd_outputs
            )
        ):
            self._validate(
                mod_with_hook, backend, x, fullgraph=True, compiled_autograd=True
            )

        # If `z` has a backward hook, result of `z = y * y` should also be saved in addition to the usual saved tensors.
        mod_no_hook_fwd_outputs_no_primal = {
            x for x in mod_no_hook_fwd_outputs if not x.startswith("primals_")
        }
        mod_with_hook_fwd_outputs_no_primal = {
            x for x in mod_with_hook_fwd_outputs if not x.startswith("primals_")
        }
        additional_saved_tensors = (
            mod_with_hook_fwd_outputs_no_primal - mod_no_hook_fwd_outputs_no_primal
        )
        expected_additional_saved_tensors = {"mul"}
        self.assertEqual(
            additional_saved_tensors,
            expected_additional_saved_tensors,
            lambda msg: f"""{msg}\n
Expected additional saved tensors: {expected_additional_saved_tensors} but got: {additional_saved_tensors}.
Non-primal fwd outputs from model w/ backward hook: {mod_with_hook_fwd_outputs_no_primal}.
Non-primal fwd outputs from model w/o backward hook: {mod_no_hook_fwd_outputs_no_primal}.""",
        )

    @requires_gpu_and_triton
    def test_fallback(self, device):
        # [FUNCTION DOC] Checkpoints (non-reentrant) a function containing two explicit graph
        # breaks, wrapped between a sin and a cos, and compiles with the aot_eager counting
        # backend. Asserts the result equals eager and that two frames, two ops and two graphs were
        # produced (one for sin on the input, one for the final cos).
        def gn(x, y):
            # [FUNCTION DOC] HELPER: graph break, sigmoid of matmul of x and y, another graph
            # break, then cos of that value.
            torch._dynamo.graph_break()
            a = torch.sigmoid(torch.matmul(x, y))
            torch._dynamo.graph_break()
            return torch.cos(a)

        def fn(x, y):
            # [FUNCTION DOC] HELPER: cos of the non-reentrant checkpoint of gn applied to
            # sin(x) and y.
            return torch.cos(checkpoint(gn, torch.sin(x), y, use_reentrant=False))

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
        args = (x, y)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        # One graph for torch.sin on the input, and other for torch.cos.
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(cnt.op_count, 2)
        self.assertEqual(len(cnt.graphs), 2)

    @requires_gpu_and_triton
    def test_kwargs(self, device):
        # [FUNCTION DOC] Checks that keyword arguments to checkpoint are forwarded: compiles fn
        # with the aot_eager counting backend and asserts the result matches eager, a single frame
        # and graph, that the tag_activation_checkpoint node has 4 args (body plus x, y, z) and
        # that the body graph has 2 call ops.
        def gn(x, y, z=None):
            # [FUNCTION DOC] HELPER: matmul of x and y; if z is given, multiplies the result
            # again by z.
            a = torch.matmul(x, y)
            if z is not None:
                return torch.matmul(a, z)
            return a

        def fn(x, y, z):
            # [FUNCTION DOC] HELPER: cos of non-reentrant checkpoint of gn with z passed as
            # a keyword argument.
            return torch.cos(checkpoint(gn, x, y, use_reentrant=False, z=z))

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
        z = torch.randn(4, 4, requires_grad=True, device=device)
        args = (x, y, z)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)

        expected = fn(*args)
        result = torch.compile(fn, backend=cnt)(*args)

        self.assertEqual(result, expected)

        self.assertEqual(cnt.frame_count, 1)
        self.assertEqual(len(cnt.graphs), 1)

        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        # one for checkpoint, and 3 for x, y, z
        self.assertEqual(len(wrap_node.args), 4)

        body_function = getattr(cnt.graphs[0], wrap_node.args[0].name)
        self.assertEqual(op_count(body_function), 2)

    @requires_gpu_and_triton
    def test_symints_location(self, device):
        # [FUNCTION DOC] Runs a reentrant-checkpointed matmul with dropout on 4x4 inputs and then
        # 5x5 inputs under the aot_eager counting backend (not fullgraph). Asserts shapes match
        # eager, that two frames and two graphs were created (the shape change forces a recompile)
        # and that the first graph's tag_activation_checkpoint node has 3 args.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: matmul of x with dropout (p 0.5) of y.
            return torch.matmul(x, torch.nn.functional.dropout(y, 0.5))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of gn on x and y.
            return torch.utils.checkpoint.checkpoint(gn, x, y, use_reentrant=True)

        backend = "aot_eager"
        cnt = CompileCounterWithBackend(backend)
        opt_fn = torch.compile(fn, backend=cnt)

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        x = torch.randn(5, 5, requires_grad=True, device=device)
        y = torch.randn(5, 5, requires_grad=True, device=device)
        args = (x, y)
        expected = fn(*args)
        result = opt_fn(*args)

        self.assertEqual(result.shape, expected.shape)
        self.assertEqual(cnt.frame_count, 2)
        self.assertEqual(len(cnt.graphs), 2)
        wrap_node = find_first_node(cnt.graphs[0], tag_activation_checkpoint)
        self.assertEqual(len(wrap_node.args), 3)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_must_recompute(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing with mm policies. With a MUST_RECOMPUTE policy for
        # mm the backward graph must have 6 mm ops; with MUST_SAVE for mm only 4. The forward
        # always has 2 mm. Eager parity is checked via _validate for each partition function.
        def context_fn_must_recompute_mm():
            # [FUNCTION DOC] HELPER: context_fn returning selective checkpoint contexts whose
            # policy marks aten.mm as MUST_RECOMPUTE.
            must_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    must_recompute_list=must_recompute_list,
                ),
            )

        def context_fn_no_recompute_mm():
            # [FUNCTION DOC] HELPER: context_fn returning selective checkpoint contexts whose
            # policy marks aten.mm as MUST_SAVE.
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(
                    no_recompute_list=no_recompute_list,
                ),
            )

        def _test(context_fn, bw_compiler, partition_fn):
            # [FUNCTION DOC] HELPER: builds a non-reentrant checkpointed function with the given
            # context_fn, compiles with aot_autograd (forward mm count fixed at 2, the given
            # bw_compiler, the given partition_fn) and validates against eager.
            def gn(x):
                # [FUNCTION DOC] HELPER: returns cos of sin of (x matmul x) matmul x.
                return torch.cos(torch.sin(torch.matmul(x, x) @ x))

            def fn(x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn on x with
                # the enclosing context_fn.
                return torch.utils.checkpoint.checkpoint(
                    gn,
                    x,
                    use_reentrant=False,
                    context_fn=context_fn,
                )

            x = torch.randn(4, 4, requires_grad=True, device=device)

            fw_compiler = functools.partial(
                count_ops,
                freq=2,
                op=torch.ops.aten.mm.default,
            )

            backend = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=partition_fn,
            )
            self._validate(fn, backend, x)

        _test(
            context_fn=context_fn_must_recompute_mm,
            bw_compiler=functools.partial(
                count_ops,
                freq=6,  # 1 matmul recompute and 2 bwd mm ops per fwd matmul, so 2 + 2 * 2 = 6)
                op=torch.ops.aten.mm.default,
            ),
            partition_fn=partition_fn,
        )
        _test(
            context_fn=context_fn_no_recompute_mm,
            bw_compiler=functools.partial(
                count_ops,
                freq=4,  # 2 bwd mm ops per fwd matmul
                op=torch.ops.aten.mm.default,
            ),
            partition_fn=partition_fn,
        )

    def test_sac_with_partial_context_fn(self):
        # [FUNCTION DOC] Checks that a context_fn built with functools.partial around
        # create_selective_checkpoint_contexts and a callable policy object works. Compiles fn with
        # aot_eager_decomp_partition (fullgraph, CPU) and asserts output equals eager.
        class CustomPolicy:
            # [CLASS DOC] HELPER: policy object whose call always returns MUST_SAVE; note it takes
            # an extra out argument in its signature.
            def __init__(self):
                # [FUNCTION DOC] HELPER: calls the base constructor only.
                super().__init__()

            def __call__(self, ctx, out, func, *args, **kwargs):
                # [FUNCTION DOC] HELPER: ignores all arguments and returns
                # CheckpointPolicy.MUST_SAVE.
                return CheckpointPolicy.MUST_SAVE

        def f(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of two chained matmuls of x with y, multiplied by y.
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        context_fn1 = functools.partial(
            create_selective_checkpoint_contexts, CustomPolicy()
        )

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of f on x and y with
            # the partial context_fn.
            return torch.utils.checkpoint.checkpoint(
                f,
                x,
                y,
                use_reentrant=False,
                context_fn=context_fn1,
            )

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        a = torch.randn(4, 4, requires_grad=True, device="cpu")
        b = torch.randn(4, 4, requires_grad=True, device="cpu")

        expected = fn(a, b)
        result = opt_fn(a, b)
        self.assertEqual(result, expected)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_must_not_recompute_gemm(
        self, device, partition_fn
    ):
        # [FUNCTION DOC] Selective checkpointing where mm is MUST_SAVE: forward has 2 mm and
        # backward only 4 (the matmuls are not recomputed; 6 would be needed without SAC). Checks
        # eager parity through _validate and also compares the plain and checkpointed functions
        # with _compare_orig_and_checkpointed_fns, per partition function.
        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn with a policy that saves aten.mm and
            # recomputes everything else.
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of two chained matmuls of x with y, times y.
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 6 here
            # (2 matmul recompute and 2 mm ops per fwd matmul, so 2 + 2 * 2 = 6)
            # if we didn't enable selective checkpointing.
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_must_not_recompute_gemm_no_functionalization(
        self, device, partition_fn
    ):
        # [FUNCTION DOC] Same selective-checkpoint setup as the previous test but the aot_autograd
        # backend runs with functionalization disabled. Mainly checks that sigmoid is recomputed
        # (one sigmoid in each of the forward and backward graphs); also validates against eager
        # and compares plain versus checkpointed functions.
        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn with a policy that saves aten.mm and
            # recomputes everything else.
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of two chained matmuls of x with y, times y.
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=torch.ops.aten.sigmoid.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # Main check here is just that sigmoid is properly recomputed
            # (we will see a sigmoid() and sigmoid_backward() in the bw graph)
            freq=1,
            op=torch.ops.aten.sigmoid.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
            disable_functionalization=True,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_triton_kernel(self, device, partition_fn):
        # [FUNCTION DOC] Same as the mm-MUST_SAVE selective checkpoint test but the region also
        # contains a custom autograd.Function that launches the triton add_one_kernel; makes sure
        # this does not error. Expects 2 forward and 4 backward mm ops and checks eager parity
        # and plain-versus-checkpointed parity.
        # Copy of the above test, but make sure that having a triton kernel in the
        # region does not error.
        def add_one(x):
            # [FUNCTION DOC] HELPER: allocates an output like x and runs the triton add_one_kernel
            # over x (one program per element, BLOCK_SIZE 4); returns the output.
            out = torch.empty_like(x)
            n_elements = x.numel()
            add_one_kernel[(n_elements,)](x, out, n_elements, BLOCK_SIZE=4)
            return out

        class AddOne(torch.autograd.Function):
            # [CLASS DOC] HELPER: custom autograd.Function wrapping the triton add_one kernel;
            # backward passes the gradient through unchanged.
            @staticmethod
            def forward(ctx, x):
                # [FUNCTION DOC] HELPER: returns add_one(x).
                return add_one(x)

            @staticmethod
            def backward(ctx, x):
                # [FUNCTION DOC] HELPER: returns the incoming gradient unchanged.
                return x

        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn with a policy that saves aten.mm and
            # recomputes everything else.
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of two chained matmuls of AddOne(sin(x)) with
            # y, times y.
            return (
                torch.sigmoid(torch.matmul(torch.matmul(AddOne.apply(x.sin()), y), y))
                * y
            )

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 6 here
            # (2 matmul recompute and 2 mm ops per fwd matmul, so 2 + 2 * 2 = 6)
            # if we didn't enable selective checkpointing.
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_tensor_subclass(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing (mm saved) with TwoTensor tensor-subclass inputs.
        # Expects 4 forward and 8 backward mm ops (12 without SAC) and checks eager parity and
        # plain-versus-checkpointed parity, per partition function.
        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn with a policy that saves aten.mm and
            # recomputes everything else.
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of two chained matmuls of x with y, times y.
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        rand_tensor = torch.randn(4, 4, requires_grad=True, device=device)

        # tensor subclasses as inputs
        x = TwoTensor(rand_tensor, rand_tensor.clone())
        y = TwoTensor(rand_tensor.clone(), rand_tensor.clone())

        fw_compiler = functools.partial(
            count_ops,
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 12 here
            # (4 matmul recompute and 4 mm ops per fwd matmul, so 4 + 2 * 4 = 12)
            # if we didn't enable selective checkpointing.
            freq=8,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_custom_rule(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing with a stateful custom policy that returns a bool
        # and counts mm calls per mode; it keeps mm results except the second one so the second mm
        # is a recompute candidate. Expects 2 forward mm and at least 4 backward mm (the
        # partitioner decides on the second); eager parity is checked.
        def _get_custom_policy(meta):
            # [FUNCTION DOC] HELPER: builds a policy using a meta dict to count mm calls per mode
            # (shadows the module-level helper of the same name).
            no_recompute_list = [
                torch.ops.aten.mm.default,
            ]

            def _custom_policy(mode, func, *args, **kwargs):
                # [FUNCTION DOC] HELPER: increments the per-mode mm counter in meta for every mm
                # call and returns True (save) for ops in the save list, except for the second mm,
                # which returns False.
                mm_count_key = f"{mode}_mm_count"
                if mm_count_key not in meta:
                    meta[mm_count_key] = 0
                if func == torch.ops.aten.mm.default:
                    meta[mm_count_key] += 1
                # Saves output of all compute ops, except second mm
                # (i.e. we will hint the partitioner to recompute second mm in backward pass)
                return func in no_recompute_list and not (
                    func == torch.ops.aten.mm.default and meta[mm_count_key] == 2
                )

            return _custom_policy

        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn creating a fresh meta dict and selective checkpoint
            # contexts using the counting policy.
            meta = {}
            return create_selective_checkpoint_contexts(_get_custom_policy(meta))

        def gn(x, y):
            # [FUNCTION DOC] HELPER: a chain of mm, mul by y, mm, mul by y followed
            # by two sigmoids.
            return torch.sigmoid(
                torch.sigmoid(torch.matmul(torch.matmul(x, y) * y, y) * y)
            )

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # Q: How do we come to this number 4?
            # A: We have 2 matmuls in the forward pass, each matmul contributes 2 `mm` ops in the backward pass,
            # so we have at least 4 `mm` ops in backward pass. It's "at least" because whether second matmul in
            # the forward pass is recomputed in the backward pass is up to the partitioner to decide.
            freq_ge=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_partial_ctx_fn(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing where the context_fn is a functools.partial
        # binding the no-recompute list (aten.mm). Expects 2 forward and 4 backward mm ops and
        # checks eager parity and plain-versus-checkpointed parity.
        def selective_checkpointing_context_fn(no_recompute_list):
            # [FUNCTION DOC] HELPER: builds selective checkpoint contexts from a policy saving the
            # ops in no_recompute_list.
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of two chained matmuls of x with y, times y.
            return torch.sigmoid(torch.matmul(torch.matmul(x, y), y)) * y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with a context_fn partial bound
            # to the mm list.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=functools.partial(
                    selective_checkpointing_context_fn, [torch.ops.aten.mm.default]
                ),
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freq=2,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            # We would've expected 6 here
            # (2 matmul recompute and 2 mm ops per fwd matmul, so 2 + 2 * 2 = 6)
            # if we didn't enable selective checkpointing.
            freq=4,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_outplace_op(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing saving mm and sigmoid for a function using out-of-
        # place ops (selu, relu). Expects forward counts of 2 mm and 1 sigmoid and backward counts
        # of 4 mm and 0 sigmoid (sigmoid is not recomputed), and checks parity with eager.
        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn saving aten.mm and aten.sigmoid.
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list),
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of selu of two chained matmuls, followed by relu.
            return torch.sigmoid(torch.selu(torch.matmul(torch.matmul(x, y), y))).relu()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_list_ops(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing with an empty save list (everything recomputed) on
        # a function whose op takes a list of tensors (cat then sin). Expects one cat in the
        # forward and one in the backward graph and checks parity with eager.
        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn with an empty save list, so
            # everything is recomputed.
            # recompute everything
            no_recompute_list = []
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: concatenates x and y and applies sin.
            return torch.cat([x, y]).sin()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freqs=[1],
            ops=[torch.ops.aten.cat.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[1],
            ops=[torch.ops.aten.cat.default],
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @unittest.skip(
        "In-place op support in selective checkpointing + torch.compile "
        "requires TorchDispatchMode + torch.compile work to complete"
    )
    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_inplace_op(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing with in-place ops (selu_, relu_) saving mm and
        # sigmoid; expects 2 mm and 1 sigmoid forward and 4 mm and 0 sigmoid backward. Currently
        # skipped: in-place op support needs TorchDispatchMode plus torch.compile work, and
        # skipped on Windows.
        def selective_checkpointing_context_fn():
            # [FUNCTION DOC] HELPER: context_fn saving aten.mm and aten.sigmoid.
            no_recompute_list = [
                torch.ops.aten.mm.default,
                torch.ops.aten.sigmoid.default,
            ]
            return create_selective_checkpoint_contexts(
                _get_custom_policy(no_recompute_list=no_recompute_list)
            )

        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of in-place selu on chained matmuls, followed
            # by in-place relu.
            return torch.sigmoid(
                torch.selu_(torch.matmul(torch.matmul(x, y), y))
            ).relu_()

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=selective_checkpointing_context_fn,
            )

        x = torch.randn(4, 4, requires_grad=True, device=device)
        y = torch.randn(4, 4, requires_grad=True, device=device)

        fw_compiler = functools.partial(
            count_ops,
            freqs=[2, 1],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[4, 0],
            ops=[torch.ops.aten.mm.default, torch.ops.aten.sigmoid.default],
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        self._validate(fn, backend, x, y)
        self._compare_orig_and_checkpointed_fns(gn, fn, x, y)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @torch._inductor.config.patch(fallback_random=True)
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_random_op(self, device, partition_fn):
        # [FUNCTION DOC] Selective checkpointing (sigmoid saved) with a dropout in the region, run
        # for preserve_rng_state True and False under inductor fallback_random. Expects forward 2
        # sigmoid and 1 native_dropout and backward 0 sigmoid and 1 native_dropout (dropout
        # recomputed). Output/gradient comparison with eager is skipped when preserve_rng_state is
        # False, since compile always preserves RNG state.
        for preserve_rng_state in [True, False]:

            def selective_checkpointing_context_fn():
                # [FUNCTION DOC] HELPER: context_fn with a policy saving aten.sigmoid.
                no_recompute_list = [
                    torch.ops.aten.sigmoid.default,
                ]
                return create_selective_checkpoint_contexts(
                    _get_custom_policy(no_recompute_list=no_recompute_list)
                )

            def gn(x):
                # [FUNCTION DOC] HELPER: sigmoid of dropout (p 0.5, training) of sigmoid of x.
                return torch.sigmoid(torch.dropout(torch.sigmoid(x), p=0.5, train=True))

            def fn(x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the SAC context_fn and
                # the loop's preserve_rng_state value.
                return torch.utils.checkpoint.checkpoint(
                    gn,
                    x,
                    use_reentrant=False,
                    # Regardless of whether `preserve_rng_state` is True or False,
                    # we will always preserve RNG state when using `torch.compile`.
                    preserve_rng_state=preserve_rng_state,
                    context_fn=selective_checkpointing_context_fn,
                )

            x = torch.randn(4, 4, requires_grad=True, device=device)

            fw_compiler = functools.partial(
                count_ops,
                freqs=[2, 1],
                ops=[
                    torch.ops.aten.sigmoid.default,
                    torch.ops.aten.native_dropout.default,
                ],
            )
            bw_compiler = functools.partial(
                count_ops,
                # NOTE: This unit test expects `dropout` to be recomputed (notice the count for `native_dropout` is 1).
                freqs=[0, 1],
                ops=[
                    torch.ops.aten.sigmoid.default,
                    torch.ops.aten.native_dropout.default,
                ],
            )
            backend = aot_autograd(
                fw_compiler=fw_compiler,
                bw_compiler=bw_compiler,
                partition_fn=partition_fn,
            )

            # NOTE: when `preserve_rng_state` is False, gradient will mismatch between torch.compile and eager,
            # because eager version doesn't preserve RNG state while torch.compile still does.
            # Hence when `preserve_rng_state` is False, we skip the output and gradient comparison
            # between torch.compile and eager.
            self._validate(fn, backend, x, skip_check=not preserve_rng_state)
            self._compare_orig_and_checkpointed_fns(gn, fn, x)

    @requires_gpu_and_triton
    @unittest.skipIf(IS_WINDOWS, "torch.compile doesn't work with windows")
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_invalid_context(self, partition_fn):
        # [FUNCTION DOC] Checks the error for a context_fn that returns two objects that are not
        # TorchDispatchModes (_invalid_context_gen): running _validate must raise an exception
        # matching must generate a tuple of two TorchDispatchModes. Takes no device argument;
        # inputs are CPU tensors.
        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of matmul of x and y, times y.
            return torch.sigmoid(torch.matmul(x, y)) * y

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn with the invalid context_fn.
            return torch.utils.checkpoint.checkpoint(
                gn,
                x,
                y,
                use_reentrant=False,
                context_fn=_invalid_context_gen,
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        fw_compiler = functools.partial(
            count_ops,
            freq=1,
            op=torch.ops.aten.mm.default,
        )
        bw_compiler = functools.partial(
            count_ops,
            freq_ge=2,
            op=torch.ops.aten.mm.default,
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )
        with self.assertRaisesRegex(
            Exception, "must generate a tuple of two `TorchDispatchMode`s"
        ):
            self._validate(fn, backend, x, y)

    @requires_gpu_and_triton
    @parametrize(
        "partition_fn",
        [
            min_cut_rematerialization_partition,
            default_partition,
        ],
    )
    def test_compile_selective_checkpoint_parametrization(self, partition_fn):
        # [FUNCTION DOC] Applies a selective-checkpointed parametrization (sigmoid of x times x,
        # with mul and sigmoid marked MUST_RECOMPUTE and all else MUST_SAVE) to the weight of a
        # bias-free Linear(16, 16) via nn.utils.parametrize, compiles fullgraph with aot_autograd
        # and compares output and input gradient with eager. Forward expects 1 mul and 1 sigmoid;
        # backward expects 1 sigmoid and 2 mul (3 with default_partition, which has no CSE).
        def sac_policy():
            # [FUNCTION DOC] HELPER: context_fn that builds selective checkpoint contexts from the
            # policy produced by _recomp_policy.
            def _recomp_policy():
                # [FUNCTION DOC] HELPER: returns the policy callable that recomputes mul.Tensor and
                # sigmoid and saves every other op.
                def _custom_policy(ctx, func, *args, **kwargs):
                    # [FUNCTION DOC] HELPER: maps aten.mul.Tensor and aten.sigmoid to
                    # MUST_RECOMPUTE and every other op to MUST_SAVE.
                    to_recompute = func in {
                        torch.ops.aten.mul.Tensor,
                        torch.ops.aten.sigmoid.default,
                    }
                    return (
                        CheckpointPolicy.MUST_RECOMPUTE
                        if to_recompute
                        else CheckpointPolicy.MUST_SAVE
                    )

                return _custom_policy

            return create_selective_checkpoint_contexts(_recomp_policy())

        class Parametrization(torch.nn.Module):
            # [CLASS DOC] HELPER: parametrization module that computes sigmoid of x squared,
            # evaluated under non-reentrant checkpoint with the selective sac_policy context_fn.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: calls the base constructor only.
                super().__init__()

            def parametrization(self, x):
                # [FUNCTION DOC] HELPER: returns sigmoid of the elementwise square of x.
                return torch.sigmoid(torch.mul(x, x))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs the parametrization under non-reentrant checkpoint
                # with the sac_policy context_fn and returns its result.
                return checkpoint(
                    self.parametrization, x, use_reentrant=False, context_fn=sac_policy
                )

        def apply_parametrization(model):
            # [FUNCTION DOC] HELPER: for every submodule of model, replaces each directly owned
            # parameter by a fresh nn.Parameter and registers a Parametrization on it (unsafe
            # mode); mutates and returns model.
            modules = list(model.modules())

            for mod in modules:
                params_dict = dict(mod.named_parameters(recurse=False))
                for p_name, p in params_dict.items():
                    mod.register_parameter(p_name, nn.Parameter(p))
                    nn.utils.parametrize.register_parametrization(
                        mod, p_name, Parametrization(), unsafe=True
                    )

            return model

        class MLPModule(nn.Module):
            # [CLASS DOC] HELPER: tiny model with one bias-free Linear(16, 16) layer, seeded with
            # torch.manual_seed(5) at construction.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: seeds the global RNG with 5 (side effect) and creates
                # the linear layer.
                super().__init__()
                torch.manual_seed(5)
                self.net1 = nn.Linear(16, 16, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the linear layer to x.
                return self.net1(x)

            def reset_parameters(self):
                # [FUNCTION DOC] HELPER: re-initializes the parameters of the linear layer.
                self.net1.reset_parameters()

        fw_compiler = functools.partial(
            count_ops,
            freqs=[1, 1],
            ops=[torch.ops.aten.mul.Tensor, torch.ops.aten.sigmoid.default],
        )
        bw_compiler = functools.partial(
            count_ops,
            freqs=[
                # 1 from mul recompute, 1 from mul backward
                # w/o CSE, we have one extra mul
                3 if partition_fn is default_partition else 2,
                1,
            ],
            ops=[torch.ops.aten.mul.Tensor, torch.ops.aten.sigmoid.default],
        )

        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=bw_compiler,
            partition_fn=partition_fn,
        )

        model = MLPModule()
        model = apply_parametrization(model)
        model_compiled = torch.compile(
            copy.deepcopy(model), backend=backend, fullgraph=True
        )
        input = torch.randn(8, 16, requires_grad=True)
        input_compiled = copy.deepcopy(input)

        out = model(input)
        out.sum().backward()
        out_compiled = model_compiled(input_compiled)
        out_compiled.sum().backward()

        self.assertEqual(out, out_compiled)
        self.assertEqual(input.grad, input_compiled.grad)

    @skipIfXpu(msg="XPU has some known limitations on flash attention.")
    @requires_gpu_and_triton
    def test_autocast_flash_attention(self, device):
        # [FUNCTION DOC] Runs a reentrant-checkpointed aten efficient scaled-dot-product attention
        # call under autocast and compares the aot_eager_decomp_partition compiled output to eager
        # with the same seed. Skipped on XPU.
        # KNOWN ISSUE: the test name says flash attention but the op used is the memory-
        # efficient attention op.
        def fn(primals_1, primals_2, primals_3):
            # [FUNCTION DOC] HELPER: calls the efficient scaled-dot-product attention aten op with
            # inputs, no mask, compute_log_sumexp true and a fixed scale, and returns the attention
            # output (first element).
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                primals_1, primals_2, primals_3, None, True, scale=0.17677669529663687
            )[0]

        def gn(*args):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of fn on all given arguments.
            return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=True)

        with torch.autocast(device_type=device):
            x = torch.randn(4, 2, 16, 32, device=device, requires_grad=True)
            y = torch.randn(4, 2, 16, 32, device=device, requires_grad=True)
            z = torch.randn(4, 2, 16, 32, device=device, requires_grad=True)
            args = (x, y, z)

            torch.manual_seed(0)
            ref = gn(*args)

            opt_gn = torch.compile(gn, backend="aot_eager_decomp_partition")
            torch.manual_seed(0)
            res = opt_gn(*args)
            self.assertEqual(ref, res)

    @requires_gpu_and_triton
    @unittest.skipIf(
        not PLATFORM_SUPPORTS_MEM_EFF_ATTENTION,
        "This platform doesn't support efficient attention",
    )
    def test_compile_checkpoint_mem_eff_attention_no_dropout(self, device):
        # [FUNCTION DOC] Checks checkpointing around a block that contains a reduce-overhead
        # compiled efficient-attention call with no dropout. Verifies the op's philox seed and
        # offset outputs are zero scalars and that the checkpointed block gives the same output and
        # input gradient as eager. Skipped when the platform lacks memory-efficient attention.
        def eager_attn(x):
            # [FUNCTION DOC] HELPER: efficient attention with q, k, v all x, no mask and dropout 0;
            # returns the attention output.
            return torch.ops.aten._scaled_dot_product_efficient_attention.default(
                x, x, x, None, True, dropout_p=0.0
            )[0]

        @torch.compile(mode="reduce-overhead")  # noqa: UNSPECIFIED_BACKEND
        def attn(x):
            # [FUNCTION DOC] HELPER: torch.compile (reduce-overhead mode) wrapper
            # that returns eager_attn(x).
            return eager_attn(x)

        def block(x):
            # [FUNCTION DOC] HELPER: returns x plus the compiled attention of x.
            return x + attn(x)

        x_ref = torch.randn(3, 1, 77, 64, device=device, requires_grad=True)
        _, _, philox_seed, philox_offset = (
            torch.ops.aten._scaled_dot_product_efficient_attention.default(
                x_ref, x_ref, x_ref, None, True, dropout_p=0.0
            )
        )
        self.assertEqual(philox_seed, torch.zeros((), device=device, dtype=torch.int64))
        self.assertEqual(
            philox_offset, torch.zeros((), device=device, dtype=torch.int64)
        )

        ref = x_ref + eager_attn(x_ref)
        ref.sum().backward()

        x = x_ref.detach().clone().requires_grad_(True)
        y = checkpoint(block, x, use_reentrant=False)
        y.sum().backward()

        self.assertEqual(ref, y)
        self.assertEqual(x_ref.grad, x.grad)

    @requires_gpu_and_triton
    def test_error_msg(self, device):
        # [FUNCTION DOC] Checks that a graph break inside a checkpointed module under
        # fullgraph compilation raises torch._dynamo.exc.Unsupported with a message about a
        # user-inserted graph break.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward applies sin, an explicit dynamo graph
            # break and cos.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: calls the base constructor only.
                super().__init__()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: sin, graph break, cos of x; returns the result.
                x = torch.sin(x)
                torch._dynamo.graph_break()
                x = torch.cos(x)
                return x

        mod = MockModule().to(device)

        def fn(x):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of the module on x.
            return torch.utils.checkpoint.checkpoint(mod, x, use_reentrant=True)

        x = torch.randn(4, 4).to(device)
        opt_fn = torch.compile(fn, fullgraph=True, backend="aot_eager_decomp_partition")
        with self.assertRaisesRegex(
            torch._dynamo.exc.Unsupported, "User-inserted graph break"
        ):
            opt_fn(x)

    @requires_gpu_and_triton
    def test_list_inputs(self, device):
        # [FUNCTION DOC] Checks that checkpointed modules taking a list input and returning a
        # nested structure (tensor and list) compile fullgraph with aot_eager_decomp_partition and
        # give the same result as eager.
        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward takes x and a list ys and returns x together
            # with a list of cos of both list elements; the sin result is unused.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: calls the base constructor only.
                super().__init__()

            def forward(self, x, ys):
                # [FUNCTION DOC] HELPER: returns a tuple of x and the list of cosines of ys
                # elements 0 and 1.
                a = torch.sin(x)  # noqa: F841
                b = torch.cos(ys[0])
                c = torch.cos(ys[1])
                return (x, [b, c])

        mod = MockModule().to(device)

        def fn(x, ys):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of the module on x and the list ys.
            return torch.utils.checkpoint.checkpoint(mod, x, ys, use_reentrant=True)

        x = torch.randn(4, 4).to(device)
        y = torch.randn(4, 4).to(device)
        z = torch.randn(4, 4).to(device)
        ref = fn(x, [y, z])
        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        res = opt_fn(x, [y, z])
        self.assertEqual(ref, res)

    @skipIfXpu(msg="https://github.com/intel/torch-xpu-ops/issues/3393")
    @requires_gpu_and_triton
    def test_pattern_matcher(self, device):
        # [FUNCTION DOC] Compiles with inductor a reentrant-checkpointed half-precision attention
        # (matmul, scale, softmax, matmul) after a sin and inspects the captured post-AOT graphs.
        # Expects one fused SDPA op in the forward, no sin in the backward (tags percolate), and
        # the SDPA op recomputed once in the backward; the op is cudnn attention on preferred
        # platforms, flash attention otherwise. Skipped on XPU.
        # KNOWN ISSUE: count_ops returns the graph module (always truthy), so the assertTrue
        # wrappers are redundant; the real check is the AssertionError raised inside count_ops.
        # Check that the sdpa op is recomputed in the backward graph
        # tests percolate_tags

        @checkpoint_wrapper
        def dot_prod_attention(
            # [FUNCTION DOC] HELPER: reentrant-checkpointed (via checkpoint_wrapper) scaled
            # dot-product attention written with matmul, scaling by 1/sqrt(last key dim),
            # softmax and matmul.
            query: torch.Tensor, key: torch.Tensor, value: torch.Tensor
        ) -> torch.Tensor:
            return (
                torch.matmul(query, key.transpose(-2, -1))
                .mul(1.0 / math.sqrt(key.shape[-1]))
                .softmax(dim=-1)
                .matmul(value)
            )

        def fn(query, key, value):
            # [FUNCTION DOC] HELPER: attention on sin(query), key, value.
            # Checks that sin is not recomputed in the backward graph
            return dot_prod_attention(query.sin(), key, value)

        tensor_shape = (4, 2, 16, 32)
        dtype = torch.float16
        args1 = [
            torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True),
            torch.randn(tensor_shape, device=device, dtype=dtype, requires_grad=True),
        ]

        # Save the AOT graphs
        aot_graphs = []
        from torch._inductor import compile_fx

        def debug_compile_fx_inner(graph, example_inputs, *args, **kwargs):
            # [FUNCTION DOC] HELPER: inner_compile hook that records each graph in aot_graphs and
            # delegates to compile_fx_inner, returning its result.
            aot_graphs.append(graph)
            return compile_fx.compile_fx_inner(graph, example_inputs, *args, **kwargs)

        backend = functools.partial(
            compile_fx.compile_fx, inner_compile=debug_compile_fx_inner
        )

        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        opt_fn(*args1).sum().backward()

        fwd_graph = aot_graphs[0]
        # Determine which fused attention backend is expected based on the
        # prioritization logic in sdp_utils.cpp:check_prefer_cudnn_attention.
        dprops = torch.get_device_module(device_type).get_device_properties(device)
        cudnn_version = (
            torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 0
        )
        prefer_cudnn = (
            cudnn_version > 91500 and dprops.major in (9, 10) and dprops.minor in (0, 3)
        )
        if prefer_cudnn and torch.version.cuda and TEST_CUDA:
            sdpa_op = torch.ops.aten._scaled_dot_product_cudnn_attention.default
        else:
            sdpa_op = torch.ops.aten._scaled_dot_product_flash_attention.default
        self.assertTrue(count_ops(fwd_graph, [], freq=1, op=sdpa_op))
        bwd_graph = aot_graphs[1]
        # Check that sin is not recomputed in the backward graph - checks percolate tags
        self.assertTrue(count_ops(bwd_graph, [], freq=0, op=torch.ops.aten.sin.default))
        # Check that the sdpa op is recomputed in the backward graph
        self.assertTrue(count_ops(bwd_graph, [], freq=1, op=sdpa_op))

    @requires_distributed()
    @requires_gpu_and_triton
    def test_distributed_utils_checkpoint_wrapper(self):
        # [FUNCTION DOC] Checks that a module wrapped with the distributed
        # checkpoint_wrapper (torch.distributed.algorithms._checkpoint) compiles fullgraph
        # with aot_eager_decomp_partition and matches eager output. Requires distributed and
        # a GPU with triton.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            checkpoint_wrapper as dist_checkpoint_wrapper,
        )

        class MockModule(torch.nn.Module):
            # [CLASS DOC] HELPER: module with Linear(4, 4) and an integer attribute c=2; forward is
            # sin, linear, cos, then multiplied by c.
            def __init__(self) -> None:
                # [FUNCTION DOC] HELPER: creates the linear layer and sets c to 2.
                super().__init__()
                self.linear = torch.nn.Linear(4, 4)
                self.c = 2

            def forward(self, x):
                # [FUNCTION DOC] HELPER: sin, linear, cos, then scale by c; returns the result.
                x = torch.sin(x)
                x = self.linear(x)
                x = torch.cos(x)
                return x * self.c

        mod = dist_checkpoint_wrapper(MockModule())
        x = torch.randn(4, 4)
        ref = mod(x)
        opt_mod = torch.compile(
            mod, backend="aot_eager_decomp_partition", fullgraph=True
        )
        res = opt_mod(x)
        self.assertEqual(ref, res)

    @requires_distributed()
    @requires_gpu_and_triton
    def test_dynamo_does_not_trace_getattr_as_top_frame(self):
        # [FUNCTION DOC] Checks that dynamo does not trace CheckpointWrapper getattr as the top
        # frame: wraps a Sequential in a CheckpointWrapper with an extra attribute a on the inner
        # module, compiles a function using the wrapper and its attribute a with fullgraph and the
        # aot_eager_decomp_partition counting backend, and asserts equality with eager. Requires
        # distributed and a GPU with triton.
        from torch.distributed.algorithms._checkpoint.checkpoint_wrapper import (
            CheckpointWrapper,
        )

        cnt = CompileCounterWithBackend("aot_eager_decomp_partition")

        lin = torch.nn.Linear(1, 1)
        mod = torch.nn.Sequential(lin, lin)
        mod = CheckpointWrapper(mod)
        mod._checkpoint_wrapped_module.a = torch.ones(1, 1)

        def fn(x):
            # [FUNCTION DOC] HELPER: returns the wrapped module applied to x, multiplied by
            # the attribute a.
            return mod(x) * mod.a

        opt_fn = torch.compile(fn, backend=cnt, fullgraph=True)
        x = torch.randn(1, 1)

        self.assertEqual(opt_fn(x), fn(x))

    def test_return_same_element_twice(self):
        # [FUNCTION DOC] Checks that a checkpointed function returning the same tensor twice
        # compiles fullgraph (reentrant) with equal outputs, and that the captured dynamo graph
        # exposes only one getitem output from the tag_activation_checkpoint higher-order op,
        # compared to an expected printed graph.
        def gn(x):
            # [FUNCTION DOC] HELPER: returns sin(x) twice as a tuple.
            y = torch.sin(x)
            return y, y

        def fn(x):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of gn on x.
            return torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=True)

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = True);  wrap_body_0 = l_x_ = None
        getitem: "f32[4, 4]" = tag_activation_checkpoint[0];  tag_activation_checkpoint = None
        return (getitem,)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[4, 4]"):
            y: "f32[4, 4]" = torch.sin(l_x_);  l_x_ = None
            return (y,)
""",
        )

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_nonlocal_mutation(self):
        # [FUNCTION DOC] With skip_fwd_side_effects_in_bwd_under_checkpoint enabled, a nonlocal
        # counter incremented inside a checkpointed function is incremented twice in eager (the
        # forward and the recompute) but once under compile, since the recompute no longer replays
        # the side effect.
        counter = 0

        def gn(x):
            # [FUNCTION DOC] HELPER: increments the enclosing counter variable (nonlocal side
            # effect) and returns sin(x).
            nonlocal counter
            counter += 1
            return torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: reentrant checkpoint of gn on x.
            return torch.utils.checkpoint.checkpoint(gn, x, use_reentrant=True)

        x = torch.randn(4, 4, requires_grad=True)
        fn(x).sum().backward()
        # The mutation is reapplied in the backward as well
        self.assertEqual(counter, 2)
        counter = 0

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        opt_fn(x).sum().backward()
        # The mutation is not reapplied in the backward because the flag was on.
        self.assertEqual(counter, 1)

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_nonlocal_list_mutation(self):
        # [FUNCTION DOC] With skip_fwd_side_effects_in_bwd_under_checkpoint enabled, checks that
        # a list mutation (append of an output) inside a non-reentrant checkpoint compiles
        # fullgraph with aot_eager_decomp_partition and gives the same returned value and list
        # element as eager.
        def gn(x, z):
            # [FUNCTION DOC] HELPER: appends sin(x) to the list z (side effect) and returns the cos
            # of sin of a chained matmul of x, together with that sin.
            out = x.sin()
            z.append(out)
            return torch.cos(torch.sin(torch.matmul(x, x) @ x)), out

        def fn(x):
            # [FUNCTION DOC] HELPER: creates an empty list, runs the non-reentrant checkpoint of gn
            # with x and the list, and returns the first output and the first list element.
            z = []

            out1, out2 = torch.utils.checkpoint.checkpoint(
                gn,
                x,
                z,
                use_reentrant=False,
            )

            return out1, z[0]

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        opt_fn = torch.compile(fn, backend="aot_eager_decomp_partition", fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_nonlocal_list_mutation_hidden(self):
        # [FUNCTION DOC] Variant of the list-mutation test where the mutated value is not directly
        # returned: gn returns two outputs plus mutates a list, fn indexes outputs by first and
        # last. Compiles fullgraph with the recording backend and checks eager parity, the captured
        # dynamo graph (one checkpoint op with three outputs) and the forward AOT graph against
        # expected text. skip_fwd_side_effects_in_bwd_under_checkpoint is enabled.
        def gn(x, z):
            # [FUNCTION DOC] HELPER: computes a chained matmul, appends sin(x) to list z (side
            # effect) and returns cos of sin of the matmul and sin(x).
            o = torch.matmul(x, x) @ x
            out = x.sin()
            z.append(out)
            return torch.cos(torch.sin(o)), torch.sin(x)

        def fn(x):
            # [FUNCTION DOC] HELPER: runs the non-reentrant checkpoint of gn with x and a fresh
            # list; returns the sum of the first and last outputs, and the list's first element.
            z = []

            outs = torch.utils.checkpoint.checkpoint(
                gn,
                x,
                z,
                use_reentrant=False,
            )
            out1 = outs[0]
            # Check that the extra output pytree handling is done properly
            out2 = outs[-1]

            return out1 + out2, z[0]

        x = torch.randn(4, 4, requires_grad=True)
        ref = fn(x)

        backend = AotEagerAndRecordGraphs()
        opt_fn = torch.compile(fn, backend=backend, fullgraph=True)
        res = opt_fn(x)
        self.assertEqual(ref[0], res[0])
        self.assertEqual(ref[1], res[1])

        self.assertExpectedInline(
            normalize_gm(backend.graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, L_x_: "f32[4, 4]"):
        l_x_ = L_x_

        wrap_body_0 = self.wrap_body_0
        tag_activation_checkpoint = torch.ops.higher_order.tag_activation_checkpoint(wrap_body_0, l_x_, use_reentrant = False);  wrap_body_0 = l_x_ = None
        getitem_3: "f32[4, 4]" = tag_activation_checkpoint[0]
        getitem_4: "f32[4, 4]" = tag_activation_checkpoint[1]

        add: "f32[4, 4]" = getitem_3 + getitem_4;  getitem_3 = getitem_4 = None

        getitem_5: "f32[4, 4]" = tag_activation_checkpoint[2];  tag_activation_checkpoint = None
        return (add, getitem_5)

    class wrap_body_0(torch.nn.Module):
        def forward(self, l_x_: "f32[4, 4]"):
            matmul: "f32[4, 4]" = torch.matmul(l_x_, l_x_)
            o: "f32[4, 4]" = matmul @ l_x_;  matmul = None

            out: "f32[4, 4]" = l_x_.sin()

            sin_1: "f32[4, 4]" = torch.sin(o);  o = None
            cos: "f32[4, 4]" = torch.cos(sin_1);  sin_1 = None
            sin_2: "f32[4, 4]" = torch.sin(l_x_);  l_x_ = None
            return (cos, sin_2, out)
""",
        )

        self.assertExpectedInline(
            normalize_gm(backend.fw_graphs[0].print_readable(print_output=False)),
            """\
class GraphModule(torch.nn.Module):
    def forward(self, primals_1: "f32[4, 4]"):
        mm: "f32[4, 4]" = torch.ops.aten.mm.default(primals_1, primals_1)
        mm_1: "f32[4, 4]" = torch.ops.aten.mm.default(mm, primals_1);  mm = None

        sin: "f32[4, 4]" = torch.ops.aten.sin.default(primals_1)

        sin_1: "f32[4, 4]" = torch.ops.aten.sin.default(mm_1);  mm_1 = None
        cos: "f32[4, 4]" = torch.ops.aten.cos.default(sin_1);  sin_1 = None
        sin_2: "f32[4, 4]" = torch.ops.aten.sin.default(primals_1)

        add: "f32[4, 4]" = torch.ops.aten.add.Tensor(cos, sin_2);  cos = sin_2 = None
        return (add, sin, primals_1)
""",
        )

    def test_frozen_dataclass_pytree_output(self):
        # [FUNCTION DOC] Registers two frozen dataclasses (InputNode and OutputNode) as
        # pytree nodes and checks that a non-reentrant checkpointed module taking and
        # returning them compiles fullgraph (aot_eager_decomp_partition) and gives the same
        # output and input gradient as eager, with the same seed. The pytree registrations
        # are removed in a finally block.
        import dataclasses

        from torch.utils import _pytree as pytree

        @dataclasses.dataclass(frozen=True)
        class InputNode:
            # [CLASS DOC] HELPER: frozen dataclass holding a single tensor field x; registered as
            # a pytree node.
            x: torch.Tensor

        @dataclasses.dataclass(frozen=True)
        class OutputNode:
            # [CLASS DOC] HELPER: frozen dataclass holding a single tensor field y; registered as
            # a pytree node.
            y: torch.Tensor

        pytree.register_dataclass(InputNode)
        pytree.register_dataclass(OutputNode)

        def cleanup():
            # [FUNCTION DOC] HELPER: removes the two test dataclasses from the pytree
            # supported-node, serialized-type and constant-node registries so they do not leak
            # into other tests.
            # Clean up pytree registrations to avoid leaking state to other tests.
            # We manually remove from the registries since _deregister_pytree_node
            # has issues with classes registered without serialized_type_name.
            for cls in [InputNode, OutputNode]:
                pytree.SUPPORTED_NODES.pop(cls, None)
                pytree.SUPPORTED_SERIALIZED_TYPES.pop(cls, None)
                pytree.CONSTANT_NODES.discard(cls)

        try:

            class TinyMLP(nn.Module):
                # [CLASS DOC] HELPER: MLP with Linear(4, 8) and Linear(8, 4); forward maps an
                # InputNode through linear, silu, linear to an OutputNode.
                def __init__(self):
                    # [FUNCTION DOC] HELPER: creates the two linear layers.
                    super().__init__()
                    self.fc1 = nn.Linear(4, 8)
                    self.fc2 = nn.Linear(8, 4)

                def forward(self, inp: InputNode) -> OutputNode:
                    # [FUNCTION DOC] HELPER: applies fc1, silu and fc2 to the x field and wraps the
                    # result in an OutputNode.
                    h = self.fc1(inp.x)
                    h = torch.nn.functional.silu(h)
                    y = self.fc2(h)
                    return OutputNode(y=y)

            mlp = TinyMLP()

            def checkpointed_forward(inp):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of the MLP forward on inp
                # with preserve_rng_state true.
                return torch.utils.checkpoint.checkpoint(
                    mlp.forward,
                    inp,
                    use_reentrant=False,
                    preserve_rng_state=True,
                )

            input_eager = InputNode(x=torch.randn(2, 4, requires_grad=True))
            torch.manual_seed(0)
            output_eager = checkpointed_forward(input_eager)
            output_eager.y.sum().backward()

            input_compiled = InputNode(
                x=input_eager.x.detach().clone().requires_grad_(True)
            )
            torch.manual_seed(0)
            compiled_fn = torch.compile(
                checkpointed_forward,
                fullgraph=True,
                backend="aot_eager_decomp_partition",
            )
            output_compiled = compiled_fn(input_compiled)
            output_compiled.y.sum().backward()

            self.assertEqual(output_eager.y, output_compiled.y)
            self.assertEqual(input_eager.x.grad, input_compiled.x.grad)
        finally:
            cleanup()

    def test_checkpoint_with_record_function(self):
        # [FUNCTION DOC] Checks that torch.profiler.record_function inside a checkpointed function
        # is allowed when capture_profiler_record_function is enabled: compiles with aot_autograd
        # (default partitioner), expects exactly one _record_function_enter_new op in the forward
        # graph and parity with eager.
        # Test that record_function ops are allowed inside checkpointed functions.
        # record_function is technically "impure" but safe to duplicate during
        # activation checkpointing recompute since it only sets up profiling spans.
        # This test verifies:
        # 1. No assertion error about impure ops in AC
        # 2. Forward graph contains record_function ops
        # 3. Code produces correct results
        def gn(x, y):
            # [FUNCTION DOC] HELPER: sigmoid of the matmul of x and y inside a record_function
            # span named matmul_region.
            with torch.profiler.record_function("matmul_region"):
                return torch.sigmoid(torch.matmul(x, y))

        def fn(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of gn on sin(x) and y.
            return torch.utils.checkpoint.checkpoint(
                gn, torch.sin(x), y, use_reentrant=False
            )

        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        # Verify record_function_enter_new appears in forward graph
        fw_compiler = functools.partial(
            count_ops, freq=1, op=torch.ops.profiler._record_function_enter_new.default
        )
        backend = aot_autograd(
            fw_compiler=fw_compiler,
            bw_compiler=nop,
            partition_fn=default_partition,
        )
        # Enable capture_profiler_record_function to trace record_function ops
        with torch._dynamo.config.patch(capture_profiler_record_function=True):
            self._validate(fn, backend, x, y)

    def _get_sac_annotations(self, checkpointed_fn, policy_fn, decompositions=None):
        # [FUNCTION DOC] HELPER: compiles a non-reentrant checkpoint of checkpointed_fn on a random
        # 4-element input with a selective-checkpoint context_fn built from policy_fn, using
        # aot_autograd with no-op compilers and a partition function that records, for each
        # call_function node of the joint graph with a recompute annotation, a line of node name,
        # target and policy name before delegating to the min-cut partitioner. Resets dynamo first,
        # runs forward and backward and returns the recorded lines joined by newlines. Optional
        # decompositions are passed to aot_autograd.
        annotations = []

        def capture_partition(joint_gm, joint_args, **kwargs):
            # [FUNCTION DOC] HELPER: partition_fn that appends one annotation line per annotated
            # call_function node (name, target, recompute policy name) to the enclosing list and
            # returns the min-cut rematerialization partition of the joint graph.
            for node in joint_gm.graph.nodes:
                if node.op == "call_function":
                    recompute = node.meta.get("recompute", None)
                    if recompute is not None:
                        annotations.append(
                            f"{node.name}: {node.target} -> {recompute.name}"
                        )
            return min_cut_rematerialization_partition(joint_gm, joint_args, **kwargs)

        backend = aot_autograd(
            fw_compiler=nop,
            bw_compiler=nop,
            partition_fn=capture_partition,
            decompositions=decompositions,
        )

        def fn(x):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of checkpointed_fn on x with
            # the selective context_fn.
            return checkpoint(
                checkpointed_fn,
                x,
                use_reentrant=False,
                context_fn=functools.partial(
                    create_selective_checkpoint_contexts, policy_fn
                ),
            )

        x = torch.randn(4, requires_grad=True)
        torch._dynamo.reset()
        compiled = torch.compile(fn, backend=backend)
        out = compiled(x)
        out.sum().backward()
        return "\n".join(annotations)

    def test_pre_mode_decomp_has_sac_ignored_ops(self):
        # [FUNCTION DOC] Verifies, via expected text, the SAC annotations when the checkpointed
        # region (an allow_in_graph function using sin, detach, add, cos) is processed with the
        # policy saving sin, add and cos: the detach node (an op SAC ignores) is annotated
        # PREFER_RECOMPUTE while the others are MUST_SAVE.
        SAVE_OPS = {
            torch.ops.aten.sin.default,
            torch.ops.aten.add.Tensor,
            torch.ops.aten.cos.default,
        }

        def policy_fn(ctx, func, *args, **kwargs):
            # [FUNCTION DOC] HELPER: policy returning MUST_SAVE for ops in SAVE_OPS
            # and PREFER_RECOMPUTE otherwise.
            if func in SAVE_OPS:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        @torch._dynamo.allow_in_graph
        def op_with_detach(x):
            # [FUNCTION DOC] HELPER: allow_in_graph function: sin of x, adds its detached
            # copy, then cos.
            a = x.sin()
            out = a.detach() + a
            out = out.cos()
            return out

        self.assertExpectedInline(
            self._get_sac_annotations(op_with_detach, policy_fn),
            """\
sin: aten.sin.default -> MUST_SAVE
detach_1: aten.detach.default -> PREFER_RECOMPUTE
add: aten.add.Tensor -> MUST_SAVE
cos: aten.cos.default -> MUST_SAVE""",
        )

    def test_post_mode_decomp(self):
        # [FUNCTION DOC] Verifies, via expected text, the SAC annotations when silu is
        # decomposed after the SAC mode runs (inductor decomposition table): ops produced by
        # decomposing silu (neg, exp, add, div) inherit MUST_SAVE from the silu policy while sin
        # and cos are PREFER_RECOMPUTE.
        from torch._inductor.compile_fx import select_decomp_table

        def policy_fn(ctx, func, *args, **kwargs):
            # [FUNCTION DOC] HELPER: policy returning MUST_SAVE for silu and PREFER_RECOMPUTE for
            # all other ops.
            if func == torch.ops.aten.silu.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: sin, silu, then cos of x.
            x = x.sin()
            x = torch.nn.functional.silu(x)
            x = x.cos()
            return x

        self.assertExpectedInline(
            self._get_sac_annotations(
                fn, policy_fn, decompositions=select_decomp_table
            ),
            """\
sin: aten.sin.default -> PREFER_RECOMPUTE
neg: aten.neg.default -> MUST_SAVE
exp: aten.exp.default -> MUST_SAVE
add: aten.add.Tensor -> MUST_SAVE
div: aten.div.Tensor -> MUST_SAVE
cos: aten.cos.default -> PREFER_RECOMPUTE""",
        )

    def test_multi_output_op(self):
        # [FUNCTION DOC] Verifies, via expected text, SAC annotations for a multi-output op: topk
        # is MUST_SAVE and both getitem nodes extracted from it inherit MUST_SAVE, while sin, sum
        # and cos are PREFER_RECOMPUTE.
        def policy_fn(ctx, func, *args, **kwargs):
            # [FUNCTION DOC] HELPER: policy returning MUST_SAVE for topk
            # and PREFER_RECOMPUTE otherwise.
            if func == torch.ops.aten.topk.default:
                return CheckpointPolicy.MUST_SAVE
            return CheckpointPolicy.PREFER_RECOMPUTE

        def fn(x):
            # [FUNCTION DOC] HELPER: sin, top-2 values, their sum, then cos; the
            # indices are unused.
            x = x.sin()
            vals, idxs = torch.topk(x, k=2)
            out = vals.sum()
            out = out.cos()
            return out

        self.assertExpectedInline(
            self._get_sac_annotations(fn, policy_fn),
            """\
sin: aten.sin.default -> PREFER_RECOMPUTE
topk: aten.topk.default -> MUST_SAVE
getitem: <built-in function getitem> -> MUST_SAVE
getitem_1: <built-in function getitem> -> MUST_SAVE
sum_1: aten.sum.default -> PREFER_RECOMPUTE
cos: aten.cos.default -> PREFER_RECOMPUTE""",
        )

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_region_activation_memory_budget_reduces_act_mem(self):
        # [FUNCTION DOC] Compares retained activation memory (CUDA allocator requested bytes
        # between forward and backward) of a 4-layer Linear plus relu stack compiled with
        # aot_eager. Without a budget it must be greater than zero; with
        # region_activation_memory_budget(0.0) around the forward it must be exactly zero
        # (everything recomputed). Requires CUDA.
        N, NUM_LAYERS = 1000, 4

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: stack of NUM_LAYERS Linear(N, N) layers followed by relu; if
            # budget is not None the forward runs under region_activation_memory_budget(budget),
            # and returns the sum of the output.
            def __init__(self, budget=None):
                # [FUNCTION DOC] HELPER: stores the optional budget and creates the
                # ModuleList of linears.
                super().__init__()
                self.budget = budget
                self.linears = torch.nn.ModuleList(
                    [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs each linear then relu on x, inside the region
                # activation memory budget context when a budget was given, and returns the sum.
                if self.budget is not None:
                    with torch.autograd.graph.region_activation_memory_budget(
                        self.budget
                    ):
                        for linear in self.linears:
                            x = linear(x).relu()
                        return x.sum()
                else:
                    for linear in self.linears:
                        x = linear(x).relu()
                    return x.sum()

        def get_act_mem(f):
            # [FUNCTION DOC] HELPER: runs f once with backward as warm-up, records current
            # requested CUDA bytes, runs f again and measures the increase in requested bytes
            # (activation memory) before calling backward, then returns that increase.
            out = f()
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            act_mem = (
                torch.cuda.memory_stats()["requested_bytes.all.current"] - start_mem
            )
            out.backward()
            return act_mem

        x = torch.randn(N, N, device="cuda")

        torch._dynamo.reset()
        compiled = torch.compile(Model().cuda(), backend="aot_eager")
        self.assertGreater(get_act_mem(lambda: compiled(x)), 0)

        torch._dynamo.reset()
        compiled = torch.compile(Model(budget=0.0).cuda(), backend="aot_eager")
        self.assertEqual(get_act_mem(lambda: compiled(x)), 0)

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    def test_region_activation_memory_budget_per_region(self):
        # [FUNCTION DOC] Checks that graphs separated by an explicit graph break can have different
        # region activation memory budgets. Compiles four models with budgets (1.0 or 0.0) per half
        # and asserts activation memory ordering: both save > either one recomputing > both
        # recomputing. Requires CUDA.
        """Different graphs (separated by a graph break) can have different
        memory budgets."""
        N, NUM_LAYERS = 1000, 2

        class Model(torch.nn.Module):
            # [CLASS DOC] HELPER: two stacks of Linear(N, N) plus relu layers; the first half runs
            # under budget_a, a dynamo graph break follows, then the second half under budget_b;
            # returns the sum.
            def __init__(self, budget_a, budget_b):
                # [FUNCTION DOC] HELPER: stores both budgets and creates the two
                # ModuleLists of linears.
                super().__init__()
                self.budget_a = budget_a
                self.budget_b = budget_b
                self.linears_a = torch.nn.ModuleList(
                    [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
                )
                self.linears_b = torch.nn.ModuleList(
                    [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
                )

            def forward(self, x):
                # [FUNCTION DOC] HELPER: first stack under budget_a, graph break, second stack
                # under budget_b; returns the sum of the output.
                with torch.autograd.graph.region_activation_memory_budget(
                    self.budget_a
                ):
                    for linear in self.linears_a:
                        x = linear(x).relu()
                torch._dynamo.graph_break()
                with torch.autograd.graph.region_activation_memory_budget(
                    self.budget_b
                ):
                    for linear in self.linears_b:
                        x = linear(x).relu()
                    return x.sum()

        def get_act_mem(f):
            # [FUNCTION DOC] HELPER: warm-up forward and backward, then measures the increase in
            # CUDA requested bytes caused by one forward (retained activations), calls backward and
            # returns the measurement.
            out = f()
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            act_mem = (
                torch.cuda.memory_stats()["requested_bytes.all.current"] - start_mem
            )
            out.backward()
            return act_mem

        x = torch.randn(N, N, device="cuda")

        torch._dynamo.reset()
        both_save = torch.compile(Model(1.0, 1.0).cuda(), backend="aot_eager")
        mem_both_save = get_act_mem(lambda: both_save(x))

        torch._dynamo.reset()
        a_recomp = torch.compile(Model(0.0, 1.0).cuda(), backend="aot_eager")
        mem_a_recomp = get_act_mem(lambda: a_recomp(x))

        torch._dynamo.reset()
        b_recomp = torch.compile(Model(1.0, 0.0).cuda(), backend="aot_eager")
        mem_b_recomp = get_act_mem(lambda: b_recomp(x))

        torch._dynamo.reset()
        both_recomp = torch.compile(Model(0.0, 0.0).cuda(), backend="aot_eager")
        mem_both_recomp = get_act_mem(lambda: both_recomp(x))

        # Both save > either one recomputing > both recomputing
        self.assertGreater(mem_both_save, mem_a_recomp)
        self.assertGreater(mem_both_save, mem_b_recomp)
        self.assertGreater(mem_a_recomp, mem_both_recomp)
        self.assertGreater(mem_b_recomp, mem_both_recomp)

    def test_region_activation_memory_budget_validation(self):
        # [FUNCTION DOC] Checks argument validation of region_activation_memory_budget: a bool or
        # string raises TypeError mentioning a float; values outside 0..1 (2.0 and -0.1) raise
        # ValueError mentioning the [0, 1] range.
        region_activation_memory_budget = (
            torch.autograd.graph.region_activation_memory_budget
        )
        with self.assertRaisesRegex(TypeError, "expects a float"):
            region_activation_memory_budget(True)
        with self.assertRaisesRegex(TypeError, "expects a float"):
            region_activation_memory_budget("0.5")
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            region_activation_memory_budget(2.0)
        with self.assertRaisesRegex(ValueError, r"\[0, 1\]"):
            region_activation_memory_budget(-0.1)

    def test_region_activation_memory_budget_eager_raises(self):
        # [FUNCTION DOC] Checks that entering region_activation_memory_budget outside a
        # torch.compile region raises RuntimeError mentioning that it must be used inside
        # a torch.compile region.
        """Using the context manager outside a torch.compile region is an error."""
        with self.assertRaisesRegex(RuntimeError, "inside a torch.compile region"):
            with torch.autograd.graph.region_activation_memory_budget(0.5):
                pass

    def test_region_activation_memory_budget_conflict_raises(self):
        # [FUNCTION DOC] Checks that two different budgets (0.3 and 0.8) in one fullgraph compiled
        # graph raise RuntimeError about conflicting budgets when the backward runs, since the
        # partitioner applies a single budget per graph.
        """Two different budgets in one graph (no graph break) is an error: the
        partitioner applies a single budget per graph."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: two mm plus add and relu blocks, the first in a 0.3 budget
            # region and the second in a 0.8 budget region; returns the second result.
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                a = (torch.mm(x, y) + 1).relu()
            with torch.autograd.graph.region_activation_memory_budget(0.8):
                return (torch.mm(a, y) + 1).relu()

        cfn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "conflicting budgets"):
            cfn(x, y).sum().backward()

    def test_region_activation_memory_budget_partial_annotation_raises(self):
        # [FUNCTION DOC] Checks that annotating only part of the forward with a budget raises
        # RuntimeError saying the budget must cover the entire forward; the ops after the
        # region are unannotated.
        """Annotating only part of a graph is rejected: the budget is applied
        graph-wide, so it must cover the entire forward."""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: mm plus one then relu under a 0.3 budget region, followed by
            # an unannotated multiply by two and relu.
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                a = (torch.mm(x, y) + 1).relu()
            # This op is outside the region -> partial annotation.
            return (a * 2).relu()

        cfn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        y = torch.randn(8, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "must cover the entire forward"):
            cfn(x, y).sum().backward()

    @unittest.skipIf(not torch.cuda.is_available(), "requires CUDA")
    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_region_activation_memory_budget_covers_invoke_subgraph(self):
        # [FUNCTION DOC] Checks that a budget around a forward that includes a
        # nested_compile_region (invoke_subgraph, not inlined) also applies inside the subgraph
        # body: baseline activation memory is positive and with budget 0.0 it is exactly zero.
        # Requires CUDA; inline_single_use_invoke_subgraph is off.
        """A budget covering a forward that contains an invoke_subgraph
        (nested_compile_region) applies inside the HOP body too: the budget
        propagates into the body so the whole forward is consistently covered."""
        from torch.compiler import nested_compile_region

        N, NUM_LAYERS = 1000, 4

        def build(budget):
            # [FUNCTION DOC] HELPER: builds a function with 4 CUDA Linear(N, N) plus relu layers
            # inside a nested_compile_region, wrapped in the memory budget context when budget is
            # not None; returns that function.
            linears = torch.nn.ModuleList(
                [torch.nn.Linear(N, N) for _ in range(NUM_LAYERS)]
            ).cuda()

            @nested_compile_region
            def region(x):
                # [FUNCTION DOC] HELPER: invoke_subgraph-marked body that applies each captured
                # linear followed by relu to x.
                for lin in linears:
                    x = lin(x).relu()
                return x

            def fn(x):
                # [FUNCTION DOC] HELPER: sum of region(x), inside the memory budget context if a
                # budget was given.
                # Wrap the entire forward (HOP call + reduction); the budget must
                # consistently cover the HOP body too.
                if budget is None:
                    return region(x).sum()
                with torch.autograd.graph.region_activation_memory_budget(budget):
                    return region(x).sum()

            return fn

        def get_act_mem(f):
            # [FUNCTION DOC] HELPER: warm-up forward and backward, then measures how much CUDA
            # requested bytes grow after one forward (retained activations), runs backward and
            # returns the growth.
            out = f()
            out.backward()
            start_mem = torch.cuda.memory_stats()["requested_bytes.all.current"]
            out = f()
            act_mem = (
                torch.cuda.memory_stats()["requested_bytes.all.current"] - start_mem
            )
            out.backward()
            return act_mem

        x = torch.randn(N, N, device="cuda")

        torch._dynamo.reset()
        baseline = torch.compile(build(None), backend="aot_eager", fullgraph=True)
        self.assertGreater(get_act_mem(lambda: baseline(x)), 0)

        torch._dynamo.reset()
        recompute = torch.compile(build(0.0), backend="aot_eager", fullgraph=True)
        self.assertEqual(get_act_mem(lambda: recompute(x)), 0)

    @torch._dynamo.config.patch(inline_single_use_invoke_subgraph=False)
    def test_region_activation_memory_budget_distinct_per_invoke_subgraph_raises(self):
        # [FUNCTION DOC] Checks that two invoke_subgraph regions with different budgets (0.3 and
        # 0.8) in one fullgraph graph raise RuntimeError about conflicting budgets, because the
        # agreement check recurses into nested subgraphs. inline_single_use_invoke_subgraph is off.
        """Different budgets in two invoke_subgraph regions of a single graph are
        rejected: the agreement check recurses into nested subgraphs, so the
        outermost graph sees both budgets. Use a graph break for different
        budgets."""
        from torch.compiler import nested_compile_region

        wa = torch.randn(8, 8, requires_grad=True)
        wb = torch.randn(8, 8, requires_grad=True)

        @nested_compile_region
        def region_a(x):
            # [FUNCTION DOC] HELPER: nested_compile_region body computing relu of x matmul wa
            # inside a 0.3 budget region.
            with torch.autograd.graph.region_activation_memory_budget(0.3):
                return (x @ wa).relu()

        @nested_compile_region
        def region_b(x):
            # [FUNCTION DOC] HELPER: nested_compile_region body computing relu of x matmul wb
            # inside a 0.8 budget region.
            with torch.autograd.graph.region_activation_memory_budget(0.8):
                return (x @ wb).relu()

        def fn(x):
            # [FUNCTION DOC] HELPER: sum of region_b applied to region_a of x.
            return region_b(region_a(x)).sum()

        cfn = torch.compile(fn, backend="aot_eager", fullgraph=True)
        x = torch.randn(8, 8, requires_grad=True)
        with self.assertRaisesRegex(RuntimeError, "conflicting budgets"):
            cfn(x).backward()


class RematerializeACNodesPassTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for the remat-using-tags pass on a single forward-loss-backward graph (the
    # backward is traced inside the compiled function via autograd.grad). Uses
    # _compile_and_capture to get the forward graph with and without
    # remat_using_tags_for_fwd_loss_bwd_graph and checks recompute counts, RNG error handling,
    # SDPA dropout RNG classification and selective checkpoint policies.
    """Tests for AC reordering optimization in full graph (forward+backward in one graph)."""

    def count_op(self, gm, target):
        # [FUNCTION DOC] HELPER: returns the number of nodes in gm.graph whose
        # target equals target.
        return sum(1 for n in gm.graph.nodes if n.target == target)

    def _compile_and_capture(self, fn, remat_using_tags_for_fwd_loss_bwd_graph, inputs):
        # [FUNCTION DOC] HELPER: compiles fn fullgraph with aot_autograd (a capturing forward
        # compiler, no backward compiler or partitioner) while patching
        # remat_using_tags_for_fwd_loss_bwd_graph to the given flag and enabling
        # trace_autograd_ops. Calls the compiled function on inputs and returns the result and
        # the captured GraphModule.
        captured_gm = None

        def compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: forward compiler that stores the graph module in the enclosing
            # captured_gm and returns its forward.
            nonlocal captured_gm
            captured_gm = gm
            return gm.forward

        backend = aot_autograd(
            fw_compiler=compiler,
            bw_compiler=None,
            partition_fn=None,
        )

        with (
            torch._functorch.config.patch(
                remat_using_tags_for_fwd_loss_bwd_graph=remat_using_tags_for_fwd_loss_bwd_graph
            ),
            torch._dynamo.config.patch(trace_autograd_ops=True),
        ):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
            result = compiled_fn(*inputs)

        return result, captured_gm

    @unittest.skipIf(not HAS_GPU_AND_TRITON, "GPU not available")
    def test_ac_rematerialize_simple_forward_backward(self):
        # [FUNCTION DOC] Compiles a fwd-loss-bwd function (checkpointed sigmoid of matmul,
        # gradients via autograd.grad) with the remat flag off and on. Gradients must match; with
        # remat the graph has 4 mm and 2 sigmoid instead of 3 and 1 (recompute in the backward),
        # and the graph code matches the expected text. Skipped without GPU and triton.
        x = torch.randn(4, 4, requires_grad=True)
        y = torch.randn(4, 4, requires_grad=True)

        def simple_fwd_bwd(x, y):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of sigmoid of matmul on x and y,
            # sums, and returns detached gradients with respect to x and y.
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sigmoid(torch.matmul(a, b)),
                x,
                y,
                use_reentrant=False,
            )
            loss = z.sum()

            dx, dy = _grad(loss, (x, y))

            return dx.detach(), dy.detach()

        (dx1, dy1), gm_without = self._compile_and_capture(
            simple_fwd_bwd, False, (x, y)
        )
        (dx2, dy2), gm_with = self._compile_and_capture(simple_fwd_bwd, True, (x, y))

        self.assertTrue(torch.allclose(dx1, dx2))
        self.assertTrue(torch.allclose(dy1, dy2))

        mm_with = self.count_op(gm_with, torch.ops.aten.mm.default)
        mm_without = self.count_op(gm_without, torch.ops.aten.mm.default)
        sigmoid_with = self.count_op(gm_with, torch.ops.aten.sigmoid.default)
        sigmoid_without = self.count_op(gm_without, torch.ops.aten.sigmoid.default)
        self.assertEqual(mm_with, 4, "mm should be recomputed in backward")
        self.assertEqual(mm_without, 3)
        self.assertEqual(sigmoid_with, 2, "sigmoid should be recomputed in backward")
        self.assertEqual(sigmoid_without, 1)

        self.assertExpectedInline(
            gm_with.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg0_1, arg1_1)
    sigmoid = torch.ops.aten.sigmoid.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    mm_recomputed = torch.ops.aten.mm.default(arg0_1, arg1_1)
    sigmoid_recomputed = torch.ops.aten.sigmoid.default(mm_recomputed);  mm_recomputed = None
    detach_2_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_4 = torch.ops.aten.detach.default(detach_2_recomputed);  detach_2_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_4);  expand = detach_4 = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_2 = torch.ops.aten.mm.default(t, sigmoid_backward);  t = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_3 = torch.ops.aten.mm.default(sigmoid_backward, t_1);  sigmoid_backward = t_1 = None
    detach_5 = torch.ops.aten.detach.default(mm_3);  mm_3 = None
    detach_6 = torch.ops.aten.detach.default(mm_2);  mm_2 = None
    return (detach_5, detach_6)""",
        )

    def test_ac_rematerialize_with_rng_ops_raises_error(self):
        # [FUNCTION DOC] Checks that compiling with remat enabled a checkpointed region containing
        # an RNG op (rand_like) in a fwd-loss-bwd graph raises BackendCompilerFailed saying
        # rematerialization does not support RNG ops in recompute regions.
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd_with_rng(x):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of sigmoid of x plus random values,
            # sums, and returns the gradient with respect to x.
            z = torch.utils.checkpoint.checkpoint(
                lambda a: torch.sigmoid(a + torch.rand_like(a)), x, use_reentrant=False
            )
            loss = z.sum()

            dx = _grad(loss, x)[0]

            return dx

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "Activation checkpoint rematerialization in `forward-loss-backward` graph does not support RNG ops in recompute regions.",
        ):
            self._compile_and_capture(fwd_bwd_with_rng, True, (x,))

    def test_ac_rematerialize_sdpa_rng_classification(self):
        # [FUNCTION DOC] Checks has_recomputable_rng_ops and is_rng_op for scaled-dot-product-
        # attention ops: dropout_p of zero (int, float, negative zero) is not RNG, while 1e-12,
        # 0.1, nan, a placeholder (dynamic) value and a nonzero SymFloat are. Builds small FX
        # graphs with a PREFER_RECOMPUTE node per op, passing dropout positionally and as a
        # keyword, and a SimpleNamespace node for the SymFloat checks.
        sdpa_ops = [
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_attention_math_for_mps.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention_backward.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.quantized,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention_backward.default,
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
            torch.ops.aten._scaled_dot_product_flash_attention_for_cpu.default,
            torch.ops.aten._flash_attention_forward.default,
            torch.ops.aten._flash_attention_forward.quantized,
            torch.ops.aten._flash_attention_forward_no_dropout_inplace.default,
            torch.ops.aten._efficient_attention_forward.default,
            torch.ops.aten._cudnn_attention_forward.default,
        ]
        shape_env = ShapeEnv()
        zero_symbol = shape_env.create_symbol(
            0.0,
            source=ConstantSource("dropout_zero"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )
        nonzero_symbol = shape_env.create_symbol(
            0.1,
            source=ConstantSource("dropout_nonzero"),
            dynamic_dim=DimDynamic.DUCK,
            constraint_dim=None,
        )
        symfloat_zero = torch.SymFloat(SymNode(zero_symbol, shape_env, float, hint=0.0))
        symfloat_nonzero = torch.SymFloat(
            SymNode(nonzero_symbol, shape_env, float, hint=0.1)
        )
        dropout_cases = [
            (0, False),
            (0.0, False),
            (-0.0, False),
            (1e-12, True),
            (0.1, True),
            (float("nan"), True),
        ]

        for op in sdpa_ops:
            dropout_arg_idx = next(
                idx
                for idx, arg in enumerate(op._schema.arguments)
                if arg.name == "dropout_p"
            )
            for dropout_p, expected in dropout_cases:
                with self.subTest(op=op, dropout_p=dropout_p):
                    graph = torch.fx.Graph()
                    q = graph.placeholder("q")
                    args = [q] * (dropout_arg_idx + 1)
                    args[dropout_arg_idx] = dropout_p
                    node = graph.call_function(op, tuple(args), {})
                    node.meta["recompute"] = (
                        torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                    )
                    graph.output(node)
                    gm = torch.fx.GraphModule({}, graph)

                    self.assertEqual(has_recomputable_rng_ops(gm), expected)

            with self.subTest(op=op, dropout_p="dynamic"):
                graph = torch.fx.Graph()
                q = graph.placeholder("q")
                dropout_p = graph.placeholder("dropout_p")
                args = [q] * (dropout_arg_idx + 1)
                args[dropout_arg_idx] = dropout_p
                node = graph.call_function(op, tuple(args), {})
                node.meta["recompute"] = (
                    torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                )
                graph.output(node)
                gm = torch.fx.GraphModule({}, graph)

                self.assertTrue(has_recomputable_rng_ops(gm))

            for dropout_p, expected in dropout_cases:
                with self.subTest(op=op, kwarg_dropout_p=dropout_p):
                    graph = torch.fx.Graph()
                    q = graph.placeholder("q")
                    node = graph.call_function(
                        op, tuple([q] * dropout_arg_idx), {"dropout_p": dropout_p}
                    )
                    node.meta["recompute"] = (
                        torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
                    )
                    graph.output(node)
                    gm = torch.fx.GraphModule({}, graph)

                    self.assertEqual(has_recomputable_rng_ops(gm), expected)

            for dropout_p, expected in [
                (symfloat_zero, False),
                (symfloat_nonzero, True),
            ]:
                with self.subTest(op=op, symfloat_dropout_p=dropout_p):
                    args = [None] * (dropout_arg_idx + 1)
                    args[dropout_arg_idx] = dropout_p
                    node = SimpleNamespace(target=op, args=tuple(args), kwargs={})

                    self.assertEqual(is_rng_op(node), expected)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_ac_rematerialize_with_sdpa_dropout_zero(self):
        # [FUNCTION DOC] On CUDA, runs a checkpointed F.scaled_dot_product_attention with
        # dropout_p 0.0 and a PREFER_RECOMPUTE policy for sdpa ops (everything else PREFER_SAVE)
        # in a fwd-loss-bwd graph, for each available fused backend (efficient fp32, flash fp16,
        # cudnn fp16). Compares remat on, remat off and eager outputs and gradients, and asserts
        # the remat graph contains exactly 2 sdpa ops (forward plus recompute). Skips if no
        # fused backend exists.
        from torch.nn.attention import sdpa_kernel, SDPBackend

        cases = []
        if PLATFORM_SUPPORTS_MEM_EFF_ATTENTION:
            cases.append((SDPBackend.EFFICIENT_ATTENTION, torch.float32))
        if PLATFORM_SUPPORTS_FLASH_ATTENTION:
            cases.append((SDPBackend.FLASH_ATTENTION, torch.float16))
        if PLATFORM_SUPPORTS_CUDNN_ATTENTION:
            cases.append((SDPBackend.CUDNN_ATTENTION, torch.float16))
        if not cases:
            self.skipTest("No fused SDPA backends available")
        sdpa_ops = {
            torch.ops.aten.scaled_dot_product_attention.default,
            torch.ops.aten._scaled_dot_product_cudnn_attention.default,
            torch.ops.aten._scaled_dot_product_flash_attention.default,
            torch.ops.aten._scaled_dot_product_efficient_attention.default,
            torch.ops.aten._scaled_dot_product_fused_attention_overrideable.default,
        }

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: policy returning PREFER_RECOMPUTE for attention ops in
            # sdpa_ops and PREFER_SAVE for others.
            if op in sdpa_ops:
                return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_SAVE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        for backend, dtype in cases:
            with self.subTest(backend=backend, dtype=dtype):
                torch._dynamo.reset()
                q = torch.randn(
                    2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True
                )
                k = torch.randn(
                    2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True
                )
                v = torch.randn(
                    2, 4, 128, 64, device="cuda", dtype=dtype, requires_grad=True
                )

                def fwd_bwd_with_sdpa(q, k, v):
                    # [FUNCTION DOC] HELPER: under the chosen sdpa_kernel backend, checkpoints
                    # scaled-dot-product attention without dropout (non-reentrant, with
                    # context_fn), sums it, takes gradients with respect to q, k, v and returns the
                    # detached output and the three gradients.
                    with sdpa_kernel(backend):
                        z = torch.utils.checkpoint.checkpoint(
                            lambda q, k, v: F.scaled_dot_product_attention(
                                q, k, v, dropout_p=0.0
                            ),
                            q,
                            k,
                            v,
                            use_reentrant=False,
                            context_fn=context_fn,
                        )
                        loss = z.sum()
                        dq, dk, dv = _grad(loss, (q, k, v))

                    return z.detach(), dq, dk, dv

                result_with, gm_with = self._compile_and_capture(
                    fwd_bwd_with_sdpa, True, (q, k, v)
                )
                torch._dynamo.reset()
                result_without, _ = self._compile_and_capture(
                    fwd_bwd_with_sdpa, False, (q, k, v)
                )
                eager_inputs = tuple(
                    t.detach().clone().requires_grad_(True) for t in (q, k, v)
                )
                result_eager = fwd_bwd_with_sdpa(*eager_inputs)

                for actual, expected in zip(result_with, result_without):
                    self.assertEqual(actual, expected)
                for actual, expected in zip(result_with, result_eager):
                    self.assertEqual(actual, expected)
                self.assertEqual(sum(self.count_op(gm_with, op) for op in sdpa_ops), 2)

    def test_ac_rematerialize_with_no_annotations(self):
        # [FUNCTION DOC] Checks remat works without explicit annotations: compiles a checkpointed
        # sigmoid of x plus 4 with remat on and off, asserts equal gradients and that the graph has
        # 2 sigmoid ops (recomputed) with remat and 1 without.
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd(x):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of sigmoid of x plus 4, sum, and
            # returns the gradient with respect to x.
            z = torch.utils.checkpoint.checkpoint(
                lambda a: torch.sigmoid(a + 4), x, use_reentrant=False
            )
            loss = z.sum()
            return _grad(loss, x)[0]

        result_with, gm_with = self._compile_and_capture(fwd_bwd, True, (x,))
        result_without, gm_without = self._compile_and_capture(fwd_bwd, False, (x,))

        self.assertTrue(torch.allclose(result_with, result_without))

        # autograd_backward tagging is automatic now, so remat should still work
        sigmoid_with = self.count_op(gm_with, torch.ops.aten.sigmoid.default)
        sigmoid_without = self.count_op(gm_without, torch.ops.aten.sigmoid.default)
        self.assertEqual(sigmoid_with, 2, "sigmoid should be recomputed in backward")
        self.assertEqual(sigmoid_without, 1)

    def test_ac_rematerialize_with_selective_checkpoint_policy(self):
        # [FUNCTION DOC] Checks remat in a fwd-loss-bwd graph with a selective checkpoint policy
        # saving addmm and recomputing everything else, on a linear plus relu region (x, weight,
        # bias inputs). Asserts gradients match remat-off, that addmm count is unchanged (saved,
        # not recomputed) and relu appears once more with remat, and that a recomputed relu node
        # exists but no recomputed addmm.
        x = torch.randn(4, 128, requires_grad=True)
        w1 = torch.randn(128, 128, requires_grad=True)
        b1 = torch.randn(128, requires_grad=True)

        def policy_fn(ctx, op, *args, **kwargs):
            # [FUNCTION DOC] HELPER: policy returning MUST_SAVE for aten.addmm and PREFER_RECOMPUTE
            # for all other ops.
            if op == torch.ops.aten.addmm.default:
                return torch.utils.checkpoint.CheckpointPolicy.MUST_SAVE
            return torch.utils.checkpoint.CheckpointPolicy.PREFER_RECOMPUTE

        context_fn = functools.partial(
            torch.utils.checkpoint.create_selective_checkpoint_contexts, policy_fn
        )

        def fwd_bwd_with_policy(x, w1, b1):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of linear plus relu with the SAC
            # context_fn, sums the result and returns gradients with respect to x, w1 and b1.
            def checkpoint_fn(inp, w, b):
                # [FUNCTION DOC] HELPER: applies a linear layer (weight w, bias b) to inp
                # and then relu.
                linear = torch.nn.functional.linear(inp, w, b)
                return torch.relu(linear)

            result = torch.utils.checkpoint.checkpoint(
                checkpoint_fn, x, w1, b1, use_reentrant=False, context_fn=context_fn
            )
            loss = result.sum()

            dx, dw, db = _grad(loss, (x, w1, b1))
            return dx, dw, db

        result_with, gm_with = self._compile_and_capture(
            fwd_bwd_with_policy, True, (x, w1, b1)
        )
        result_without, gm_without = self._compile_and_capture(
            fwd_bwd_with_policy, False, (x, w1, b1)
        )

        torch.testing.assert_close(result_with[0], result_without[0])
        torch.testing.assert_close(result_with[1], result_without[1])
        torch.testing.assert_close(result_with[2], result_without[2])

        addmm_without = self.count_op(gm_without, torch.ops.aten.addmm.default)
        relu_without = self.count_op(gm_without, torch.ops.aten.relu.default)

        addmm_with = self.count_op(gm_with, torch.ops.aten.addmm.default)
        relu_with = self.count_op(gm_with, torch.ops.aten.relu.default)

        self.assertEqual(addmm_without, addmm_with)
        self.assertEqual(relu_with, relu_without + 1)

        recomputed_nodes = [
            n.name for n in gm_with.graph.nodes if "_recomputed" in n.name
        ]
        self.assertNotIn("addmm_recomputed", recomputed_nodes)

        self.assertTrue(
            any("relu" in name for name in recomputed_nodes),
            lambda msg: f"{msg}\nExpected relu_recomputed but got: {recomputed_nodes}",
        )

    def _compile_with_joint_graph_pass_and_capture(self, fn, inputs):
        # [FUNCTION DOC] HELPER: compiles fn fullgraph with aot_autograd whose forward compiler
        # deep-copies the graph, runs inductor joint_graph_passes on it in place and returns its
        # forward; trace_autograd_ops is enabled. Returns the result, the copy before the passes
        # and the graph after the passes.
        from torch._inductor.fx_passes.joint_graph import joint_graph_passes

        captured_gm_before = None
        captured_gm_after = None

        def custom_compiler(gm, example_inputs):
            # [FUNCTION DOC] HELPER: forward compiler that stores a deep copy of gm, applies
            # joint_graph_passes to gm in place, stores gm and returns gm.forward.
            nonlocal captured_gm_before, captured_gm_after
            import copy

            captured_gm_before = copy.deepcopy(gm)
            joint_graph_passes(gm)
            captured_gm_after = gm
            return gm.forward

        backend = aot_autograd(
            fw_compiler=custom_compiler,
            bw_compiler=None,
            partition_fn=None,
        )

        with torch._dynamo.config.patch(trace_autograd_ops=True):
            compiled_fn = torch.compile(fn, backend=backend, fullgraph=True)
            result = compiled_fn(*inputs)

        return result, captured_gm_before, captured_gm_after

    def test_joint_graph_passes_view_optimization(self):
        # [FUNCTION DOC] Checks that joint graph passes fold chained views: a checkpointed function
        # with two view calls in a fwd-loss-bwd graph has 6 view ops before the passes and 0 after;
        # the gradient matches eager sigmoid gradient and the post-pass code matches expected text.
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd_with_views(x):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of a view-to-16, view-to-4x4, sigmoid
            # region; sums and returns the detached gradient with respect to x.
            def checkpoint_fn(a):
                # [FUNCTION DOC] HELPER: two successive views of a (flatten to 16 then back to 4x4)
                # followed by sigmoid.
                b = a.view(16)
                c = b.view(4, 4)
                return torch.sigmoid(c)

            z = torch.utils.checkpoint.checkpoint(
                checkpoint_fn,
                x,
                use_reentrant=False,
            )
            loss = z.sum()

            dx = _grad(loss, x)[0]

            return dx.detach()

        result, gm_before, gm_after = self._compile_with_joint_graph_pass_and_capture(
            fwd_bwd_with_views, (x,)
        )

        result_eager = torch.autograd.grad(torch.sigmoid(x).sum(), x)[0]
        self.assertTrue(torch.allclose(result, result_eager, atol=1e-5))

        view_count_before = self.count_op(gm_before, torch.ops.aten.view.default)
        view_count_after = self.count_op(gm_after, torch.ops.aten.view.default)
        self.assertTrue(view_count_after == 0)
        self.assertTrue(view_count_before == 6)

        self.assertExpectedInline(
            gm_after.code.strip(),
            """\
def forward(self, arg0_1):
    sigmoid = torch.ops.aten.sigmoid.default(arg0_1)
    sum_1 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    sigmoid_recomputed = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
    detach_3_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_5 = torch.ops.aten.detach.default(detach_3_recomputed);  detach_3_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_5);  expand = detach_5 = None
    detach_6 = torch.ops.aten.detach.default(sigmoid_backward);  sigmoid_backward = None
    return (detach_6,)""",
        )

    def test_joint_graph_passes_permute_optimization(self):
        # [FUNCTION DOC] Checks that joint graph passes eliminate double permutes: a checkpointed
        # function with two permutes has 6 permute ops before the passes and 0 after; gradient
        # matches eager and the post-pass code matches expected text.
        x = torch.randn(4, 4, requires_grad=True)

        def fwd_bwd_with_permute(x):
            # [FUNCTION DOC] HELPER: non-reentrant checkpoint of a double-permute plus sigmoid
            # region; sums and returns the detached gradient with respect to x.
            def checkpoint_fn(a):
                # [FUNCTION DOC] HELPER: two successive transposing permutes of a
                # followed by sigmoid.
                b = a.permute(1, 0)
                c = b.permute(1, 0)
                return torch.sigmoid(c)

            z = torch.utils.checkpoint.checkpoint(
                checkpoint_fn,
                x,
                use_reentrant=False,
            )
            loss = z.sum()

            dx = _grad(loss, x)[0]

            return dx.detach()

        result, gm_before, gm_after = self._compile_with_joint_graph_pass_and_capture(
            fwd_bwd_with_permute, (x,)
        )

        result_eager = torch.autograd.grad(torch.sigmoid(x).sum(), x)[0]
        self.assertTrue(torch.allclose(result, result_eager, atol=1e-5))

        permute_count_before = self.count_op(gm_before, torch.ops.aten.permute.default)
        permute_count_after = self.count_op(gm_after, torch.ops.aten.permute.default)
        self.assertTrue(permute_count_after == 0)
        self.assertTrue(permute_count_before == 6)

        self.assertExpectedInline(
            gm_after.code.strip(),
            """\
def forward(self, arg0_1):
    sigmoid = torch.ops.aten.sigmoid.default(arg0_1)
    sum_1 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    sigmoid_recomputed = torch.ops.aten.sigmoid.default(arg0_1);  arg0_1 = None
    detach_3_recomputed = torch.ops.aten.detach.default(sigmoid_recomputed);  sigmoid_recomputed = None
    detach_5 = torch.ops.aten.detach.default(detach_3_recomputed);  detach_3_recomputed = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand, detach_5);  expand = detach_5 = None
    detach_6 = torch.ops.aten.detach.default(sigmoid_backward);  sigmoid_backward = None
    return (detach_6,)""",
        )

    @torch._dynamo.config.patch(skip_fwd_side_effects_in_bwd_under_checkpoint=True)
    def test_attr_compile_submodules_in_checkpoint_wrapper(self):
        # [FUNCTION DOC] Checks that compiling each child module of a non-reentrant checkpointed
        # block with aot_eager does not exceed a recompile limit of 2 (pack hook guards must not
        # cause recompiles). Compares output and input gradient with an eager block with the same
        # weights. skip_fwd_side_effects_in_bwd_under_checkpoint is enabled.
        """Compiling submodules inside a checkpointed block should not hit the
        recompile limit due to WeakKeyDictionary guards in the pack_hook."""
        from torch.utils.checkpoint import checkpoint

        class Block(nn.Module):
            # [CLASS DOC] HELPER: residual block of three RMSNorm plus bias-free Linear pairs, each
            # added to the running activation.
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: creates three RMSNorm(dim) and three bias-free
                # Linear(dim, dim) layers.
                super().__init__()
                self.norm1 = nn.RMSNorm(dim)
                self.linear1 = nn.Linear(dim, dim, bias=False)
                self.norm2 = nn.RMSNorm(dim)
                self.linear2 = nn.Linear(dim, dim, bias=False)
                self.norm3 = nn.RMSNorm(dim)
                self.linear3 = nn.Linear(dim, dim, bias=False)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies the three norm-linear residual updates in order
                # and returns the result.
                x = x + self.linear1(self.norm1(x))
                x = x + self.linear2(self.norm2(x))
                x = x + self.linear3(self.norm3(x))
                return x

        class CheckpointedBlock(nn.Module):
            # [CLASS DOC] HELPER: wrapper holding a block whose forward runs it
            # under non-reentrant checkpoint.
            def __init__(self, block):
                # [FUNCTION DOC] HELPER: stores the wrapped block as a submodule.
                super().__init__()
                self.block = block

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of the block on x.
                return checkpoint(self.block, x, use_reentrant=False)

        dim = 32
        block = Block(dim)

        x_ref = torch.randn(4, dim, requires_grad=True)
        ref = block(x_ref)
        ref.sum().backward()

        block_cp = Block(dim)
        block_cp.load_state_dict(block.state_dict())
        wrapped = CheckpointedBlock(block_cp)

        for _, submod in wrapped.block.named_children():
            submod.compile(backend="aot_eager")

        with torch._dynamo.config.patch(recompile_limit=2):
            x_test = x_ref.detach().clone().requires_grad_(True)
            result = wrapped(x_test)
            result.sum().backward()

        self.assertEqual(ref, result)
        self.assertEqual(x_ref.grad, x_test.grad)

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_multiple_user_phase_annotations_errors(self):
        # [FUNCTION DOC] Checks that two separate backward-annotated regions (annotate phase
        # backward) in one fwd-loss-bwd graph, with unannotated work between them, raise
        # RuntimeError about annotated backward regions when compiled with remat. Requires CUDA.
        x = torch.randn(4, 4, requires_grad=True)
        w = torch.randn(4, 4, requires_grad=True)

        def fn(x, w):
            # [FUNCTION DOC] HELPER: checkpointed sin of a matmul, a backward-annotated gradient,
            # an add of the gradients, then a second backward-annotated multiply by two; returns
            # the detached result.
            z = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sin(a @ b), x, w, use_reentrant=False
            )
            loss = z.sum()
            with torch.fx.traceback.annotate({"phase": "backward"}):
                dx, dw = _grad(loss, (x, w))
            # Non-backward computation between two backward annotations
            out = dx + dw
            with torch.fx.traceback.annotate({"phase": "backward"}):
                out = out * 2
            return out.detach()

        with self.assertRaisesRegex(RuntimeError, "backward regions annotated"):
            self._compile_and_capture(fn, True, (x, w))

    @unittest.skipIf(not torch.cuda.is_available(), "CUDA not available")
    def test_user_phase_annotation_with_extra_autograd_grad(self):
        # [FUNCTION DOC] Checks that only the user-annotated backward region gets
        # rematerialization: the first gradient is annotated as phase backward and the second
        # autograd.grad is not; with remat the captured code matches expected text. Requires CUDA.
        """Only the user-annotated backward region gets rematerialization."""
        x = torch.randn(4, 4, requires_grad=True)
        w1 = torch.randn(4, 4, requires_grad=True)
        w2 = torch.randn(4, 4, requires_grad=True)

        def fn(x, w1, w2):
            # [FUNCTION DOC] HELPER: two checkpointed regions (sin of matmul with w1, sigmoid of
            # matmul with w2); computes the gradient of the first loss inside the backward
            # annotation and of the second outside it; returns the detached sum of both gradients.
            z1 = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sin(a @ b), x, w1, use_reentrant=False
            )
            z2 = torch.utils.checkpoint.checkpoint(
                lambda a, b: torch.sigmoid(a @ b), x, w2, use_reentrant=False
            )
            loss1 = z1.sum()
            loss2 = z2.sum()
            # Only the first backward is annotated
            with torch.fx.traceback.annotate({"phase": "backward"}):
                dx1 = _grad(loss1, (x,))
            # Second backward NOT annotated — should not get remat
            dx2 = _grad(loss2, (x,))
            return (dx1[0] + dx2[0]).detach()

        _, gm_with = self._compile_and_capture(fn, True, (x, w1, w2))

        self.assertExpectedInline(
            gm_with.code.strip(),
            """\
def forward(self, arg0_1, arg1_1, arg2_1):
    mm = torch.ops.aten.mm.default(arg2_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [4, 4]);  ones_like = None
    mm_recomputed = torch.ops.aten.mm.default(arg2_1, arg0_1)
    cos = torch.ops.aten.cos.default(mm_recomputed);  mm_recomputed = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    t = torch.ops.aten.t.default(arg0_1);  arg0_1 = None
    mm_2 = torch.ops.aten.mm.default(mul, t);  mul = t = None
    mm_3 = torch.ops.aten.mm.default(arg2_1, arg1_1);  arg2_1 = None
    sigmoid = torch.ops.aten.sigmoid.default(mm_3);  mm_3 = None
    detach_4 = torch.ops.aten.detach.default(sigmoid)
    sum_2 = torch.ops.aten.sum.default(sigmoid);  sigmoid = None
    ones_like_1 = torch.ops.aten.ones_like.default(sum_2, pin_memory = False, memory_format = torch.preserve_format);  sum_2 = None
    expand_1 = torch.ops.aten.expand.default(ones_like_1, [4, 4]);  ones_like_1 = None
    detach_5 = torch.ops.aten.detach.default(detach_4);  detach_4 = None
    sigmoid_backward = torch.ops.aten.sigmoid_backward.default(expand_1, detach_5);  expand_1 = detach_5 = None
    t_1 = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_4 = torch.ops.aten.mm.default(sigmoid_backward, t_1);  sigmoid_backward = t_1 = None
    add = torch.ops.aten.add.Tensor(mm_2, mm_4);  mm_2 = mm_4 = None
    detach_6 = torch.ops.aten.detach.default(add);  add = None
    return (detach_6,)""",
        )

    def test_chunked_loss_remat(self):
        # [FUNCTION DOC] Checks remat on a chunked loss pattern: per-chunk backward calls on a
        # detached input are skipped and only the final backward through the checkpointed block is
        # rematerialized. Results with remat on and off match and gelu appears once without remat
        # and twice with it.
        """Chunked loss pattern: multiple backward regions from chunk_loss.backward()
        calls, but only the final x.backward() region needs remat. The pass should
        skip the chunk backwards and only rematerialize for the final one."""
        dim = 32
        chunksz = 4

        class Block(nn.Module):
            # [CLASS DOC] HELPER: block of two bias-free Linear layers with a tanh-approximated
            # gelu between them, run under non-reentrant checkpoint.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates two bias-free linear layers.
                super().__init__()
                self.l1 = nn.Linear(dim, dim, bias=False)
                self.l2 = nn.Linear(dim, dim, bias=False)

            def _fn(self, x):
                # [FUNCTION DOC] HELPER: first linear, tanh-gelu, second linear.
                return self.l2(F.gelu(self.l1(x), approximate="tanh"))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of _fn on x.
                return checkpoint(self._fn, x, use_reentrant=False)

        class ChunkedLoss(nn.Module):
            # [CLASS DOC] HELPER: module that runs the checkpointed block, detaches its
            # output, computes a summed mse loss per chunk of 4 rows through a head layer
            # with backward called on each chunk, then calls backward on the block output
            # with the accumulated gradient.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the block and a bias-free head linear.
                super().__init__()
                self.block = Block()
                self.head = nn.Linear(dim, dim, bias=False)

            def forward(self, x, y):
                # [FUNCTION DOC] HELPER: runs the block, performs chunked loss backward passes on a
                # detached copy and a final backward through the block with the accumulated
                # gradient; returns the total chunk loss as a detached sum.
                x = self.block(x)
                x_detached = x.detach().requires_grad_()
                total = 0
                for start in range(0, x_detached.shape[0], chunksz):
                    end = start + chunksz
                    chunk_loss = (
                        F.mse_loss(
                            self.head(x_detached[start:end]),
                            y[start:end],
                            reduction="sum",
                        )
                        / x_detached.shape[0]
                    )
                    chunk_loss.backward()
                    total = total + chunk_loss.detach()
                x.backward(x_detached.grad)
                return total

        model = ChunkedLoss()
        x = torch.randn(12, dim)
        y = torch.randn(12, dim)

        result_with, gm_with = self._compile_and_capture(model, True, (x, y))
        result_without, gm_without = self._compile_and_capture(model, False, (x, y))

        torch.testing.assert_close(result_with, result_without)

        # Without remat, gelu appears once (forward only).
        # With remat, gelu is duplicated into the backward region.
        gelu_without = self.count_op(gm_without, torch.ops.aten.gelu.default)
        gelu_with = self.count_op(gm_with, torch.ops.aten.gelu.default)
        self.assertEqual(gelu_without, 1)
        self.assertEqual(gelu_with, 2, "gelu should be recomputed in backward")

    def test_two_backward_regions_needing_remat_errors(self):
        # [FUNCTION DOC] Checks that two independent backward calls that both need recomputation in
        # one graph raise BackendCompilerFailed with a message about requiring recomputation.
        """Two independent backward calls that both need recompute should error."""
        dim = 32

        class Block(nn.Module):
            # [CLASS DOC] HELPER: block of two bias-free Linear layers with a tanh-approximated
            # gelu between them, run under non-reentrant checkpoint.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates two bias-free linear layers.
                super().__init__()
                self.l1 = nn.Linear(dim, dim, bias=False)
                self.l2 = nn.Linear(dim, dim, bias=False)

            def _fn(self, x):
                # [FUNCTION DOC] HELPER: first linear, tanh-gelu, second linear.
                return self.l2(F.gelu(self.l1(x), approximate="tanh"))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of _fn on x.
                return checkpoint(self._fn, x, use_reentrant=False)

        class TwoBackwards(nn.Module):
            # [CLASS DOC] HELPER: two checkpointed blocks in sequence whose forward runs two
            # backward passes (one from z and one from y) and returns the detached z.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates two blocks.
                super().__init__()
                self.block1 = Block()
                self.block2 = Block()

            def forward(self, x):
                # [FUNCTION DOC] HELPER: runs both blocks, calls backward on the sum of z with
                # retain_graph and on the sum of y, and returns detached z.
                y = self.block1(x)
                z = self.block2(y)
                z.sum().backward(retain_graph=True)
                y.sum().backward()
                return z.detach()

        x = torch.randn(8, dim, requires_grad=True)

        with self.assertRaisesRegex(
            torch._dynamo.exc.BackendCompilerFailed,
            "require recomputation",
        ):
            self._compile_and_capture(TwoBackwards(), True, (x,))


instantiate_device_type_tests(
    ActivationCheckpointingViaTagsTests, globals(), except_for="cpu"
)


class ActivationCheckpointingNonStrictTracerTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests for the non-strict tracing flow (make_fx under torch.compiler non-strict
    # tracing context with autograd.grad/backward patches) interacting with eager
    # torch.utils.checkpoint: backward node tagging and sequence numbers, patch preconditions, tag
    # leakage, backward region counting and golden generated code for checkpoint, SAC and RNG
    # cases. Uses _trace_train_step and _count_backward_regions as shared helpers.
    """Tests for non-strict tracing flag interaction with checkpoint."""

    @staticmethod
    def _count_backward_regions(gm):
        # [FUNCTION DOC] HELPER: counts contiguous runs of nodes in gm.graph whose
        # autograd_backward metadata is true (a new region starts when a backward node follows a
        # non-backward node) and returns the count.
        regions = 0
        in_backward = False
        for node in gm.graph.nodes:
            is_backward = bool(node.meta.get("autograd_backward", False))
            if is_backward and not in_backward:
                regions += 1
            in_backward = is_backward
        return regions

    def test_backward_nodes_have_seq_nr_under_non_strict(self):
        # [FUNCTION DOC] Traces a train step (make_fx) of a model with a non-reentrant checkpointed
        # sin of a matmul and checks that the sets of seq_nr values on forward call_function nodes
        # (not autograd_backward) and on backward nodes are both non-empty and equal.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: module with a 4x4 parameter w whose forward checkpoints sin of
            # x matmul w.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the parameter w as a random 4x4 tensor.
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of a lambda computing sin(x @ w).
                return checkpoint(
                    lambda x: torch.sin(x @ self.w), x, use_reentrant=False
                )

        gm = self._trace_train_step(Model(), torch.randn(2, 4))
        forward_seq_nrs = {
            node.meta["seq_nr"]
            for node in gm.graph.nodes
            if node.op == "call_function"
            and not node.meta.get("autograd_backward", False)
            and "seq_nr" in node.meta
        }
        backward_seq_nrs = {
            node.meta["seq_nr"]
            for node in gm.graph.nodes
            if node.op == "call_function"
            and node.meta.get("autograd_backward", False)
            and "seq_nr" in node.meta
        }

        self.assertTrue(forward_seq_nrs)
        self.assertTrue(backward_seq_nrs)
        self.assertSetEqual(backward_seq_nrs, forward_seq_nrs)

    def test_patch_autograd_grad_requires_non_strict_tracing(self):
        # [FUNCTION DOC] Checks that calling torch.autograd.grad inside the patch_autograd_grad
        # context without the non-strict tracing context raises AssertionError saying the patch
        # must be used under the non-strict tracing context.
        x = torch.randn(2, 4, requires_grad=True)
        loss = torch.sin(x).sum()

        with torch.compiler._patch_autograd_grad():
            with self.assertRaisesRegex(
                AssertionError,
                "_patch_autograd_grad\\(\\) must be used under "
                "_non_strict_tracing_context\\(\\)",
            ):
                torch.autograd.grad(loss, (x,))

    def test_patch_engine_backward_requires_non_strict_tracing(self):
        # [FUNCTION DOC] Checks that calling backward inside the patch_engine_backward context
        # without the non-strict tracing context raises AssertionError saying the patch must be
        # used under the non-strict tracing context.
        x = torch.randn(2, 4, requires_grad=True)
        loss = torch.sin(x).sum()

        with torch.compiler._patch_engine_backward():
            with self.assertRaisesRegex(
                AssertionError,
                "_patch_engine_backward\\(\\) must be used under "
                "_non_strict_tracing_context\\(\\)",
            ):
                loss.backward()

    def test_patch_autograd_grad_does_not_leak_backward_tag(self):
        # [FUNCTION DOC] Under make_fx with the autograd.grad patch and node meta
        # preservation, checks that backward nodes get the autograd_backward tag but a neg
        # node after an autograd.grad call inside an annotation does not get it and keeps its
        # custom annotation metadata.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        x = torch.randn(2, 4, requires_grad=True)

        def fn(x):
            # [FUNCTION DOC] HELPER: inside an annotation ac_region_id 0, computes sin, calls
            # autograd.grad on its sum and returns neg of the sin result.
            with torch.fx.traceback.annotate({"ac_region_id": 0}):
                y = torch.sin(x)
                torch.autograd.grad(y.sum(), (x,))
                return torch.neg(y)

        with (
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_autograd_grad(),
            preserve_node_meta(),
        ):
            gm = make_fx(fn)(x)

        backward_nodes = [
            node for node in gm.graph.nodes if node.meta.get("autograd_backward", False)
        ]
        self.assertTrue(backward_nodes)

        neg_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.neg.default
        )
        self.assertEqual(len(neg_nodes), 1)
        self.assertNotIn("autograd_backward", neg_nodes[0].meta)
        self.assertEqual(neg_nodes[0].meta.get("custom", {}), {"ac_region_id": 0})

    def test_patch_engine_backward_does_not_leak_backward_tag(self):
        # [FUNCTION DOC] Same as the previous leakage test but for the patched engine backward:
        # backward nodes are tagged autograd_backward, while the neg node after the backward call
        # has no such tag and retains the ac_region_id custom metadata.
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        x = torch.randn(2, 4, requires_grad=True)

        def fn(x):
            # [FUNCTION DOC] HELPER: inside an annotation ac_region_id 0, computes sin, calls
            # backward on its sum and returns neg of the sin result.
            with torch.fx.traceback.annotate({"ac_region_id": 0}):
                y = torch.sin(x)
                y.sum().backward()
                return torch.neg(y)

        with (
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_engine_backward(),
            preserve_node_meta(),
        ):
            gm = make_fx(fn)(x)

        backward_nodes = [
            node for node in gm.graph.nodes if node.meta.get("autograd_backward", False)
        ]
        self.assertTrue(backward_nodes)

        neg_nodes = gm.graph.find_nodes(
            op="call_function", target=torch.ops.aten.neg.default
        )
        self.assertEqual(len(neg_nodes), 1)
        self.assertNotIn("autograd_backward", neg_nodes[0].meta)
        self.assertEqual(neg_nodes[0].meta.get("custom", {}), {"ac_region_id": 0})

    def test_patch_autograd_grad_mlp_has_single_contiguous_backward_region(self):
        # [FUNCTION DOC] Traces a train step of a 6-block residual MLP (no checkpoint) and asserts
        # the graph has exactly one contiguous backward region.
        class Block(nn.Module):
            # [CLASS DOC] HELPER: residual block x plus Linear(dim*2 to dim) of relu of
            # Linear(dim to dim*2).
            def __init__(self, dim):
                # [FUNCTION DOC] HELPER: creates the two linear layers.
                super().__init__()
                self.fc1 = nn.Linear(dim, dim * 2)
                self.fc2 = nn.Linear(dim * 2, dim)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: returns x plus fc2 of relu of fc1 of x.
                return x + self.fc2(torch.relu(self.fc1(x)))

        class Model(nn.Module):
            # [CLASS DOC] HELPER: sequence of depth residual Blocks held in a ModuleList.
            def __init__(self, dim=32, depth=6):
                # [FUNCTION DOC] HELPER: creates depth blocks of width dim.
                super().__init__()
                self.blocks = nn.ModuleList([Block(dim) for _ in range(depth)])

            def forward(self, x):
                # [FUNCTION DOC] HELPER: applies each block in order and returns the result.
                for block in self.blocks:
                    x = block(x)
                return x

        gm = self._trace_train_step(Model(), torch.randn(4, 16, 32))
        self.assertEqual(self._count_backward_regions(gm), 1)

    def _trace_train_step(self, mod, x):
        # [FUNCTION DOC] HELPER: traces a training step of mod with make_fx under non-strict
        # tracing, the autograd.grad patch and node-meta preservation. The step reparametrizes the
        # module with the traced parameter inputs, computes the summed output and returns gradients
        # with respect to all parameters. Returns the traced GraphModule taking the flattened
        # parameters followed by x.
        import torch.utils._pytree as pytree
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta
        from torch.nn.utils import stateless

        params = dict(mod.named_parameters())
        flat_params, params_spec = pytree.tree_flatten(params)
        params_len = len(flat_params)

        def train_step(*all_args):
            # [FUNCTION DOC] HELPER: rebuilds the parameter dictionary from the leading flat
            # arguments, runs the reparametrized module on the last argument, sums the output as
            # loss and returns torch.autograd.grad of the loss with respect to the parameters.
            p = pytree.tree_unflatten(list(all_args[:params_len]), params_spec)
            with stateless._reparametrize_module(mod, p):
                loss = mod(all_args[params_len]).sum()
                return torch.autograd.grad(loss, all_args[:params_len])

        full_args = (*flat_params, x)

        with (
            torch.compiler._non_strict_tracing_context(),
            torch.compiler._patch_autograd_grad(),
            preserve_node_meta(),
        ):
            return make_fx(train_step)(*full_args)

    def test_checkpoint_traces_through_eager_ac_under_non_strict(self):
        # [FUNCTION DOC] Traces a train step of a model with non-reentrant checkpoint and compares
        # the generated code with expected text: the forward mm is recomputed in the backward, so
        # mm appears twice before the gradient mm.
        class Model(nn.Module):
            # [CLASS DOC] HELPER: module with a 4x4 parameter w whose forward checkpoints sin of
            # x matmul w.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the parameter w as a random 4x4 tensor.
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of a lambda computing sin(x @ w).
                return checkpoint(
                    lambda x: torch.sin(x @ self.w), x, use_reentrant=False
                )

        gm = self._trace_train_step(Model(), torch.randn(2, 4))

        # Everything is recomputed in backward: mm_1 is the recomputed mm.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg1_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
    mm_1 = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg0_1 = None
    alias = torch.ops.aten.alias.default(mm_1);  mm_1 = None
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    cos = torch.ops.aten.cos.default(alias_1);  alias_1 = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(t, mul);  t = mul = None
    return (mm_2,)""",
        )

    def test_sac_traces_through_eager_ac_under_non_strict(self):
        # [FUNCTION DOC] Same as the previous test but with a selective-checkpoint policy (mm
        # PREFER_RECOMPUTE, all else MUST_SAVE) and expected generated code: mm is recomputed in
        # the backward while sin output is saved.
        def policy_fn(ctx, func, *args, **kwargs):
            # [FUNCTION DOC] HELPER: returns PREFER_RECOMPUTE for aten.mm and MUST_SAVE otherwise.
            if func == torch.ops.aten.mm.default:
                return CheckpointPolicy.PREFER_RECOMPUTE
            return CheckpointPolicy.MUST_SAVE

        def context_fn():
            # [FUNCTION DOC] HELPER: returns selective checkpoint contexts built from policy_fn.
            return create_selective_checkpoint_contexts(policy_fn)

        class Model(nn.Module):
            # [CLASS DOC] HELPER: module with a 4x4 parameter w whose forward checkpoints sin of x
            # matmul w using the SAC context_fn.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the parameter w as a random 4x4 tensor.
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of fn on x with
                # the SAC context_fn.
                def fn(x):
                    # [FUNCTION DOC] HELPER: sin of x matmul w.
                    return torch.sin(x @ self.w)

                return checkpoint(fn, x, use_reentrant=False, context_fn=context_fn)

        gm = self._trace_train_step(Model(), torch.randn(2, 4))

        # mm is PREFER_RECOMPUTE so it gets recomputed in backward (mm_1).
        # sin is MUST_SAVE so its output is saved via alias under make_fx's
        # default detach decomposition.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg1_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    alias = torch.ops.aten.alias.default(sin);  alias = None
    sum_1 = torch.ops.aten.sum.default(sin);  sin = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
    mm_1 = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg0_1 = None
    alias_1 = torch.ops.aten.alias.default(mm_1);  mm_1 = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    cos = torch.ops.aten.cos.default(alias_2);  alias_2 = None
    mul = torch.ops.aten.mul.Tensor(expand, cos);  expand = cos = None
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(t, mul);  t = mul = None
    return (mm_2,)""",
        )

    def test_checkpoint_with_rng_op_under_non_strict(self):
        # [FUNCTION DOC] Traces a train step (seeded with 42, under non-strict tracing with the
        # autograd.grad patch) of a model checkpointing a dropout region with preserve_rng_state,
        # and compares generated code with expected text. Documents that the traced graph holds two
        # independent bernoulli_ calls (forward and recompute) so masks differ at replay; this is a
        # known limitation of tracing eager checkpoint.
        from torch.fx.experimental.proxy_tensor import make_fx

        class Model(nn.Module):
            # [CLASS DOC] HELPER: module with a 4x4 parameter w whose forward checkpoints dropout
            # (p 0.5) of sin of x matmul w, preserving the RNG state.
            def __init__(self):
                # [FUNCTION DOC] HELPER: creates the parameter w as a random 4x4 tensor.
                super().__init__()
                self.w = nn.Parameter(torch.randn(4, 4))

            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of fn on x
                # with preserve_rng_state true.
                def fn(x):
                    # [FUNCTION DOC] HELPER: dropout (p 0.5) of sin of x matmul w.
                    return torch.nn.functional.dropout(torch.sin(x @ self.w), p=0.5)

                return checkpoint(fn, x, use_reentrant=False, preserve_rng_state=True)

        mod = Model()
        x = torch.randn(2, 4)

        import torch.utils._pytree as pytree
        from torch.nn.utils import stateless

        params = dict(mod.named_parameters())
        flat_params, params_spec = pytree.tree_flatten(params)
        params_len = len(flat_params)

        def train_step(*all_args):
            # [FUNCTION DOC] HELPER: rebuilds the parameter dictionary from leading arguments, runs
            # the reparametrized module on the last argument, sums as loss and returns gradients
            # with respect to the parameters.
            p = pytree.tree_unflatten(list(all_args[:params_len]), params_spec)
            with stateless._reparametrize_module(mod, p):
                loss = mod(all_args[params_len]).sum()
                return torch.autograd.grad(loss, all_args[:params_len])

        full_args = (*flat_params, x)

        torch.manual_seed(42)
        with torch.compiler._non_strict_tracing_context():
            gm = make_fx(train_step)(*full_args)

        # The traced graph has two independent bernoulli_ calls (forward and
        # recomputed). At replay time these produce different dropout masks
        # because get/set_rng_state are silently dropped by make_fx, so the
        # traced graph is NOT bitwise equivalent to eager. This is a known
        # limitation of tracing through eager checkpoint.
        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, arg0_1, arg1_1):
    mm = torch.ops.aten.mm.default(arg1_1, arg0_1)
    sin = torch.ops.aten.sin.default(mm);  mm = None
    empty_like = torch.ops.aten.empty_like.default(sin)
    bernoulli_ = torch.ops.aten.bernoulli_.float(empty_like);  empty_like = None
    div_ = torch.ops.aten.div_.Scalar(bernoulli_, 0.5);  bernoulli_ = None
    mul = torch.ops.aten.mul.Tensor(sin, div_);  sin = div_ = None
    sum_1 = torch.ops.aten.sum.default(mul);  mul = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format);  sum_1 = None
    expand = torch.ops.aten.expand.default(ones_like, [2, 4]);  ones_like = None
    mm_1 = torch.ops.aten.mm.default(arg1_1, arg0_1);  arg0_1 = None
    alias = torch.ops.aten.alias.default(mm_1)
    sin_1 = torch.ops.aten.sin.default(mm_1);  mm_1 = None
    empty_like_1 = torch.ops.aten.empty_like.default(sin_1);  sin_1 = None
    bernoulli__1 = torch.ops.aten.bernoulli_.float(empty_like_1);  empty_like_1 = None
    div__1 = torch.ops.aten.div_.Scalar(bernoulli__1, 0.5);  bernoulli__1 = None
    mul_1 = torch.ops.aten.mul.Tensor(expand, div__1);  expand = div__1 = None
    alias_1 = torch.ops.aten.alias.default(alias);  alias = None
    cos = torch.ops.aten.cos.default(alias_1);  alias_1 = None
    mul_2 = torch.ops.aten.mul.Tensor(mul_1, cos);  mul_1 = cos = None
    t = torch.ops.aten.t.default(arg1_1);  arg1_1 = None
    mm_2 = torch.ops.aten.mm.default(t, mul_2);  t = mul_2 = None
    return (mm_2,)""",
        )


class ActivationCheckpointingNestedCompileTests(torch._dynamo.test_case.TestCase):
    # [CLASS DOC] Tests that checkpoint recompute preserves nested fx trace policy when a
    # torch.compile function is called inside a checkpointed region while tracing with make_fx.
    @requires_gpu_and_triton
    def test_checkpoint_recompute_preserves_nested_fx_trace_policy(self):
        # [FUNCTION DOC] Traces with make_fx, under fake mode, a tracing context, node meta
        # preservation, non-strict tracing and a temporary disable of the nested-fx-trace error, a
        # model that checkpoints a call to a compiled sin-cos function; the step returns the
        # detached loss and the input gradient. Compares generated code with expected text,
        # showing the region recomputed in the backward. Requires GPU and triton; the model is
        # moved to the accelerator device.
        from torch._guards import tracing, TracingContext
        from torch._subclasses import FakeTensorMode
        from torch.fx.experimental.proxy_tensor import make_fx
        from torch.fx.traceback import preserve_node_meta

        compiled_f = torch.compile(lambda x: x.sin().cos(), fullgraph=True)  # noqa: UNSPECIFIED_BACKEND

        @contextlib.contextmanager
        def skip_nested_compile():
            # [FUNCTION DOC] HELPER: context manager that sets
            # torch._dynamo.config.error_on_nested_fx_trace to False for its body and restores the
            # previous value afterward, even on error.
            prev = torch._dynamo.config.error_on_nested_fx_trace
            torch._dynamo.config.error_on_nested_fx_trace = False
            try:
                yield
            finally:
                torch._dynamo.config.error_on_nested_fx_trace = prev

        class M(torch.nn.Module):
            # [CLASS DOC] HELPER: module whose forward runs the non-reentrant checkpoint of its
            # block method, which calls the compiled sin-cos function.
            def forward(self, x):
                # [FUNCTION DOC] HELPER: non-reentrant checkpoint of the block method on x.
                return checkpoint(self.block, x, use_reentrant=False)

            def block(self, x):
                # [FUNCTION DOC] HELPER: returns the result of the compiled sin-cos function on x.
                return compiled_f(x)

        m = getattr(M(), device_type)()
        x = torch.randn(8, device=device_type, requires_grad=True)

        def fn(x):
            # [FUNCTION DOC] HELPER: runs the model, sums it as y, computes the gradient of y with
            # respect to x and returns the detached y and the gradient.
            y = m(x).sum()
            (gx,) = torch.autograd.grad(y, (x,))
            return y.detach(), gx

        fake_mode = FakeTensorMode(
            allow_non_fake_inputs=True,
            shape_env=torch.fx.experimental.symbolic_shapes.ShapeEnv(),
        )
        fx_x = fake_mode.from_tensor(x, static_shapes=True)
        if x.requires_grad and not fx_x.requires_grad:
            fx_x.requires_grad_(True)

        ctx = TracingContext(fake_mode)

        with (
            fake_mode,
            tracing(ctx),
            preserve_node_meta(),
            skip_nested_compile(),
            torch.compiler._non_strict_tracing_context(),
        ):
            gm = make_fx(fn)(fx_x)

        self.assertExpectedInline(
            gm.code.strip(),
            """\
def forward(self, x_1):
    sin = torch.ops.aten.sin.default(x_1)
    cos = torch.ops.aten.cos.default(sin);  sin = None
    sum_1 = torch.ops.aten.sum.default(cos);  cos = None
    ones_like = torch.ops.aten.ones_like.default(sum_1, pin_memory = False, memory_format = torch.preserve_format)
    expand = torch.ops.aten.expand.default(ones_like, [8]);  ones_like = None
    alias = torch.ops.aten.alias.default(x_1)
    sin_1 = torch.ops.aten.sin.default(x_1);  x_1 = None
    alias_1 = torch.ops.aten.alias.default(sin_1);  sin_1 = None
    alias_2 = torch.ops.aten.alias.default(alias_1);  alias_1 = None
    sin_2 = torch.ops.aten.sin.default(alias_2);  alias_2 = None
    neg = torch.ops.aten.neg.default(sin_2);  sin_2 = None
    mul = torch.ops.aten.mul.Tensor(expand, neg);  expand = neg = None
    alias_3 = torch.ops.aten.alias.default(alias);  alias = None
    cos_1 = torch.ops.aten.cos.default(alias_3);  alias_3 = None
    mul_1 = torch.ops.aten.mul.Tensor(mul, cos_1);  mul = cos_1 = None
    alias_4 = torch.ops.aten.alias.default(sum_1);  sum_1 = None
    return (alias_4, mul_1)""",
        )


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
