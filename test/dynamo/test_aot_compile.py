# Owner(s): ["module: dynamo"]
# [MODULE DOC] Tests for Dynamo ahead-of-time (AOT) compilation and serialization.
# Defines helper models and a custom serializable backend, plus TestAOTCompile
# (torch.compile(...).aot_compile, save_compiled_function / load_compiled_function,
# module-level AOT compile, AOTI, DTensor, closures, external refs) and
# TestTritonKernelSerialization (kernel side table round trip). Import-time side
# effects: creates the global EPS tensor and MY_LAMBDA; no other global state.

import contextlib
import copy
import functools
import inspect
import multiprocessing as mp
import os
import pickle
import tempfile
import unittest
from collections import namedtuple
from collections.abc import Callable
from contextlib import contextmanager
from unittest.mock import patch

import torch
import torch._dynamo.testing
import torch._inductor.config
import torch._inductor.test_case
import torch.distributed as c10d
import torch.fx.traceback as fx_traceback
import torch.nn as nn
import torch.nn.functional as F
import torch.onnx.operators
import torch.utils.cpp_extension
from torch._dynamo.aot_compile import AOTCompiledModel, ModelInput, SerializableCallable
from torch._dynamo.aot_compile_types import BundledAOTAutogradSerializableCallable
from torch._dynamo.exc import PackageError, Unsupported
from torch._dynamo.package import DynamoCache
from torch._dynamo.precompile_context import PrecompileContext
from torch._functorch.aot_autograd import (
    aot_compile_joint_with_descriptors,
    aot_export_joint_with_descriptors,
)
from torch._guards import tracing, TracingContext
from torch._inductor.runtime.runtime_utils import cache_dir
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx._graph_pickler import GraphPickler
from torch.fx.passes.regional_inductor import regional_inductor
from torch.nn.attention.flex_attention import create_block_mask, flex_attention
from torch.testing._internal.common_utils import instantiate_parametrized_tests
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU
from torch.utils.checkpoint import checkpoint


MY_LAMBDA = lambda x: x + 1  # noqa: E731

EPS = torch.tensor(1e-7)


def aot_eager_regional_inductor():
    # [FUNCTION DOC] HELPER: Builds an aot_autograd backend whose forward and backward
    # compilers are both regional_inductor, so only regions annotated with
    # compile_with_inductor are lowered by Inductor. Takes no arguments.
    from torch._dynamo.backends.common import aot_autograd
    from torch.fx.passes.regional_inductor import regional_inductor

    return aot_autograd(
        fw_compiler=regional_inductor,
        bw_compiler=regional_inductor,
    )


class SingleCondModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Small module with two Linear layers (width d) and a
    # torch.cond between them that multiplies by 2 or 3 depending on whether the batch
    # size is below 32. Used to exercise cond predicate caching.
    def __init__(self, d=64):
        # [FUNCTION DOC] HELPER: Creates fc1 and fc2, both Linear(d, d); d defaults to 64.
        super().__init__()
        self.fc1 = torch.nn.Linear(d, d)
        self.fc2 = torch.nn.Linear(d, d)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies fc1, then a torch.cond on batch size < 32 choosing
        # the double or triple branch, then fc2; returns the result.
        x = self.fc1(x)

        def true_fn(x):
            # [FUNCTION DOC] HELPER: Cond true branch; returns the input scaled by 2.0.
            return x * 2.0

        def false_fn(x):
            # [FUNCTION DOC] HELPER: Cond false branch; returns the input scaled by 3.0.
            return x * 3.0

        x = torch.cond(x.shape[0] < 32, true_fn, false_fn, (x,))
        return self.fc2(x)


class MooType:
    # [CLASS DOC] HELPER: Plain user-defined object holding one attribute x. Lives at
    # module level so it can be referenced as an external global by compiled code.
    def __init__(self, x):
        # [FUNCTION DOC] HELPER: Stores the argument as attribute x.
        self.x = x


class CustomCompiledFunction(torch._dynamo.aot_compile.SerializableCallable):
    # [CLASS DOC] HELPER: Minimal SerializableCallable backend result. Holds a
    # GraphModule and its example inputs, runs the graph module when called, and
    # serializes the graph module via GraphPickler (with the other attributes pickled).
    def __init__(self, gm: torch.fx.GraphModule, example_inputs: list[torch.Tensor]):
        # [FUNCTION DOC] HELPER: Stores the graph module and example inputs as attributes.
        self.gm = gm
        self.example_inputs = example_inputs

    @classmethod
    def serialize_compile_artifacts(cls, fn) -> bytes:
        # [FUNCTION DOC] HELPER: Serializes a CustomCompiledFunction to bytes. Copies its
        # attribute mapping, pickles the graph module with GraphPickler (ops filter off),
        # then pickles the mapping. While pickling, GraphPickler.reducer_override is
        # temporarily patched so sympy functions with a torch unpickler are rebuilt by
        # name and any FakeTensorMode is replaced by None.
        import sympy

        from torch._subclasses import FakeTensorMode
        from torch.fx._graph_pickler import Options

        state = fn.__dict__.copy()
        graph_reducer_override = GraphPickler.reducer_override

        def _graph_reducer_override(self, obj):
            # [FUNCTION DOC] HELPER: Replacement reducer used while pickling the graph module.
            # Returns an unpickler and its handler name for sympy Function subclasses that
            # define a torch unpickler; returns a None-constructor for a FakeTensorMode;
            # otherwise defers to the original GraphPickler reducer_override.
            if (
                inspect.isclass(obj)
                and issubclass(obj, sympy.Function)
                and hasattr(obj, "_torch_unpickler")
            ):
                return obj._torch_unpickler, (obj._torch_handler_name,)
            if isinstance(obj, FakeTensorMode):
                return type(None), ()
            return graph_reducer_override(self, obj)

        with patch.object(GraphPickler, "reducer_override", _graph_reducer_override):
            state["gm"] = GraphPickler.dumps(state["gm"], Options(ops_filter=None))
        return pickle.dumps(state)

    @classmethod
    def deserialize_compile_artifacts(cls, data: bytes):
        # [FUNCTION DOC] HELPER: Inverse of serialize_compile_artifacts. Unpickles the
        # state, rebuilds the graph module with GraphPickler under a fresh FakeTensorMode,
        # recompiles it, and returns a new instance built from the state.
        state = pickle.loads(data)
        fake_mode = torch._subclasses.FakeTensorMode()
        state["gm"] = GraphPickler.loads(state["gm"], fake_mode)
        state["gm"].recompile()
        return cls(**state)

    def __call__(self, *args, **kwargs):
        # [FUNCTION DOC] HELPER: Runs the stored graph module on the given args and kwargs
        # and returns its result.
        return self.gm(*args, **kwargs)


class MultiHeadSelfAttention(nn.Module):
    # [CLASS DOC] HELPER: Grouped-query self attention module using flex_attention with
    # a causal block mask. Class attributes cache the compiled flex_attention (keyed by
    # compile spec) and the compiled create_block_mask so they are shared by instances.
    # Holds q/k/v/out projections; supports plain tensors and DTensors.
    _flex_attention_cache: dict = {}
    _create_block_mask_fn = None

    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        dropout: float = 0.0,
    ):
        # [FUNCTION DOC] HELPER: Builds the projections (k and v sized by the kv head
        # count; num_kv_heads falls back to num_heads when None), records enable_gqa when
        # head counts differ, and lazily fills the class-level caches with torch.compile
        # wrapped flex_attention and create_block_mask (fullgraph, static shapes).
        # KNOWN ISSUE: dropout is stored but never applied in forward.
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.num_kv_heads = num_kv_heads if num_kv_heads is not None else num_heads
        self.head_dim = embed_dim // num_heads
        self.kv_dim = self.num_kv_heads * self.head_dim
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, self.kv_dim)
        self.v_proj = nn.Linear(embed_dim, self.kv_dim)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = dropout
        self.enable_gqa = self.num_heads != self.num_kv_heads

        # Compile flex_attention with default compile_spec
        # This creates a nested torch.compile that triggers flex_attention_hop
        compile_spec = {
            "mode": "default",
            "fullgraph": True,
            "dynamic": False,
        }
        compile_key = tuple(sorted(compile_spec.items()))
        if compile_key not in MultiHeadSelfAttention._flex_attention_cache:
            MultiHeadSelfAttention._flex_attention_cache[compile_key] = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                flex_attention, **compile_spec
            )
        self._flex_attention = MultiHeadSelfAttention._flex_attention_cache[compile_key]

        # Also compile create_block_mask
        if MultiHeadSelfAttention._create_block_mask_fn is None:
            MultiHeadSelfAttention._create_block_mask_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                create_block_mask, dynamic=False, fullgraph=True
            )

    def _shape_heads(self, x, B, S, num_heads):
        # [FUNCTION DOC] HELPER: Reshapes a (B, S, hidden) tensor to (B, num_heads, S,
        # head_dim) by viewing then swapping the sequence and head dimensions.
        return x.view(B, S, num_heads, self.head_dim).transpose(1, 2)

    def _forward_local(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        block_mask,
    ) -> torch.Tensor:
        # [FUNCTION DOC] HELPER: Calls the compiled flex_attention on local query, key and
        # value with the given block mask and the GQA flag, under an fx annotation marking
        # the region to be compiled with Inductor. Returns the attention output.
        with fx_traceback.annotate({"compile_with_inductor": 1}):
            return self._flex_attention(
                query=query,
                key=key,
                value=value,
                block_mask=block_mask,
                enable_gqa=self.enable_gqa,
            )

    def _qkv_to_local(
        self,
        query,
        key,
        value,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # [FUNCTION DOC] HELPER: Converts DTensor q, k, v to local tensors, choosing the
        # gradient placements: sharded on dim 0 or 1 (or replicated) on all three keeps the
        # placements; query sharded on dim 2 with replicated key and value gives key and
        # value Partial gradient placements; anything else raises NotImplementedError.
        # KNOWN ISSUE: the return statement is indented inside the placement loop, so only
        # the first mesh dimension is examined and returned; multi-dim meshes are ignored.
        from torch.distributed.tensor import Partial

        q_grad_placements = []
        kv_grad_placements = []

        for query_p, key_p, value_p in zip(
            query.placements, key.placements, value.placements
        ):
            if (
                (
                    query_p.is_shard(dim=0)
                    and key_p.is_shard(dim=0)
                    and value_p.is_shard(dim=0)
                )
                or (
                    query_p.is_shard(dim=1)
                    and key_p.is_shard(dim=1)
                    and value_p.is_shard(dim=1)
                )
                or (
                    query_p.is_replicate()
                    and key_p.is_replicate()
                    and value_p.is_replicate()
                )
            ):
                q_grad_placements.append(query_p)
                kv_grad_placements.append(key_p)
            elif (
                query_p.is_shard(dim=2)
                and key_p.is_replicate()
                and value_p.is_replicate()
            ):
                q_grad_placements.append(query_p)
                kv_grad_placements.append(Partial())
            else:
                raise NotImplementedError(
                    "Currently only supports Data Parallel, Tensor Parallel, "
                    "and all-gather based Context Parallel."
                )

            return (
                query.to_local(grad_placements=q_grad_placements),
                key.to_local(grad_placements=kv_grad_placements),
                value.to_local(grad_placements=kv_grad_placements),
            )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Projects x to q, k, v, builds a causal block mask via the
        # compiled create_block_mask, and runs flex attention (locally, or via
        # _qkv_to_local and DTensor.from_local when any of q, k, v is a DTensor). Returns
        # the output projection of the re-merged heads.
        from torch.distributed.tensor import DTensor

        B, S, _ = x.shape

        q = self._shape_heads(self.q_proj(x), B, S, self.num_heads)
        k = self._shape_heads(self.k_proj(x), B, S, self.num_kv_heads)
        v = self._shape_heads(self.v_proj(x), B, S, self.num_kv_heads)

        # Create block_mask inside forward to test cross-compilation
        def causal_mask(b, h, q_idx, kv_idx):
            # [FUNCTION DOC] HELPER: Causal mask predicate; True when the query index is at
            # least the key/value index. Ignores batch and head indices.
            return q_idx >= kv_idx

        with fx_traceback.annotate({"compile_with_inductor": 1}):
            block_mask = MultiHeadSelfAttention._create_block_mask_fn(
                causal_mask, B, self.num_heads, S, S, device=x.device
            )

        if not any(isinstance(t, DTensor) for t in (q, k, v)):
            attn_out = self._forward_local(q, k, v, block_mask)
        else:
            q_local, k_local, v_local = self._qkv_to_local(q, k, v)
            attn_out_local = self._forward_local(q_local, k_local, v_local, block_mask)
            attn_out = DTensor.from_local(
                attn_out_local,
                device_mesh=q.device_mesh,
                placements=q.placements,
            )

        attn_out = attn_out.transpose(1, 2).contiguous().view(B, S, self.embed_dim)
        return self.out_proj(attn_out)


class TransformerBlock(nn.Module):
    # [CLASS DOC] HELPER: Pre-norm transformer block: LayerNorm, self attention with
    # residual, then a LayerNorm plus MLP (Linear, GELU, Linear, Dropout) under
    # activation checkpointing with residual.
    def __init__(
        self,
        embed_dim: int,
        num_heads: int,
        num_kv_heads: int,
        mlp_ratio: float = 4.0,
        dropout: float = 0.0,
    ):
        # [FUNCTION DOC] HELPER: Builds layer norms, the attention module, dropout and the
        # MLP with hidden size embed_dim times mlp_ratio. The dropout argument is used only
        # for the block's own dropouts, not passed to attention.
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, num_kv_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(embed_dim)
        hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Adds the dropped-out attention output to x, then adds the
        # non-reentrant checkpointed MLP output; returns the new activations.
        x = x + self.dropout1(self.attn(self.ln1(x)))
        x = x + checkpoint(lambda inp: self.mlp(self.ln2(inp)), x, use_reentrant=False)
        return x


class Transformer(nn.Module):
    # [CLASS DOC] HELPER: Small decoder-style model: embedding, learned positional
    # parameter, a list of TransformerBlock layers, final LayerNorm and an untied
    # bias-free output head. Stores the device mesh used to redistribute inputs.
    def __init__(
        self,
        vocab_size: int,
        embed_dim: int,
        num_heads: int,
        num_layers: int,
        max_seq_len: int,
        num_kv_heads: int,
        device_mesh=None,
    ):
        # [FUNCTION DOC] HELPER: Builds all submodules and stores device_mesh; positional
        # embedding is a zero-initialized parameter of shape (1, max_seq_len, embed_dim).
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim)
        self.pos_embed = nn.Parameter(torch.zeros(1, max_seq_len, embed_dim))
        self.layers = nn.ModuleList(
            [
                TransformerBlock(embed_dim, num_heads, num_kv_heads)
                for _ in range(num_layers)
            ]
        )
        self.ln_f = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, vocab_size, bias=False)
        self.device_mesh = device_mesh

    def forward(self, input_ids):
        # [FUNCTION DOC] HELPER: Redistributes input_ids (must be a DTensor) to Replicate on
        # the stored mesh, embeds and adds positions, runs all blocks, final norm and head,
        # and returns the logits.
        from torch.distributed.tensor import Replicate

        input_ids = input_ids.redistribute(self.device_mesh, [Replicate()])
        x = self.embed(input_ids) + self.pos_embed[:, : input_ids.shape[1], :]

        for block in self.layers:
            x = block(x)

        x = self.ln_f(x)
        logits = self.head(x)
        return logits


class SimpleLinearModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module wrapping a single Linear(3, 3); the standard tiny model
    # used across the AOT tests.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the Linear(3, 3) layer.
        super().__init__()
        self.linear = torch.nn.Linear(3, 3)

    def forward(self, x):
        # [FUNCTION DOC] HELPER: Applies the linear layer to x and returns the result.
        return self.linear(x)


class RepeatInterleaveModule(torch.nn.Module):
    # [CLASS DOC] HELPER: Module that splits the last dim into two chunks and repeats
    # each element of the first chunk twice along the last dim.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns the first of two last-dim chunks of x with
        # repeat_interleave(2) applied along the last dim.
        chunk = x.chunk(2, dim=-1)
        y = chunk[0]
        y_repeat = y.repeat_interleave(2, dim=-1)
        return y_repeat


class MultiModalMixin(torch.nn.Module):
    # [CLASS DOC] HELPER: Mixin whose forward delegates to the next class in the MRO,
    # used to build a forward that has a super call (closure over __class__).
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Calls the next forward in the MRO with x and returns it.
        return super().forward(x)


class TextModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Module whose forward adds one to its input.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Returns x plus one.
        return x + 1


class TestVLLMModel(MultiModalMixin, TextModel):
    # [CLASS DOC] HELPER: Model combining MultiModalMixin and TextModel to reproduce a
    # vLLM-style multiple-inheritance forward with super calls.
    def forward(self, x):
        # [FUNCTION DOC] HELPER: Delegates to the next forward in the MRO; result is x plus
        # one after passing through the mixin.
        return super().forward(x)


def _subprocess_entry(fn, queue):
    # [FUNCTION DOC] HELPER: Child-process entry point. Runs the callable fn; on any
    # exception puts (exception name, message, formatted traceback) on the queue and
    # re-raises; on success puts None on the queue.
    try:
        fn()
    except BaseException as exc:
        import traceback

        queue.put((type(exc).__name__, str(exc), traceback.format_exc()))
        raise
    else:
        queue.put(None)


def _run_in_subprocess(fn):
    # [FUNCTION DOC] HELPER: Runs fn in a spawned child process and waits for it. Raises
    # AssertionError with the child's error name, message and traceback if the child
    # reported a failure. KNOWN ISSUE: the queue read happens after join and blocks
    # forever if the child dies without reporting (e.g. a hard crash).
    ctx = mp.get_context("spawn")
    queue = ctx.Queue()
    proc = ctx.Process(target=_subprocess_entry, args=(fn, queue))
    proc.start()
    proc.join()
    result = queue.get()
    if result is not None:
        name, msg, tb = result
        raise AssertionError(f"Subprocess failure ({name}: {msg})\n{tb}")


def _subprocess_disable_guard_check():
    # [FUNCTION DOC] HELPER: Subprocess body. With AOT compile enabled, AOT-compiles a
    # tensor add, flips the global grad mode, and expects the call to fail with a
    # GuardManager check error. After disable_guard_check the call must match eager.
    # Always restores the previous grad mode; raises AssertionError on mismatch.
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        prev_grad = torch.is_grad_enabled()
        try:
            torch.set_grad_enabled(not prev_grad)
            try:
                compiled_fn(*inputs)
            except RuntimeError as exc:  # pragma: no cover
                if "GuardManager check failed" not in str(exc):
                    raise
            else:  # pragma: no cover
                raise AssertionError("Guard check should have failed")
            compiled_fn.disable_guard_check()
            actual = compiled_fn(*inputs)
            if not torch.allclose(actual, expected):
                raise AssertionError(
                    f"Expected tensors to be close, got {actual} vs {expected}"
                )
        finally:
            torch.set_grad_enabled(prev_grad)


def _subprocess_grad_mode_after_prior_compile():
    # [FUNCTION DOC] HELPER: Subprocess body. AOT-compiles one function, resets dynamo,
    # then AOT-compiles a subtraction under no_grad and checks that, called under
    # no_grad, it matches eager; raises AssertionError otherwise.
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):

        def warmup_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y (warmup compile target).
            return x + y

        def target_fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x minus y (the function under test).
            return x - y

        torch.compile(warmup_fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        torch._dynamo.reset()

        with torch.no_grad():
            compiled_fn = torch.compile(target_fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
                ((torch.randn(3, 4), torch.randn(3, 4)), {})
            )

        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        with torch.no_grad():
            actual = compiled_fn(*inputs)
            expected = target_fn(*inputs)
            if not torch.allclose(actual, expected):
                raise AssertionError(
                    f"Expected tensors to be close, got {actual} vs {expected}"
                )


def _subprocess_aot_compile_module():
    # [FUNCTION DOC] HELPER: Subprocess body. Compiles a SimpleLinearModule with the
    # inductor backend, AOT-compiles it for an eval/no_grad input and a train input,
    # checks eval results match eager under fail_on_recompile and that backward runs,
    # saves the compiled module to a temp file, resets dynamo, reloads it into a fresh
    # compiled module and checks eval output again. Raises AssertionError on mismatch.
    import torch
    from torch._dynamo import config

    with config.patch(enable_aot_compile=True):
        mod = SimpleLinearModule()
        model = torch.compile(
            mod,
            fullgraph=True,
            backend="inductor",
            options={
                "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
            },
        )

        @contextmanager
        def train_mode(mdl):
            # [FUNCTION DOC] HELPER: Context manager that puts the module in train mode and then
            # yields.
            mdl.train()
            yield

        @contextmanager
        def eval_mode(mdl):
            # [FUNCTION DOC] HELPER: Context manager that puts the module in eval mode and then
            # yields.
            mdl.eval()
            yield

        inputs = [
            ModelInput(
                args=(torch.randn(3, 3),),
                kwargs={},
                contexts=[torch.no_grad(), eval_mode(model)],
            ),
            ModelInput(
                args=(torch.randn(3, 3),), kwargs={}, contexts=[train_mode(model)]
            ),
        ]
        if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
            raise AssertionError(
                f"Expected OptimizedModule, got {type(model).__name__}"
            )
        model._aot_compile(inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            model.eval()
            eager_inputs = (torch.randn(3, 3),)
            expected = mod(*eager_inputs)
            actual = model(*eager_inputs)
            if not torch.allclose(expected, actual):
                raise AssertionError(
                    f"Expected tensors to be close, got {actual} vs {expected}"
                )
            model.train()
            expected.sum().backward()

        with tempfile.TemporaryDirectory() as tmpdir:
            path = os.path.join(tmpdir, "model.pt")
            model._save_aot_compiled_module(path)
            torch._dynamo.reset()
            model = torch.compile(
                mod,
                fullgraph=True,
                backend="inductor",
                options={
                    "guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe,
                },
            )
            if not isinstance(model, torch._dynamo.eval_frame.OptimizedModule):
                raise AssertionError(
                    f"Expected OptimizedModule, got {type(model).__name__}"
                )
            with open(path, "rb") as f:
                data = f.read()
                model._load_aot_compiled_module(data)

            with torch.compiler.set_stance("fail_on_recompile"):
                model.eval()
                eager_inputs = (torch.randn(3, 3),)
                expected = mod(*eager_inputs)
                actual = model(*eager_inputs)
                if not torch.allclose(expected, actual):
                    raise AssertionError(
                        f"Expected tensors to be close, got {actual} vs {expected}"
                    )


class RedistributeModel(torch.nn.Module):
    # [CLASS DOC] HELPER: Module with a Linear(32, 32) whose forward also redistributes
    # a DTensor input to fully replicated placements on a 2D mesh.
    def __init__(self):
        # [FUNCTION DOC] HELPER: Creates the Linear(32, 32) layer.
        super().__init__()
        self.linear = torch.nn.Linear(32, 32)

    def forward(self, x, d_x, mesh):
        # [FUNCTION DOC] HELPER: Applies the linear layer to x and redistributes d_x on the
        # given mesh to (Replicate, Replicate). Returns the pair (linear output, result).
        x = self.linear(x)

        # need to do local import since tests don't always have c10d
        # and precompile needs this class to be available at the module
        # level.
        from torch.distributed.tensor import Replicate

        y = d_x.redistribute(mesh, placements=(Replicate(), Replicate()))

        return x, y


def wrap_forward_function(fn: Callable):
    # [FUNCTION DOC] HELPER: Returns a thin wrapper around fn that forwards all args and
    # kwargs. Only the doc, annotations and type params are copied from fn (not the
    # name or wrapped attribute), so the wrapper hides the original callable.
    @functools.wraps(fn, assigned=("__doc__", "__annotations__", "__type_params__"))
    def wrapped(*args, **kwargs):
        # [FUNCTION DOC] HELPER: Calls the captured fn with the same args and kwargs.
        return fn(*args, **kwargs)

    return wrapped


@torch._dynamo.config.patch("enable_aot_compile", True)
@instantiate_parametrized_tests
class TestAOTCompile(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Main test case for AOT compile. The class decorators enable the
    # enable_aot_compile config and expand parametrized tests. Tests compile a function
    # or module, compare with eager, save it to a per-test path under the Inductor cache
    # dir, reset dynamo, reload it, and check results under fail_on_recompile.
    def path(self):
        # [FUNCTION DOC] HELPER: Returns the per-test model file path (a model.pt inside a
        # package directory named by the test id under the Inductor cache dir), creating
        # the directory if needed.
        path = os.path.join(cache_dir(), f"package_{self.id()}")
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, "model.pt")

    def setUp(self):
        # [FUNCTION DOC] HELPER: Test setup. Calls the parent setUp, resets dynamo, clears
        # the dynamo counters and clears DynamoCache and PrecompileContext.
        super().setUp()
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        DynamoCache.clear()
        PrecompileContext.clear()

    def test_aot_compile_basic_fn(self):
        # [FUNCTION DOC] Test: AOT compiles an add using the custom serializable backend,
        # checks it matches eager, saves, resets, reloads under fail_on_recompile and
        # checks the loaded function again.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        compiled_fn = torch.compile(fn, fullgraph=True, backend=backend).aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_autocast_guard_reload(self):
        # [FUNCTION DOC] Test: AOT compiles under a CPU bfloat16 autocast context, saves,
        # resets and reloads, then runs under autocast again and checks the output equals
        # the eager autocast result (autocast guard survives reload).
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus x times one.
            return x + 1 * x

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        x = torch.randn(3, 4)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            compiled_fn = torch.compile(
                fn, fullgraph=True, backend=backend
            ).aot_compile(((x,), {}))
            expected = fn(x)
            self.assertEqual(expected, compiled_fn(x))

        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        with torch.amp.autocast("cpu", dtype=torch.bfloat16):
            actual = compiled_fn(x)
        self.assertEqual(expected, actual)

    def test_aot_compile_basic_forward(self):
        # [FUNCTION DOC] Test: AOT compiles a module's forward (the compiled function takes
        # the module as first argument) with the custom backend, checks against eager, then
        # saves, resets and reloads, and checks again.
        mod = SimpleLinearModule()

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        compiled_fn = torch.compile(
            mod,
            fullgraph=True,
            backend=backend,
        ).forward.aot_compile(((torch.randn(3, 3),), {}))
        inputs = (torch.randn(3, 3),)
        expected = mod(*inputs)
        actual = compiled_fn(mod, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_repeat_interleave(self):
        # [FUNCTION DOC] Test: AOT compiles the repeat_interleave module forward on a (2, 4)
        # input with the custom backend, checks against eager, then saves, reloads and
        # checks again. The mark_dynamic call is commented out in the code.
        mod = RepeatInterleaveModule()

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        inputs = (torch.randn(2, 4),)

        # The first dim should be dynamic to repro the issue of repeat_interleave
        # torch._dynamo.mark_dynamic(inputs[0], [0])

        compiled_fn = torch.compile(
            mod,
            fullgraph=True,
            backend=backend,
        ).forward.aot_compile((inputs, {}))

        expected = mod(*inputs)
        actual = compiled_fn(mod, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)

    def test_code_cache(self):
        # [FUNCTION DOC] Test: round-trips a code object through SerializedCode after
        # adding a dict constant to co_consts, and checks the re-serialized form equals the
        # original.
        from torch._dynamo.package import SerializedCode

        def foo():
            # [FUNCTION DOC] HELPER: Empty function whose code object is serialized.
            pass

        serialized_code = SerializedCode.from_code_object(foo.__code__)
        object.__setattr__(
            serialized_code, "co_consts", serialized_code.co_consts + ({1: 2},)
        )

        new_code = SerializedCode.to_code_object(serialized_code)
        new_serialized_code = SerializedCode.from_code_object(new_code)
        self.assertEqual(new_serialized_code, serialized_code)

    def test_decorated_function_aot(self):
        # [FUNCTION DOC] Test: AOT compiles a function wrapped by a decorator that does not
        # use functools.wraps (and asserts first dim > 1), with the custom backend, under
        # fail_on_recompile, and checks the result against eager.
        def check_inputs(fn):
            # [FUNCTION DOC] HELPER: Decorator returning a wrapper that checks each positional
            # argument has first dim greater than one and then calls fn.
            def _fn(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper doing the shape check then calling the function.
                for arg in args:
                    assert arg.shape[0] > 1  # noqa: S101

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns twice x plus twice y.
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)

    def test_eager_backend(self):
        # [FUNCTION DOC] Test: AOT compiles an add with the eager backend, checks against
        # eager, then saves, resets, reloads under fail_on_recompile and checks again.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True, backend="eager").aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_eager_backend(self):
        # [FUNCTION DOC] Test: same flow as the eager backend test but with the aot_eager
        # backend (compile, compare, save, reset, reload, compare).
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        compiled_fn = torch.compile(
            fn, fullgraph=True, backend="aot_eager"
        ).aot_compile(((torch.randn(3, 4), torch.randn(3, 4)), {}))
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_decorated_function_with_functools_wrap_aot(self):
        # [FUNCTION DOC] Test: like the decorated function test but the decorator uses
        # functools.wraps; AOT compile under fail_on_recompile must match eager.
        def check_inputs(fn):
            # [FUNCTION DOC] HELPER: Decorator returning a functools.wraps wrapper that checks
            # each positional argument has first dim greater than one, then calls fn.
            @functools.wraps(fn)
            def _fn(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper doing the shape check then calling the function.
                for arg in args:
                    assert arg.shape[0] > 1  # noqa: S101

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns twice x plus twice y.
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            compiled_fn = torch.compile(
                foo,
                fullgraph=True,
                backend=backend,
            ).aot_compile((example_inputs, {}))
            actual = compiled_fn(*example_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_disable_guard_check(self):
        # [FUNCTION DOC] Test: runs the disable_guard_check subprocess scenario in a
        # spawned process.
        _run_in_subprocess(_subprocess_disable_guard_check)

    def test_aot_compile_grad_mode_after_prior_compile(self):
        # [FUNCTION DOC] Test: runs the grad-mode-after-prior-compile subprocess scenario in
        # a spawned process.
        _run_in_subprocess(_subprocess_grad_mode_after_prior_compile)

    def test_aot_compile_torch_func_vmap_grad(self):
        # [FUNCTION DOC] Test: AOT compiles a vmap of grad of a vector norm (float64, static
        # shapes), compares with eager, then saves, resets, reloads and compares again.
        import torch.func as tf

        def value(pos):
            # [FUNCTION DOC] HELPER: Returns the norm of the difference between the second and
            # first rows of pos.
            return torch.linalg.norm(pos[1] - pos[0])

        batched_grad = tf.vmap(tf.grad(value, argnums=0), in_dims=(0,))
        x = torch.randn(64, 2, 3, dtype=torch.float64)
        compiled_fn = torch.compile(
            batched_grad, fullgraph=True, dynamic=False
        ).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((x,), {})
        )
        expected = batched_grad(x)
        actual = compiled_fn(x)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                loaded_fn = torch.compiler.load_compiled_function(f)
            actual = loaded_fn(x)
            self.assertEqual(expected, actual)

    def test_aot_compile_source_info(self):
        # [FUNCTION DOC] Test: checks source_info() on an AOT compiled function returns a
        # SourceInfo with two inlined sources whose first module is this test module, both
        # before saving and after save and load.
        from torch._dynamo.package import SourceInfo

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Applies the global MY_LAMBDA to x and adds y.
            return MY_LAMBDA(x) + y

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )

        source_info = compiled_fn.source_info()
        self.assertIsInstance(source_info, SourceInfo)
        self.assertEqual(len(source_info.inlined_sources), 2)
        self.assertEqual(next(iter(source_info.inlined_sources)).module, __name__)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        source_info = compiled_fn.source_info()
        self.assertIsInstance(source_info, SourceInfo)
        self.assertEqual(len(source_info.inlined_sources), 2)
        self.assertEqual(next(iter(source_info.inlined_sources)).module, __name__)

    def test_regional_inductor_backend(self):
        # [FUNCTION DOC] Test: AOT compiles with the regional inductor backend where one
        # annotated region is lowered by Inductor, checks forward and backward run and
        # match eager, saves and reloads, then checks forward and backward again.
        import torch.fx.traceback as fx_traceback

        def fn(x, y):
            # [FUNCTION DOC] HELPER: sin(x), then a multiply by y and add of one inside a
            # compile_with_inductor annotation, then sin of the result.
            sin = torch.sin(x)
            # Mark this region to be compiled with inductor
            with fx_traceback.annotate({"compile_with_inductor": 0}):
                mul = sin * y
                add = mul + 1
            return torch.sin(add)

        def make_inputs():
            # [FUNCTION DOC] HELPER: Returns two random (3, 4) tensors that require grad.
            return (
                torch.randn(3, 4, requires_grad=True),
                torch.randn(3, 4, requires_grad=True),
            )

        compiled_fn = torch.compile(
            fn, fullgraph=True, backend=aot_eager_regional_inductor()
        ).aot_compile((make_inputs(), {}))
        test_inputs = make_inputs()
        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))
        compiled_fn(*test_inputs).sum().backward()
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)

        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))
        compiled_fn(*test_inputs).sum().backward()

    def test_aot_compile_graph_break_error_fmt(self):
        # [FUNCTION DOC] Test: AOT compiling a function with an explicit graph break and
        # fullgraph raises Unsupported, and the munged error text must match the expected
        # message (user-inserted graph break, hint, link and user code frame).
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Function with a torch._dynamo.graph_break between two
            # doubled adds; used to trigger the Unsupported error.
            a = x + x
            torch._dynamo.graph_break()
            b = y + y
            c = a + b
            return c

        self.assertExpectedInlineMunged(
            Unsupported,
            lambda: torch.compile(foo, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
                ((torch.ones(3), torch.ones(3)), {})
            ),
            """\
Call to `torch._dynamo.graph_break()`
  Explanation: User-inserted graph break. Message: None
  Hint: Remove the `torch._dynamo.graph_break()` call.

  Developer debug context: Called `torch._dynamo.graph_break()` with args `[]`, kwargs `{}`

 For more details about this graph break, please visit: https://meta-pytorch.github.io/compile-graph-break-site/gb/gb0025.html

from user code:
   File "test_aot_compile.py", line N, in foo
    torch._dynamo.graph_break()""",
        )

    def test_guard_filter_override_aot(self):
        # [FUNCTION DOC] Test: a guard_filter_fn that keeps every guard makes AOT compile
        # raise PackageError mentioning that the CLOSURE_MATCH guard cannot be serialized.
        def check_inputs(fn):
            # [FUNCTION DOC] HELPER: Decorator returning a wrapper that checks first dims and
            # calls fn.
            def _fn(*args, **kwargs):
                # [FUNCTION DOC] HELPER: Wrapper doing the shape check then calling the function.
                for arg in args:
                    assert arg.shape[0] > 1  # noqa: S101

                return fn(*args, **kwargs)

            return _fn

        @check_inputs
        def foo(x, y):
            # [FUNCTION DOC] HELPER: Returns twice x plus twice y.
            a = x + x
            b = y + y
            c = a + b
            return c

        example_inputs = (torch.ones(3), torch.ones(3))
        expected = foo(*example_inputs)  # noqa: F841

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend returning a CustomCompiledFunction for the graph.
            return CustomCompiledFunction(gm, example_inputs)

        with torch.compiler.set_stance("fail_on_recompile"):
            with self.assertRaisesRegex(
                PackageError,
                "CLOSURE_MATCH guard cannot be serialized.",
            ):
                compiled_fn = torch.compile(  # noqa: F841
                    foo,
                    fullgraph=True,
                    backend=backend,
                    options={
                        "guard_filter_fn": lambda guard_entries: [
                            True for g in guard_entries
                        ]
                    },
                ).aot_compile((example_inputs, {}))

    def test_aot_compile_basic_fn_inductor(self):
        # [FUNCTION DOC] Test: AOT compiles an add with the inductor backend, compares with
        # eager, saves, resets, reloads under fail_on_recompile and compares again.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        compiled_fn = torch.compile(fn, fullgraph=True, backend="inductor").aot_compile(
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_module(self):
        # [FUNCTION DOC] Test: runs the AOT compile module save/load subprocess scenario in
        # a spawned process.
        _run_in_subprocess(_subprocess_aot_compile_module)

    def test_aot_module_simplified_serializable_autograd(self):
        # [FUNCTION DOC] Test: AOT compiles a Linear module forward (training graph) with
        # inductor and checks the backend result is a
        # BundledAOTAutogradSerializableCallable whose inner compiled_fn has a serialize
        # attribute.
        mod = SimpleLinearModule()
        compiled_fn: SerializableCallable = torch.compile(
            mod, fullgraph=True, backend="inductor"
        ).forward.aot_compile(((torch.randn(3, 3),), {}))
        backend_result = compiled_fn._artifacts.compiled_fn
        self.assertTrue(
            isinstance(
                backend_result,
                torch._dynamo.aot_compile.BundledAOTAutogradSerializableCallable,
            )
        )
        if not hasattr(backend_result.compiled_fn, "serialize"):
            raise AssertionError("Expected compiled_fn to have 'serialize' attribute")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_aot_compile_portable_guards_unsafe(self):
        # [FUNCTION DOC] Test: AOT compiles a function taking a sequence with the
        # keep_portable_guards_unsafe guard filter, runs it on a namedtuple input, checks
        # against eager, then saves, loads and checks again.
        def fn(xy):
            # [FUNCTION DOC] HELPER: Returns the sum of the first two items of xy.
            return xy[0] + xy[1]

        compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            fn,
            fullgraph=True,
            options={"guard_filter_fn": torch.compiler.keep_portable_guards_unsafe},
        ).aot_compile((((torch.randn(3, 4), torch.randn(3, 4)),), {}))
        Tup = namedtuple("Tup", ["x", "y"])

        inputs = Tup(torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(inputs)
        actual = compiled_fn(inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(inputs)
        self.assertEqual(expected, actual)

    def test_aot_module_simplified_serializable_inference(self):
        # [FUNCTION DOC] Test: same as the autograd serializable test but for a sin function
        # compiled for inference; checks the backend result type and serialize attribute.
        def fn(x):
            # [FUNCTION DOC] HELPER: Returns sin of x.
            return x.sin()

        compiled_fn: SerializableCallable = torch.compile(
            fn, fullgraph=True, backend="inductor"
        ).aot_compile(((torch.randn(3, 3),), {}))
        backend_result = compiled_fn._artifacts.compiled_fn
        self.assertTrue(
            isinstance(
                backend_result,
                torch._dynamo.aot_compile.BundledAOTAutogradSerializableCallable,
            )
        )
        if not hasattr(backend_result.compiled_fn, "serialize"):
            raise AssertionError("Expected compiled_fn to have 'serialize' attribute")
        self.assertIsNotNone(backend_result.compiled_fn.serialize)

    def test_aot_cache_predicate_not_pickleable(self):
        # [FUNCTION DOC] Test: with cond marked cacheable and autograd and FX graph caching
        # enabled, compiles the cond model with dynamic shapes for batch sizes 16 and 64
        # and checks results match eager and that the fxgraph_cache_bypass counter stays
        # zero (the SymBool predicate is picklable). Restores the cacheable flag after.
        import torch._functorch.config as functorch_config
        import torch._inductor.config as inductor_config

        model = SingleCondModel().eval()

        old_cacheable = torch.ops.higher_order.cond._cacheable
        torch.ops.higher_order.cond._cacheable = True
        try:
            with (
                functorch_config.patch(
                    enable_autograd_cache=True,
                    force_non_lazy_backward_lowering=True,
                    strict_autograd_cache=True,
                ),
                inductor_config.patch(
                    fx_graph_cache=True,
                    fx_graph_remote_cache=False,
                ),
            ):
                compiled = torch.compile(model, backend="inductor", dynamic=True)

                # Test both branches of the cond predicate (x.shape[0] < 32).
                for batch_size in (16, 64):
                    inp = torch.randn(batch_size, 64)
                    expected = model(inp)
                    actual = compiled(inp)
                    self.assertEqual(expected, actual)

                # If the SymBool predicate is not pickleable, the FxGraphCache
                # silently bypasses instead of caching. Assert no bypass occurred.
                self.assertEqual(
                    torch._dynamo.utils.counters["inductor"]["fxgraph_cache_bypass"], 0
                )
        finally:
            torch.ops.higher_order.cond._cacheable = old_cacheable

    def test_fullgraph_capture_with_pytree_module(self):
        # [FUNCTION DOC] Test: captures a module with a pytree (dict, tensor, tensor) input
        # and dict output via dynamo_graph_capture_for_export and checks the captured
        # module matches the eager module on fresh inputs.
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        class Module(torch.nn.Module):
            # [CLASS DOC] HELPER: Module with four Linear(3, 3) layers whose forward takes a
            # tuple (dict with a and b, tensor, tensor) and returns a dict of four results.
            def __init__(self):
                # [FUNCTION DOC] HELPER: Creates linear, linear1, linear2 and linear3 layers.
                super().__init__()
                self.linear = torch.nn.Linear(3, 3)
                self.linear1 = torch.nn.Linear(3, 3)
                self.linear2 = torch.nn.Linear(3, 3)
                self.linear3 = torch.nn.Linear(3, 3)

            def forward(self, x):
                # [FUNCTION DOC] HELPER: Applies the layers to shifted entries of the input pytree
                # and returns a dict with keys y, z, w and v.
                return {
                    "y": self.linear2(x[2] + 1),
                    "z": self.linear3(x[1] - 1),
                    "w": self.linear(x[0]["b"] + 2),
                    "v": self.linear1(x[0]["a"] - 2),
                }

        mod = Module()
        compiled_mod = dynamo_graph_capture_for_export(mod)(
            (
                {"a": torch.randn(3, 3), "b": torch.randn(3, 3)},
                torch.randn(3, 3),
                torch.randn(3, 3),
            )
        )

        inputs = (
            {"a": torch.randn(3, 3), "b": torch.randn(3, 3)},
            torch.randn(3, 3),
            torch.randn(3, 3),
        )
        self.assertEqual(compiled_mod(inputs), mod(inputs))

    def test_dynamic_settings(self):
        # [FUNCTION DOC] Test: compiling with dynamic=False makes the backend see
        # automatic_dynamic_shapes disabled while the global config stays enabled outside;
        # the compiled add still matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        def backend(gm, example_inputs):
            # [FUNCTION DOC] HELPER: Backend that asserts automatic_dynamic_shapes is False
            # during compilation and returns a CustomCompiledFunction.
            self.assertFalse(torch._dynamo.config.automatic_dynamic_shapes)
            return CustomCompiledFunction(gm, example_inputs)

        self.assertTrue(torch._dynamo.config.automatic_dynamic_shapes)
        compiled_fn = torch.compile(
            fn, fullgraph=True, backend=backend, dynamic=False
        ).aot_compile(((torch.randn(3, 4), torch.randn(3, 4)), {}))
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    def test_fullgraph_capture_with_pytree_func(self):
        # [FUNCTION DOC] Test: function version of the pytree capture test; captures a
        # function with a pytree input and dict output and compares with eager.
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        def foo(x):
            # [FUNCTION DOC] HELPER: Returns a dict from tuple entries: y, z, w and v.
            return {
                "y": x[2] + 1,
                "z": x[1] - 1,
                "w": x[0]["b"] + 2,
                "v": x[0]["a"] - 2,
            }

        compiled_foo = dynamo_graph_capture_for_export(foo)(
            (
                {"a": torch.randn(4, 3), "b": torch.randn(3, 2)},
                torch.randn(2, 3),
                torch.randn(3, 4),
            )
        )

        inputs = (
            {"a": torch.randn(4, 3), "b": torch.randn(3, 2)},
            torch.randn(2, 3),
            torch.randn(3, 4),
        )
        self.assertEqual(compiled_foo(inputs), foo(inputs))

    def test_fullgraph_capture_schema_self_arg_no_collision(self):
        # [FUNCTION DOC] Test: regression test; captures aten.where.self (whose schema has
        # self as a non-first argument) and checks the captured function matches the op,
        # guarding against a duplicate self parameter name and a SyntaxError on recompile.
        """Regression: aten op schemas with `self` at non-first position
        (e.g. `aten.where.self(Tensor condition, Tensor self, Tensor other)`)
        must not produce `def forward(self, condition, self, other):` and
        SyntaxError at `graph_module.recompile()`."""
        from torch._dynamo.functional_export import dynamo_graph_capture_for_export

        cond = torch.tensor([True, False, True, False])
        x = torch.tensor(0.0)
        y = torch.tensor([1.0, 2.0, 3.0, 4.0])
        op = torch.ops.aten.where.self
        compiled = dynamo_graph_capture_for_export(op)(cond, x, y)
        self.assertEqual(compiled(cond, x, y), op(cond, x, y))

    def test_aot_compile_with_closure_save_and_load(self):
        # [FUNCTION DOC] Test: AOT compiles a function that closes over a local integer,
        # saves, loads without extra data and checks the loaded function matches eager.
        tmp = 2

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y plus the closed-over value.
            return x + y + tmp

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_with_super_call(self):
        # [FUNCTION DOC] Test: AOT compiles the forward of a model whose forward has a super
        # call (free variable __class__), checks it matches eager, then saves, loads and
        # checks again.
        fn = TestVLLMModel()
        compiled_fn = torch.compile(fn.forward, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4),), {})
        )
        self.assertEqual(fn.forward.__code__.co_freevars, ("__class__",))
        inputs = (torch.randn(3, 4),)
        expected = fn(*inputs)
        actual = compiled_fn(fn, *inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(fn, *inputs)
        self.assertEqual(expected, actual)

    def test_aot_compile_with_global_tensor(self):
        # [FUNCTION DOC] Test: AOT compiles a function that reads the module-level global
        # tensor EPS and checks it matches eager.
        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y plus the global EPS tensor.
            return x + y + EPS

        def make_inputs():
            # [FUNCTION DOC] HELPER: Returns two random (3, 4) tensors.
            return (torch.randn(3, 4), torch.randn(3, 4))

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile((make_inputs(), {}))  # noqa: UNSPECIFIED_BACKEND

        test_inputs = make_inputs()
        self.assertEqual(compiled_fn(*test_inputs), fn(*test_inputs))

    def test_aot_compile_with_default_args(self):
        # [FUNCTION DOC] Test: AOT compiles a function with a defaulted second parameter
        # called with one argument, then saves, loads and checks results match eager.
        def fn(x, y=1):
            # [FUNCTION DOC] HELPER: Returns x plus x; y is unused.
            return x + x

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4),), {})
        )
        inputs = (torch.randn(3, 4),)
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(f)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_aot_compile_with_aoti(self):
        # [FUNCTION DOC] Test (GPU only): compiles an add with aot_compile_fullgraph and the
        # AOTInductor wrapper under a default GPU device, checks against eager, then saves,
        # loads and checks again.
        with torch.device(GPU_TYPE):
            from torch._dynamo.hooks import Hooks

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

            def make_inputs():
                # [FUNCTION DOC] HELPER: Returns two random (3, 4) tensors.
                return (torch.randn(3, 4), torch.randn(3, 4))

            compiled_fn = torch._dynamo.aot_compile.aot_compile_fullgraph(
                fn,
                (make_inputs(), {}),
                Hooks(),
                torch._TorchCompileAOTInductorWrapper(None, None, None),
            )

            test_inputs = make_inputs()
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)
            compiled_fn.save_compiled_function(self.path())
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_aot_compile_with_aoti_module(self):
        # [FUNCTION DOC] Test (GPU only): compiles a Linear module with aot_compile_module
        # and the AOTInductor wrapper, checks forward, serializes, deserializes onto a
        # deep copy of the module and checks that gradients after backward equal the
        # eager gradients.
        with torch.device(GPU_TYPE):
            from torch._dynamo.hooks import Hooks

            mod = SimpleLinearModule()

            def make_inputs():
                # [FUNCTION DOC] HELPER: Returns one random (4, 3) tensor in a tuple.
                return (torch.randn(4, 3),)

            compiled_mod = torch._dynamo.aot_compile.aot_compile_module(
                mod,
                [ModelInput(make_inputs(), {}, [])],
                Hooks(),
                torch._TorchCompileAOTInductorWrapper(None, None, None),
            )

            def get_grads(m: torch.nn.Module):
                # [FUNCTION DOC] HELPER: Returns a mapping from parameter name to its gradient.
                return {name: p.grad for name, p in m.named_parameters()}

            original_mod = copy.deepcopy(mod)
            test_inputs = make_inputs()
            expected = mod(*test_inputs)
            expected.sum().backward()
            expected_grads = get_grads(mod)

            actual = compiled_mod(*test_inputs)
            self.assertEqual(expected, actual)
            serialized = compiled_mod.serialize()
            compiled_fn = AOTCompiledModel.deserialize(original_mod, serialized)
            actual = compiled_fn(*test_inputs)
            actual.sum().backward()
            self.assertEqual(get_grads(original_mod), expected_grads)

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_aot_compile_with_aoti_torch_compile(self):
        # [FUNCTION DOC] Test (GPU only): torch.compile with the use_aoti option and
        # aot_compile; checks it matches eager, saves, loads and checks again, and that the
        # loaded artifacts report the aotinductor backend name.
        with torch.device(GPU_TYPE):

            def fn(x, y):
                # [FUNCTION DOC] HELPER: Returns x plus y.
                return x + y

            def make_inputs():
                # [FUNCTION DOC] HELPER: Returns two random (3, 4) tensors.
                return (torch.randn(3, 4), torch.randn(3, 4))

            compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                fn, fullgraph=True, options={"use_aoti": True}
            ).aot_compile((make_inputs(), {}))
            test_inputs = make_inputs()
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)
            compiled_fn.save_compiled_function(self.path())
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(compiled_fn._artifacts.backend_name, "aotinductor")
            self.assertEqual(expected, actual)

    @unittest.skipIf(not c10d.is_available(), "requires c10d")
    def test_aot_compile_with_redistribute(self):
        # [FUNCTION DOC] Test (requires c10d): under a fake process group of world size 4
        # and a 2x2 CPU mesh, AOT compiles a module forward that redistributes a DTensor,
        # checks against eager, saves, resets, reloads under fail_on_recompile and checks
        # again. Always destroys the process group.
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import DTensor, Replicate
        from torch.testing._internal.distributed.fake_pg import FakeStore

        fake_store = FakeStore()
        torch.distributed.init_process_group(
            "fake", store=fake_store, rank=0, world_size=4
        )
        try:
            mesh = init_device_mesh("cpu", (2, 2), mesh_dim_names=("dp", "tp"))
            input_tensor = torch.randn(32, 32, device="cpu")
            placements = (Replicate(), Replicate())
            d_input_tensor = DTensor.from_local(input_tensor, mesh, placements)
            mod = RedistributeModel()

            compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
                mod,
                fullgraph=True,
            ).forward.aot_compile(((input_tensor, d_input_tensor, mesh), {}))
            inputs = (input_tensor, d_input_tensor, mesh)
            expected = mod(*inputs)
            actual = compiled_fn(mod, *inputs)
            self.assertEqual(expected, actual)
            compiled_fn.save_compiled_function(self.path())
            torch._dynamo.reset()
            with torch.compiler.set_stance("fail_on_recompile"):
                with open(self.path(), "rb") as f:
                    compiled_fn = torch.compiler.load_compiled_function(f)
                actual = compiled_fn(mod, *inputs)
                self.assertEqual(expected, actual)
        finally:
            torch.distributed.destroy_process_group()

    def test_aot_compile_with_captured_module(self):
        # [FUNCTION DOC] Test: a module whose forward was replaced by a compiled wrapper
        # cannot be saved by default (RuntimeError listing the unserializable module); it
        # saves when the module is passed as external_data, loading without it raises
        # Missing required external ref, and loading with it matches eager.
        mod = SimpleLinearModule()

        fn = mod.forward

        def with_processing(f, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls f with the given args and kwargs.
            return f(*args, **kwargs)

        fn = functools.partial(with_processing, fn)

        fn = wrap_forward_function(fn)
        mod.forward = fn

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(4, 3),), {})
        )
        mod.forward = compiled_fn
        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed to serialize the following objects: \[SimpleLinearModule",
        ):
            compiled_fn.save_compiled_function(self.path())
        compiled_fn.save_compiled_function(
            self.path(),
            external_data={"mod": mod},
        )
        with open(self.path(), "rb") as f:
            with self.assertRaisesRegex(RuntimeError, "Missing required external ref"):
                torch.compiler.load_compiled_function(f)

        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(
                f,
                external_data={"mod": mod},
            )
            test_inputs = (torch.randn(4, 3),)
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_with_captured_module_2(self):
        # [FUNCTION DOC] Test: same as the captured module test except the module forward is
        # only reassigned after compiling; verifies the same save, missing external ref and
        # load-with-external_data behavior.
        mod = SimpleLinearModule()

        fn = mod.forward

        def with_processing(f, *args, **kwargs):
            # [FUNCTION DOC] HELPER: Calls f with the given args and kwargs.
            return f(*args, **kwargs)

        fn = functools.partial(with_processing, fn)

        fn = wrap_forward_function(fn)

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(4, 3),), {})
        )
        mod.forward = compiled_fn
        with self.assertRaisesRegex(
            RuntimeError,
            r"Failed to serialize the following objects: \[SimpleLinearModule",
        ):
            compiled_fn.save_compiled_function(self.path())
        compiled_fn.save_compiled_function(
            self.path(),
            external_data={"mod": mod},
        )
        with open(self.path(), "rb") as f:
            with self.assertRaisesRegex(RuntimeError, "Missing required external ref"):
                torch.compiler.load_compiled_function(f)

        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(
                f,
                external_data={"mod": mod},
            )
            test_inputs = (torch.randn(4, 3),)
            expected = fn(*test_inputs)
            actual = compiled_fn(*test_inputs)
            self.assertEqual(expected, actual)

    def test_aot_compile_with_checkpoint(self):
        # [FUNCTION DOC] Test: AOT compiles a function using non-reentrant activation
        # checkpointing, checks against eager, then saves, resets, reloads under
        # fail_on_recompile and checks again.
        from torch.utils.checkpoint import checkpoint

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Wraps compute in a non-reentrant checkpoint call.
            def compute(x, y):
                # [FUNCTION DOC] HELPER: Returns twice x plus three times y.
                return x * 2 + y * 3

            return checkpoint(compute, x, y, use_reentrant=False)

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(  # noqa: UNSPECIFIED_BACKEND
            ((torch.randn(3, 4), torch.randn(3, 4)), {})
        )
        inputs = (torch.randn(3, 4), torch.randn(3, 4))
        expected = fn(*inputs)
        actual = compiled_fn(*inputs)
        self.assertEqual(expected, actual)
        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()
        with torch.compiler.set_stance("fail_on_recompile"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)
            actual = compiled_fn(*inputs)
            self.assertEqual(expected, actual)

    def test_external_refs_validation(self):
        # [FUNCTION DOC] Test: a compiled function returning a MooType instance fails to
        # load without globals (RuntimeError Missing required external ref) and loads
        # correctly when f_globals is the function's globals.
        """Test that external refs tracking and f_globals parameter work correctly"""

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns a MooType holding x plus y.
            return MooType(x + y)

        def make_inputs():
            # [FUNCTION DOC] HELPER: Returns two random (3, 4) tensors.
            return (torch.randn(3, 4), torch.randn(3, 4))

        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile((make_inputs(), {}))  # noqa: UNSPECIFIED_BACKEND
        test_inputs = make_inputs()
        expected = fn(*test_inputs)
        actual = compiled_fn(*test_inputs)
        self.assertEqual(expected.x, actual.x)
        compiled_fn.save_compiled_function(self.path())

        with self.assertRaisesRegex(RuntimeError, "Missing required external ref"):
            with open(self.path(), "rb") as f:
                compiled_fn = torch.compiler.load_compiled_function(f)

        with open(self.path(), "rb") as f:
            compiled_fn = torch.compiler.load_compiled_function(
                f, f_globals=fn.__globals__
            )
        actual = compiled_fn(*test_inputs)
        self.assertEqual(expected.x, actual.x)

    def test_builtins_dict_survives_serialization(self):
        # [FUNCTION DOC] Test: AOT compiles a function that returns a value and the builtin
        # type, saves, loads with the function globals and checks both outputs match eager.
        # The comments in the body mention loading without f_globals, but the load passes
        # them.
        """Test that __builtins_dict__ is preserved through serialize/deserialize."""

        def fn(x):
            # [FUNCTION DOC] HELPER: Returns x plus one together with the builtin type.
            return x + 1, type

        x = torch.randn(4)
        compiled_fn = torch.compile(fn, fullgraph=True).aot_compile(((x,), {}))  # noqa: UNSPECIFIED_BACKEND

        # Save and reload without f_globals
        compiled_fn.save_compiled_function(self.path())
        with open(self.path(), "rb") as f:
            loaded_fn = torch.compiler.load_compiled_function(
                f, f_globals=fn.__globals__
            )

        expected = fn(x)
        actual = loaded_fn(x)
        self.assertEqual(expected[0], actual[0])
        self.assertEqual(expected[1], actual[1])

    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_cross_aot_compile(self):
        # [FUNCTION DOC] Test (GPU only): AOT compiles an add using fake GPU inputs that
        # require grad, saves, resets and reloads, then checks the loaded function output
        # and the gradients from backward against eager.
        """Test cross-compilation using fake tensors and backward correctness"""
        from torch._subclasses.fake_tensor import FakeTensorMode

        def fn(x, y):
            # [FUNCTION DOC] HELPER: Returns x plus y.
            return x + y

        with FakeTensorMode(allow_non_fake_inputs=True):
            fake_inputs = (
                torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
                torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
            )
        compiled_fn = torch.compile(  # noqa: UNSPECIFIED_BACKEND
            fn,
            fullgraph=True,
        ).aot_compile((fake_inputs, {}))

        compiled_fn.save_compiled_function(self.path())
        torch._dynamo.reset()

        with open(self.path(), "rb") as f:
            loaded_fn = torch.compiler.load_compiled_function(f)

        inputs = (
            torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
            torch.randn(3, 4, device=GPU_TYPE, requires_grad=True),
        )
        expected = fn(*inputs)
        actual = loaded_fn(*inputs)
        self.assertEqual(expected, actual)

        # Backward check: compare gradients between eager and loaded compiled function
        eager_loss = expected.sum()
        eager_loss.backward()
        eager_grads = tuple(inp.grad.clone() for inp in inputs)

        # Reset grads for compiled run
        for inp in inputs:
            inp.grad = None

        compiled_out = loaded_fn(*inputs)
        compiled_loss = compiled_out.sum()
        compiled_loss.backward()
        compiled_grads = tuple(inp.grad.clone() for inp in inputs)

        for eg, cg in zip(eager_grads, compiled_grads):
            self.assertEqual(eg, cg)

    @unittest.skipIf(not c10d.is_available(), "requires c10d")
    @unittest.skipIf(not HAS_GPU, "requires gpu")
    def test_cross_compile_realistic_transformer_model(self):
        # [FUNCTION DOC] Test (requires c10d and GPU): cross-compiles a DTensor Transformer
        # on a fake process group (world size 1): captures a graph under fake mode, exports
        # joint descriptors, compiles with regional inductor into a serializable callable,
        # serializes and deserializes it, and compares logits, loss and parameter
        # gradients bitwise against an identically initialized eager model. Always
        # destroys the process group.
        """
        Test cross-compilation with transformer model with DTensors,
        FlexAttention, and checkpointing using the compiler toolkit.
        Compares compiled execution against eager execution for bitwise
        equivalence of logits and gradients.
        """
        from torch.distributed._tensor import DTensor
        from torch.distributed.device_mesh import init_device_mesh
        from torch.distributed.tensor import Placement, Replicate, Shard
        from torch.testing._internal.distributed.fake_pg import FakeStore

        def dtensorify_module(
            module: nn.Module,
            device_mesh,
            *,
            param_placements: list[Placement] | None = None,
            buffer_placements: list[Placement] | None = None,
        ) -> None:
            # [FUNCTION DOC] HELPER: Recursively converts non-DTensor parameters and buffers of
            # the module (and children) to DTensors on the mesh with the given placements
            # (default Replicate), replacing the attributes in place. Returns nothing.
            if param_placements is None:
                param_placements = [Replicate()]
            if buffer_placements is None:
                buffer_placements = [Replicate()]

            for name, p in list(module.named_parameters(recurse=False)):
                if p is None or isinstance(p, DTensor):
                    continue
                dt = DTensor.from_local(p.data, device_mesh, param_placements)
                new_p = nn.Parameter(dt, requires_grad=p.requires_grad)
                setattr(module, name, new_p)

            for name, b in list(module.named_buffers(recurse=False)):
                if b is None or isinstance(b, DTensor):
                    continue
                dt = DTensor.from_local(b, device_mesh, buffer_placements)
                module._buffers[name] = dt

            for child in module.children():
                dtensorify_module(
                    child,
                    device_mesh,
                    param_placements=param_placements,
                    buffer_placements=buffer_placements,
                )

        def init_weights_deterministic(module: nn.Module, seed: int = 42) -> None:
            # [FUNCTION DOC] HELPER: Seeds the CPU and GPU generators, then fills every
            # parameter that requires grad and every buffer (local tensor for DTensors) with
            # normal values with mean zero and std 0.02 in place.
            """
            Initialize module weights deterministically using a fixed seed.
            This ensures reproducible results across eager and compiled runs.
            """
            torch.manual_seed(seed)
            getattr(torch, GPU_TYPE).manual_seed(seed)
            for name, param in module.named_parameters():
                if param.requires_grad:
                    local_param = (
                        param.to_local() if isinstance(param, DTensor) else param
                    )
                    local_param.data.normal_(mean=0.0, std=0.02)
            for name, buf in module.named_buffers():
                local_buf = buf.to_local() if isinstance(buf, DTensor) else buf
                local_buf.data.normal_(mean=0.0, std=0.02)

        fake_store = FakeStore()
        c10d.init_process_group(backend="fake", store=fake_store, rank=0, world_size=1)

        try:
            rank = c10d.get_rank()
            device = torch.device(f"{GPU_TYPE}:{rank}")
            vocab_size = 1000
            embed_dim = 256
            num_heads = 8
            num_kv_heads = 2
            num_layers = 2
            max_seq_len = 32
            batch_size = 2
            seq_len = 16

            device_mesh = init_device_mesh(
                GPU_TYPE,
                (1,),
                mesh_dim_names=("dp",),
            )

            with torch.device("meta"):
                model = Transformer(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_layers,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    device_mesh=device_mesh,
                )

            dtensorify_module(
                model,
                device_mesh,
                param_placements=[Replicate()],
                buffer_placements=[Replicate()],
            )

            outer_fake_mode = FakeTensorMode(allow_non_fake_inputs=True)
            with outer_fake_mode:
                # Convert meta tensors -> fake tensors on target device
                model.to_empty(device=device)

                local_input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )
                input_ids_dt = DTensor.from_local(
                    local_input_ids, device_mesh, [Shard(0)]
                )

            from torch._dynamo.functional_export import dynamo_graph_capture_for_export

            gm = dynamo_graph_capture_for_export(model)(input_ids_dt)

            fake_mode = gm.meta["fake_mode"]

            # Pre-create a temp file path and remove delete=False since we control cleanup
            with (
                tempfile.NamedTemporaryFile(suffix=".pt") as f,
                torch._functorch.config.patch(force_autograd_cache=True),
            ):
                serialization_path = f.name

                with contextlib.ExitStack() as stack:
                    if fake_mode is not None:
                        stack.enter_context(tracing(TracingContext(fake_mode)))
                        stack.enter_context(fake_mode)

                    jd = aot_export_joint_with_descriptors(
                        stack,
                        gm,
                        (input_ids_dt,),
                    )

                    compiled_wrapper = aot_compile_joint_with_descriptors(
                        jd,
                        fw_compiler=regional_inductor,
                        bw_compiler=regional_inductor,
                        serializable=True,
                    )

                    f.write(
                        BundledAOTAutogradSerializableCallable.serialize_compile_artifacts(
                            compiled_wrapper
                        )
                    )
                    f.flush()

                with open(serialization_path, "rb") as f_r:
                    loaded_fn = BundledAOTAutogradSerializableCallable.deserialize_compile_artifacts(
                        f_r.read()
                    )

                # Create compiled model with deterministic initialization
                local_input_ids = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )
                input_ids_dt = DTensor.from_local(
                    local_input_ids, device_mesh, [Shard(0)]
                )
                targets = torch.randint(
                    0, vocab_size, (batch_size, seq_len), device=device
                )

                compiled_model = Transformer(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_layers,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    device_mesh=device_mesh,
                )
                dtensorify_module(
                    compiled_model,
                    device_mesh,
                    param_placements=[Replicate()],
                    buffer_placements=[Replicate()],
                )
                compiled_model.to_empty(device=device)
                init_weights_deterministic(compiled_model)

                eager_model = Transformer(
                    vocab_size,
                    embed_dim,
                    num_heads,
                    num_layers,
                    max_seq_len,
                    num_kv_heads=num_kv_heads,
                    device_mesh=device_mesh,
                )
                dtensorify_module(
                    eager_model,
                    device_mesh,
                    param_placements=[Replicate()],
                    buffer_placements=[Replicate()],
                )
                eager_model.to_empty(device=device)
                init_weights_deterministic(eager_model)

                # Run compiled forward pass
                (compiled_logits_dt,) = loaded_fn(
                    *compiled_model.parameters(),
                    *compiled_model.buffers(),
                    input_ids_dt,
                )
                compiled_logits = (
                    compiled_logits_dt.to_local()
                    if isinstance(compiled_logits_dt, DTensor)
                    else compiled_logits_dt
                )

                # Run eager forward pass with same input
                eager_logits_dt = eager_model(input_ids_dt)
                eager_logits = (
                    eager_logits_dt.to_local()
                    if isinstance(eager_logits_dt, DTensor)
                    else eager_logits_dt
                )

                # Compare logits for bitwise equivalence
                self.assertEqual(
                    compiled_logits,
                    eager_logits,
                    msg="Compiled and eager logits should be bitwise equivalent",
                )

                # Run backward pass on compiled model
                compiled_loss = F.cross_entropy(
                    compiled_logits.view(-1, vocab_size), targets.view(-1)
                )
                compiled_loss.backward()
                compiled_grads = {
                    name: p.grad.clone() if p.grad is not None else None
                    for name, p in compiled_model.named_parameters()
                }

                # Run backward pass on eager model
                eager_loss = F.cross_entropy(
                    eager_logits.view(-1, vocab_size), targets.view(-1)
                )
                eager_loss.backward()
                eager_grads = {
                    name: p.grad.clone() if p.grad is not None else None
                    for name, p in eager_model.named_parameters()
                }

                # Compare losses for bitwise equivalence
                self.assertEqual(
                    compiled_loss,
                    eager_loss,
                    msg="Compiled and eager losses should be bitwise equivalent",
                )

                # Compare gradients for bitwise equivalence
                for name in compiled_grads:
                    self.assertEqual(
                        compiled_grads[name],
                        eager_grads[name],
                        msg=lambda msg: f"{msg}\nGradients for {name} should be bitwise equivalent",
                    )
        finally:
            c10d.destroy_process_group()


class TestTritonKernelSerialization(torch._inductor.test_case.TestCase):
    # [CLASS DOC] Test case for the triton kernel side table serialization helpers. It
    # mutates the global kernel_side_table and resets it after the test.
    """Tests for triton kernel side table serialization."""

    def test_kernel_side_table_serialization_roundtrip(self):
        # [FUNCTION DOC] Test: adds a mock kernel and constant args to kernel_side_table,
        # serializes the kernels, resets the table (lookup then asserts), restores kernels
        # and constant args via the deserialize helper, and checks the lookup returns the
        # underlying function (torch.sin) and the constant args match. Resets the table at
        # the end.
        """
        Test that the kernel_side_table is properly serialized and restored.

        This test verifies that when we serialize the triton kernel side table
        and then clear it (simulating a new process), deserialization properly
        restores the kernels so they can be looked up by index.

        Without this fix, deserialization in a new process would fail with:
            AssertionError: Kernel index X not found in id_to_kernel
        """
        from torch._dynamo.aot_compile_types import (
            _deserialize_triton_kernel,
            _serialize_triton_kernel,
        )
        from torch._higher_order_ops.triton_kernel_wrap import kernel_side_table

        try:
            # Create a mock kernel-like object that mimics triton JITFunction structure.
            # Triton JITFunction has a `fn` attribute pointing to the wrapped function.
            class MockTritonKernel:
                # [CLASS DOC] HELPER: Mock mimicking a triton JITFunction by exposing the wrapped
                # function as attribute fn.
                def __init__(self, fn):
                    # [FUNCTION DOC] HELPER: Stores the wrapped function as attribute fn.
                    self.fn = fn

            # Use a real importable function (torch.sin) as the wrapped function
            mock_kernel = MockTritonKernel(torch.sin)

            # Add the kernel to the side table (this is what dynamo does during tracing)
            kernel_idx = kernel_side_table.add_kernel(mock_kernel)

            # Add some constant args too
            const_args = {"BLOCK_SIZE": 128, "num_warps": 4}
            const_args_idx = kernel_side_table.add_constant_args(const_args)

            # Simulate serialization: capture the kernel side table state
            triton_kernels = {
                idx: _serialize_triton_kernel(kernel)
                for idx, kernel in kernel_side_table.id_to_kernel.items()
            }
            triton_constant_args = dict(kernel_side_table.constant_args)

            # Simulate a new process by clearing the side table
            kernel_side_table.reset_table()

            # Verify the table is empty - looking up the kernel should fail
            with self.assertRaisesRegex(AssertionError, "not found in id_to_kernel"):
                kernel_side_table.get_kernel(kernel_idx)

            # Simulate deserialization: restore the kernel side table
            for idx, kernel_info in triton_kernels.items():
                restored_kernel = _deserialize_triton_kernel(kernel_info)
                kernel_side_table.id_to_kernel[idx] = restored_kernel
                kernel_side_table.kernel_to_id[restored_kernel] = idx

            for idx, args in triton_constant_args.items():
                kernel_side_table.constant_args[idx] = args

            # Now the kernel lookup should succeed
            restored = kernel_side_table.get_kernel(kernel_idx)
            # The restored kernel is torch.sin (the underlying function), not the mock wrapper
            self.assertIs(restored, torch.sin)

            # Constant args should also be restored
            self.assertEqual(
                kernel_side_table.constant_args[const_args_idx], const_args
            )
        finally:
            kernel_side_table.reset_table()


if __name__ == "__main__":
    from torch._dynamo.test_case import run_tests

    run_tests()
